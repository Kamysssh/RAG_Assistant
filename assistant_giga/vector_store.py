"""
Модуль работы с векторным хранилищем ChromaDB для GigaChat.
Обрабатывает загрузку документов, chunking и поиск по векторам.
"""

import chromadb
from typing import List, Dict, Any
import os
from gigachat_client import GigaChatClient
from pathlib import Path
from datetime import datetime
import shutil


class VectorStore:
    """Векторное хранилище на основе ChromaDB с GigaChat embeddings."""
    
    def __init__(self, collection_name: str = "rag_collection", persist_directory: str = None):
        """
        Инициализация векторного хранилища.
        
        Args:
            collection_name: имя коллекции в ChromaDB
            persist_directory: директория для хранения данных
        """
        self.collection_name = collection_name
        if persist_directory is None:
            persist_directory = str(Path(__file__).resolve().parent / "chroma_db")
        self.persist_directory = persist_directory
        
        # Инициализация ChromaDB клиента
        self.client = self._init_client_with_recovery()
        
        # Получение или создание коллекции
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Коллекция '{collection_name}' загружена. Документов: {self.collection.count()}")
        except Exception:
            self.collection = self._create_collection()
            print(f"Создана новая коллекция '{collection_name}'")
        
        # GigaChat клиент для создания embeddings
        self.gigachat_client = GigaChatClient()

    def _create_collection(self):
        """Создание новой коллекции с базовыми настройками."""
        return self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def _init_client_with_recovery(self):
        """
        Инициализация клиента ChromaDB с авто-восстановлением.
        Иногда локальная БД может повредиться и Rust backend падает с PanicException.
        """
        try:
            return chromadb.PersistentClient(path=self.persist_directory)
        except BaseException as exc:
            persist_path = Path(self.persist_directory)
            backup_path = persist_path.with_name(
                f"{persist_path.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            if persist_path.exists():
                shutil.move(str(persist_path), str(backup_path))
                print(
                    f"[WARN] Локальная база ChromaDB повреждена, создан бэкап: {backup_path}"
                )
            else:
                print("[WARN] Не удалось открыть ChromaDB, выполняется чистая инициализация")

            try:
                return chromadb.PersistentClient(path=self.persist_directory)
            except BaseException as second_exc:
                # Дополнительный fallback: уходим в новую директорию.
                # Это помогает, если после panic внутреннее состояние Rust backend
                # для старого пути остаётся неконсистентным в текущем процессе.
                try:
                    fallback_path = persist_path.with_name(
                        f"{persist_path.name}_recovered_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )
                    fallback_path.mkdir(parents=True, exist_ok=True)
                    self.persist_directory = str(fallback_path)
                    print(f"[WARN] Переключение на новую директорию ChromaDB: {self.persist_directory}")
                    return chromadb.PersistentClient(
                        path=self.persist_directory
                    )
                except Exception as third_exc:
                    raise RuntimeError(
                        "Не удалось инициализировать ChromaDB после восстановления: "
                        f"{third_exc}"
                    ) from exc

    @staticmethod
    def _read_text_with_fallbacks(file_path: Path) -> str:
        """
        Чтение текста с подбором кодировки.
        Нужен для документов, сохраненных не в UTF-8.
        """
        encodings = ["utf-8", "utf-8-sig", "cp1251", "windows-1251"]
        for encoding in encodings:
            try:
                return file_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue

        # Последний fallback: читаем с заменой битых символов,
        # чтобы процесс индексации не падал.
        return file_path.read_text(encoding="utf-8", errors="replace")
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        Умное разбиение текста на чанки с учётом семантики.
        
        Стратегия:
        1. Приоритет абзацам (разделение по \n\n)
        2. Разбиение длинных абзацев по предложениям
        3. Сохранение контекста через overlap
        4. Учёт минимального и максимального размера чанка
        
        Args:
            text: исходный текст
            chunk_size: целевой размер чанка в символах
            overlap: размер перекрытия между чанками
            
        Returns:
            список чанков
        """
        # Разделяем текст на абзацы
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Если абзац помещается в текущий чанк
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            
            # Если текущий чанк не пустой и добавление абзаца превысит размер
            elif current_chunk:
                chunks.append(current_chunk)
                # Добавляем overlap из конца предыдущего чанка
                overlap_text = self._get_overlap_text(current_chunk, overlap)
                current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
            
            # Если абзац слишком большой, разбиваем его на предложения
            else:
                if len(paragraph) > chunk_size:
                    # Разбиваем длинный абзац на предложения
                    sentence_chunks = self._split_long_paragraph(paragraph, chunk_size, overlap)
                    
                    # Добавляем все чанки кроме последнего
                    if sentence_chunks:
                        chunks.extend(sentence_chunks[:-1])
                        current_chunk = sentence_chunks[-1]
                else:
                    current_chunk = paragraph
        
        # Добавляем последний чанк
        if current_chunk:
            chunks.append(current_chunk)
        
        # Пост-обработка: фильтруем слишком короткие чанки
        chunks = [chunk for chunk in chunks if len(chunk) >= 50]
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """
        Получение текста для overlap из конца предыдущего чанка.
        Пытается взять целые предложения.
        
        Args:
            text: текст для извлечения overlap
            overlap_size: желаемый размер overlap
            
        Returns:
            текст overlap
        """
        if len(text) <= overlap_size:
            return text
        
        # Берём последние overlap_size символов
        overlap_candidate = text[-overlap_size:]
        
        # Ищем начало предложения в overlap
        sentence_starts = ['. ', '! ', '? ', '\n']
        best_start = 0
        
        for delimiter in sentence_starts:
            pos = overlap_candidate.find(delimiter)
            if pos != -1 and pos > best_start:
                best_start = pos + len(delimiter)
        
        if best_start > 0:
            return overlap_candidate[best_start:].strip()
        
        return overlap_candidate.strip()
    
    def _split_long_paragraph(self, paragraph: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Разбиение длинного абзаца на чанки по предложениям.
        
        Args:
            paragraph: абзац для разбиения
            chunk_size: целевой размер чанка
            overlap: размер перекрытия
            
        Returns:
            список чанков
        """
        # Разделяем на предложения
        import re
        sentences = re.split(r'([.!?]+\s+)', paragraph)
        
        # Собираем предложения обратно с их разделителями
        full_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                full_sentences.append(sentences[i] + sentences[i + 1])
            else:
                full_sentences.append(sentences[i])
        
        # Если осталось что-то в конце без разделителя
        if len(sentences) % 2 == 1:
            full_sentences.append(sentences[-1])
        
        chunks = []
        current_chunk = ""
        
        for sentence in full_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Если предложение помещается в текущий чанк
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                # Сохраняем текущий чанк
                if current_chunk:
                    chunks.append(current_chunk)
                    # Добавляем overlap
                    overlap_text = self._get_overlap_text(current_chunk, overlap)
                    current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                else:
                    # Если одно предложение больше chunk_size, всё равно добавляем его
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def load_documents(self, file_path: str, force_reload: bool = False):
        """
        Загрузка документов из файла в векторное хранилище.
        
        Args:
            file_path: путь к файлу с документами
        """
        # Проверка, не загружены ли уже документы
        if self.collection.count() > 0 and not force_reload:
            print("Документы уже загружены в коллекцию")
            return
        if force_reload and self.collection.count() > 0:
            print("Обнаружена существующая коллекция. Выполняется переиндексация...")
            self.client.delete_collection(name=self.collection_name)
            self.collection = self._create_collection()

        # Нормализуем путь относительно директории текущего проекта
        input_path = Path(file_path)
        if not input_path.is_absolute():
            input_path = Path(__file__).resolve().parent / input_path

        # Поддержка как одного файла, так и директории с *.txt
        if not input_path.exists():
            raise FileNotFoundError(f"Путь {input_path} не найден")

        if input_path.is_dir():
            source_files = sorted(input_path.glob("*.txt"))
            if not source_files:
                raise FileNotFoundError(f"В директории {input_path} не найдено .txt файлов")
        else:
            source_files = [input_path]

        chunks_with_source = []
        for source_file in source_files:
            content = self._read_text_with_fallbacks(source_file).strip()
            if not content:
                continue
            file_chunks = self._chunk_text(content)
            for chunk in file_chunks:
                chunks_with_source.append(f"Источник: {source_file.name}\n{chunk}")

        if not chunks_with_source:
            raise ValueError(f"В источнике {input_path} нет текста для индексации")
        print(f"Загружено файлов: {len(source_files)}. Подготовлено {len(chunks_with_source)} чанков")
        
        # Создание embeddings и добавление в ChromaDB
        documents = []
        ids = []
        embeddings = []
        
        for i, chunk in enumerate(chunks_with_source):
            # Создание embedding через OpenAI
            embedding = self._create_embedding(chunk)
            
            documents.append(chunk)
            ids.append(f"doc_{i}")
            embeddings.append(embedding)
            
            if (i + 1) % 10 == 0:
                print(f"Обработано {i + 1}/{len(chunks_with_source)} чанков")
        
        # Добавление в ChromaDB батчами
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids
        )
        
        print(f"Загружено {len(chunks_with_source)} документов в коллекцию '{self.collection_name}'")
    
    def _create_embedding(self, text: str) -> List[float]:
        """
        Создание векторного представления текста через GigaChat.
        
        Args:
            text: текст для векторизации
            
        Returns:
            вектор embeddings
        """
        embeddings = self.gigachat_client.get_embeddings([text])
        return embeddings[0]
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Поиск релевантных документов по запросу.
        
        Args:
            query: текст запроса
            top_k: количество документов для возврата
            
        Returns:
            список документов с метаданными
        """
        # Создание embedding для запроса
        query_embedding = self._create_embedding(query)
        
        # Поиск в ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Форматирование результатов
        documents = []
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                documents.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
        
        return documents
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Получение статистики коллекции.
        
        Returns:
            словарь со статистикой
        """
        return {
            'name': self.collection_name,
            'count': self.collection.count(),
            'persist_directory': self.persist_directory
        }


if __name__ == "__main__":
    # Тестирование векторного хранилища
    import sys
    
    if not os.getenv("GIGACHAT_AUTH_KEY") or not os.getenv("GIGACHAT_RQUID"):
        print("Ошибка: установите переменные GIGACHAT_AUTH_KEY и GIGACHAT_RQUID")
        sys.exit(1)
    
    vector_store = VectorStore(collection_name="test_collection")
    
    # Загрузка документов
    if os.path.exists("data"):
        vector_store.load_documents("data")
    
    # Поиск
    results = vector_store.search("Когда сотруднику не возмещают поездку на корпоративном такси?", top_k=3)
    print("\nРезультаты поиска:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc['text'][:200]}...")
        print(f"   Distance: {doc['distance']}")
    
    # Статистика
    stats = vector_store.get_collection_stats()
    print(f"\nСтатистика: {stats}")

