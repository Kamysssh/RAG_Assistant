"""
Основной RAG pipeline для GigaChat.
Управляет потоком: запрос → кеш → vector search → GigaChat → ответ → кеш.
"""

from typing import Dict, Any, List
import os
from pathlib import Path
from gigachat_client import GigaChatClient

from vector_store import VectorStore
from cache import RAGCache


class RAGPipeline:
    """Основной pipeline для RAG системы с GigaChat."""
    
    def __init__(self, 
                 collection_name: str = "rag_collection",
                 cache_db_path: str = "rag_cache.db",
                 data_file: str = "data",
                 model: str = "GigaChat"):
        """
        Инициализация RAG pipeline.
        
        Args:
            collection_name: имя коллекции в ChromaDB
            cache_db_path: путь к базе данных кеша
            data_file: путь к файлу с документами
            model: модель GigaChat для генерации ответов
        """
        # Проверка ключей
        if not os.getenv("GIGACHAT_AUTH_KEY"):
            raise ValueError("GIGACHAT_AUTH_KEY не установлен")
        if not os.getenv("GIGACHAT_RQUID"):
            raise ValueError("GIGACHAT_RQUID не установлен")
        
        self.model = model
        self.gigachat_client = GigaChatClient()
        project_dir = Path(__file__).resolve().parent
        resolved_data_file = str((project_dir / data_file).resolve()) if not Path(data_file).is_absolute() else data_file
        resolved_cache_db_path = str((project_dir / cache_db_path).resolve()) if not Path(cache_db_path).is_absolute() else cache_db_path
        force_reindex = os.getenv("RAG_FORCE_REINDEX", "0").strip().lower() in {"1", "true", "yes", "y"}
        
        # Инициализация компонентов
        print("Инициализация векторного хранилища...")
        self.vector_store = VectorStore(collection_name=collection_name)
        
        # Загрузка документов: при необходимости принудительно переиндексируем
        if self.vector_store.collection.count() == 0 or force_reindex:
            print(f"Загрузка документов из {resolved_data_file}...")
            self.vector_store.load_documents(resolved_data_file, force_reload=force_reindex)
        
        print("Инициализация кеша...")
        self.cache = RAGCache(db_path=resolved_cache_db_path)
        
        print("RAG Pipeline инициализирован (GigaChat)")
    
    
    def _create_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Создание промпта для LLM с контекстом.
        
        Args:
            query: вопрос пользователя
            context_docs: релевантные документы из векторного хранилища
            
        Returns:
            сформированный промпт
        """
        # Формирование контекста из документов
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            context_parts.append(f"Документ {i}:\n{doc['text']}\n")
        
        context = "\n".join(context_parts)
        
        # Создание промпта
        prompt = f"""Ты - полезный AI ассистент. Ответь на вопрос пользователя на основе предоставленного контекста.

Контекст:
{context}

Вопрос: {query}

Инструкции:
- Отвечай только на основе предоставленного контекста
- Если в контексте нет информации для ответа, скажи об этом
- Будь точным и кратким
- Отвечай на русском языке

Ответ:"""
        
        return prompt
    
    def _generate_answer(self, prompt: str) -> str:
        """
        Генерация ответа через GigaChat.
        
        Args:
            prompt: промпт для модели
            
        Returns:
            сгенерированный ответ
        """
        messages = [
            {"role": "system", "content": "Ты - полезный AI ассистент, который отвечает на вопросы на основе предоставленного контекста."},
            {"role": "user", "content": prompt}
        ]
        
        answer = self.gigachat_client.chat_completion(
            messages=messages,
            model=self.model,
            temperature=0.3,  # Низкая температура для более точных ответов
            max_tokens=500
        )
        
        return answer.strip()
    
    def query(self, user_query: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Основной метод для обработки запроса пользователя.
        
        Поток:
        1. Проверка кеша
        2. Если в кеше нет - поиск в векторном хранилище
        3. Формирование промпта с контекстом
        4. Генерация ответа через LLM
        5. Сохранение в кеш
        
        Args:
            user_query: запрос пользователя
            use_cache: использовать ли кеш
            
        Returns:
            словарь с ответом и метаданными
        """
        print(f"\n{'='*60}")
        print(f"Запрос: {user_query}")
        print(f"{'='*60}")
        
        # Шаг 1: Проверка кеша
        if use_cache:
            print("[*] Проверка кеша...")
            cached_result = self.cache.get(user_query)
            
            if cached_result:
                print("[OK] Ответ найден в кеше")
                return {
                    "query": user_query,
                    "answer": cached_result["answer"],
                    "from_cache": True,
                    "context_docs": cached_result.get("context"),
                    "cached_at": cached_result.get("created_at")
                }
            else:
                print("[-] Ответ не найден в кеше")
        
        # Шаг 2: Поиск релевантных документов
        print("[*] Поиск релевантных документов...")
        context_docs = self.vector_store.search(user_query, top_k=3)
        print(f"[OK] Найдено {len(context_docs)} релевантных документов")
        
        # Шаг 3: Формирование промпта
        print("[*] Формирование промпта...")
        prompt = self._create_prompt(user_query, context_docs)
        
        # Шаг 4: Генерация ответа
        print(f"[*] Генерация ответа через {self.model}...")
        answer = self._generate_answer(prompt)
        print("[OK] Ответ получен")
        
        # Шаг 5: Сохранение в кеш
        if use_cache:
            print("[*] Сохранение в кеш...")
            context_for_cache = [doc['text'] for doc in context_docs]
            self.cache.set(user_query, answer, context_for_cache)
            print("[OK] Сохранено в кеш")
        
        return {
            "query": user_query,
            "answer": answer,
            "from_cache": False,
            "context_docs": context_docs,
            "model": self.model
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики системы.
        
        Returns:
            словарь со статистикой
        """
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "cache": self.cache.get_stats(),
            "model": self.model
        }


if __name__ == "__main__":
    # Тестирование RAG pipeline
    import sys
    
    try:
        pipeline = RAGPipeline()
        
        # Тестовые запросы
        test_queries = [
            "Когда выплачивается премия за успешную рекомендацию кандидата?",
            "Какие элементы одежды запрещены по дресс-коду?",
            "Можно ли ездить на такси из дома в офис за счет компании?"
        ]
        
        for query in test_queries:
            result = pipeline.query(query)
            print(f"\n{'='*60}")
            print(f"Вопрос: {result['query']}")
            print(f"Из кеша: {result['from_cache']}")
            print(f"Ответ: {result['answer']}")
            print(f"{'='*60}\n")
        
        # Повторный запрос (должен быть из кеша)
        print("\n--- Повторный запрос ---")
        result = pipeline.query(test_queries[0])
        print(f"Из кеша: {result['from_cache']}")
        
        # Статистика
        stats = pipeline.get_stats()
        print(f"\nСтатистика системы:")
        print(f"Векторное хранилище: {stats['vector_store']}")
        print(f"Кеш: {stats['cache']}")
        
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)

