"""
Оценка качества RAG системы через RAGAS для assistant_api.
Использует OpenAI API для RAG и для метрик RAGAS.
"""

import os
import sys
import math
from typing import Any, Dict, List
from pathlib import Path
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

try:
    from datasets import Dataset
    from ragas import evaluate
except ModuleNotFoundError as exc:
    missing_module = getattr(exc, "name", "unknown")
    print(f"[ОШИБКА] Не найден модуль: {missing_module}")
    print("\nСкорее всего, файл запущен НЕ из виртуального окружения проекта.")
    print("Запустите так (из корня проекта):")
    print(r"  .\venv\Scripts\python.exe assistant_api\evaluate_ragas.py")
    print("\nЕсли нужно, переустановите зависимости в venv:")
    print(r"  .\venv\Scripts\python.exe -m pip install -r requirements.txt")
    sys.exit(1)

# Импорт метрик RAGAS (как готовых объектов метрик, без вызова ())
try:
    from ragas.metrics import faithfulness, context_precision, answer_relevancy
except ImportError:
    # Старый fallback: answer_relevance в некоторых версиях
    from ragas.metrics import faithfulness, context_precision, answer_relevance as answer_relevancy

from rag_pipeline import RAGPipeline


# Тестовые вопросы (база: assistant_api/knowledge/hr/, роль hr)
EVALUATION_QUESTIONS = [
    "За сколько дней нужно подать заявку на отпуск через HR-портал?",
    "Как часто проходит обязательное обучение по продукту?",
    "Можно ли пересылать персональные данные клиентов в личный мессенджер?",
    "Куда обращаться по вопросам больничного и справок?",
]


def _create_ragas_embeddings():
    """
    Создание embeddings для метрики Answer Relevancy.

    Источник задаётся через переменную окружения:
    - RAGAS_EMBEDDINGS_PROVIDER=openai      (по умолчанию)
    - RAGAS_EMBEDDINGS_PROVIDER=huggingface (локально, без OpenAI embeddings API)
    """
    provider = os.getenv("RAGAS_EMBEDDINGS_PROVIDER", "openai").strip().lower()

    if provider == "huggingface":
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Для HuggingFace embeddings нужен пакет langchain-community."
            ) from exc

        model_name = os.getenv(
            "RAGAS_HF_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        print(f"   [+] Embeddings provider: HuggingFace ({model_name})")
        return HuggingFaceEmbeddings(model_name=model_name)

    # provider == openai
    try:
        from langchain_openai import OpenAIEmbeddings
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Для OpenAI embeddings нужен пакет langchain-openai."
        ) from exc

    model_name = os.getenv("RAGAS_OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")
    print(f"   [+] Embeddings provider: OpenAI ({model_name})")
    return OpenAIEmbeddings(model=model_name)


def _get_metric_values(result: Any, key: str) -> List[float]:
    """
    Безопасное извлечение значений метрики из результата RAGAS.

    Поддерживает:
    - dict-подобный результат
    - EvaluationResult из ragas 0.4.x
    """
    values = None

    # 1) dict-подобный доступ: result[key]
    try:
        values = result[key]
    except Exception:
        values = None

    # 2) доступ как атрибут: result.faithfulness
    if values is None:
        values = getattr(result, key, None)

    # 3) некоторые версии ragas хранят построчные score в result.scores
    if values is None:
        scores = getattr(result, "scores", None)
        if isinstance(scores, list):
            collected = []
            for row in scores:
                if isinstance(row, dict) and key in row:
                    collected.append(row[key])
            if collected:
                values = collected

    # 4) fallback через DataFrame
    if values is None and hasattr(result, "to_pandas"):
        try:
            df = result.to_pandas()
            if key in df.columns:
                values = df[key].tolist()
        except Exception:
            values = None

    if values is None:
        return []

    return [v for v in values if not (isinstance(v, float) and math.isnan(v))]


def prepare_dataset(pipeline: RAGPipeline, questions: list) -> Dataset:
    """
    Подготовка датасета для RAGAS из вопросов.
    
    Args:
        pipeline: RAG pipeline для получения ответов
        questions: список вопросов для оценки
    
    Returns:
        Dataset для RAGAS с полями: question, answer, contexts, ground_truth
    """
    questions_list = []
    answers_list = []
    contexts_list = []
    ground_truths_list = []
    
    print("[*] Получение ответов от RAG системы...\n")
    
    for i, question in enumerate(questions, 1):
        print(f"  {i}/{len(questions)}: {question}")
        
        # Получаем ответ от RAG системы (без использования кеша)
        result = pipeline.query(question, use_cache=False)
        
        # Формируем данные для RAGAS
        questions_list.append(question)
        answers_list.append(result["answer"])
        
        # Контекст - список текстов из найденных документов
        context_texts = [doc["text"] for doc in result["context_docs"]]
        contexts_list.append(context_texts)
        
        # Ground truth - эталонный ответ (для демонстрации используем часть ответа)
        # В реальном проекте здесь должны быть вручную подготовленные эталонные ответы
        ground_truths_list.append(result["answer"][:100])
        
        print(f"     [+] Ответ получен от OpenAI API")
    
    print()
    
    # Создаём датасет для RAGAS
    dataset_dict = {
        "question": questions_list,
        "answer": answers_list,
        "contexts": contexts_list,
        "ground_truth": ground_truths_list
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    return dataset


def evaluate_rag_system():
    """
    Основная функция оценки RAG-системы через RAGAS.
    
    Процесс:
    1. Инициализация RAG pipeline
    2. Генерация ответов на тестовые вопросы
    3. Подготовка датасета для RAGAS
    4. Запуск оценки метрик
    5. Вывод результатов
    """
    print("=" * 70)
    print("ОЦЕНКА КАЧЕСТВА RAG (роль HR, OpenAI) ЧЕРЕЗ RAGAS")
    print("=" * 70)
    print()
    
    # Проверка наличия API ключа
    if not os.getenv("OPENAI_API_KEY"):
        print("[ОШИБКА] OPENAI_API_KEY не установлен")
        print("\nУстановите переменную окружения:")
        print("  Windows (PowerShell): $env:OPENAI_API_KEY='your-key'")
        print("  Windows (CMD): set OPENAI_API_KEY=your-key")
        print("  Linux/Mac: export OPENAI_API_KEY='your-key'")
        print("\nИли создайте файл .env в корне проекта с содержимым:")
        print("  OPENAI_API_KEY=your-key-here")
        sys.exit(1)
    
    # Инициализация RAG pipeline
    try:
        print("[*] Инициализация RAG системы (API mode)...\n")
        pipeline = RAGPipeline(role="hr")
        print("\n[OK] RAG система готова к оценке\n")
    except Exception as e:
        print(f"[ОШИБКА] Ошибка инициализации RAG pipeline: {e}")
        sys.exit(1)
    
    # Подготовка датасета
    print("=" * 70)
    dataset = prepare_dataset(pipeline, EVALUATION_QUESTIONS)
    print("=" * 70)
    
    print("\n[*] Запуск оценки метрик RAGAS...")
    print("   Метрики: Faithfulness, Context Precision, Answer Relevancy")
    print("   (это займёт 1-2 минуты, так как RAGAS использует OpenAI для оценки)\n")

    # Используем готовые объекты метрик RAGAS.
    # В текущей версии ragas вызов faithfulness() приводит к ошибке "module is not callable".
    print("   [+] Подготовка метрик RAGAS")
    metrics_to_use = [faithfulness, context_precision, answer_relevancy]

    # Для Answer Relevancy нужны embeddings
    embeddings = _create_ragas_embeddings()
    print()
    
    # Запускаем оценку RAGAS
    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics_to_use,
            embeddings=embeddings
        )
    except Exception as e:
        print(f"[ОШИБКА] Ошибка при оценке: {e}")
        print("\nПодсказка:")
        print("  Если OpenAI embeddings недоступны, попробуйте локальные:")
        print("  PowerShell: $env:RAGAS_EMBEDDINGS_PROVIDER='huggingface'")
        sys.exit(1)
    
    # Обработка и вывод результатов
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 70)
    
    # Вычисляем средние значения метрик (игнорируя NaN)
    faithfulness_values = _get_metric_values(result, "faithfulness")
    context_precision_values = _get_metric_values(result, "context_precision")
    answer_relevancy_values = _get_metric_values(result, "answer_relevancy")
    if not answer_relevancy_values:
        # fallback на старое имя ключа, если встретится
        answer_relevancy_values = _get_metric_values(result, "answer_relevance")
    
    avg_faithfulness = (
        sum(faithfulness_values) / len(faithfulness_values) 
        if faithfulness_values else 0
    )
    avg_context_precision = (
        sum(context_precision_values) / len(context_precision_values) 
        if context_precision_values else 0
    )
    avg_answer_relevancy = (
        sum(answer_relevancy_values) / len(answer_relevancy_values)
        if answer_relevancy_values else 0
    )
    
    # Выводим общие метрики
    print()
    print("[МЕТРИКИ] Средние значения:")
    print(f"   Faithfulness (точность ответа):          {avg_faithfulness:.4f}")
    print(f"   Context Precision (точность контекста):  {avg_context_precision:.4f}")
    print(f"   Answer Relevancy (релевантность ответа): {avg_answer_relevancy:.4f}")
    
    # Вычисляем и выводим средний балл
    avg_score = (avg_faithfulness + avg_context_precision + avg_answer_relevancy) / 3
    print(f"\n{'-'*70}")
    print(f"[ИТОГО] Средний балл: {avg_score:.4f}")
    
    # Оценка качества системы
    if avg_score >= 0.7:
        print("   Оценка: Отличное качество! [OK]")
        print("   Система показывает высокую точность и релевантность ответов.")
    elif avg_score >= 0.5:
        print("   Оценка: Удовлетворительное качество [!]")
        print("   Рекомендуется улучшить качество документов или промптов.")
    else:
        print("   Оценка: Требует значительного улучшения [X]")
        print("   Необходимо пересмотреть стратегию chunking или качество данных.")
    
    # Выводим детали по каждому вопросу
    print("\n" + "=" * 70)
    print("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ПО ВОПРОСАМ")
    print("=" * 70)
    
    for i, question in enumerate(EVALUATION_QUESTIONS):
        print(f"\n{i+1}. {question}")
        
        # Faithfulness
        faith_val = result['faithfulness'][i]
        if not (isinstance(faith_val, float) and math.isnan(faith_val)):
            print(f"   Faithfulness:       {faith_val:.4f}")
        else:
            print(f"   Faithfulness:       не удалось вычислить")
        
        # Context Precision
        cp_val = result['context_precision'][i]
        if not (isinstance(cp_val, float) and math.isnan(cp_val)):
            print(f"   Context Precision:  {cp_val:.4f}")
        else:
            print(f"   Context Precision:  не удалось вычислить")

        # Answer Relevancy
        ar_values = _get_metric_values(result, "answer_relevancy")
        if not ar_values:
            ar_values = _get_metric_values(result, "answer_relevance")
        ar_val = ar_values[i] if i < len(ar_values) else float("nan")
        if not (isinstance(ar_val, float) and math.isnan(ar_val)):
            print(f"   Answer Relevancy:   {ar_val:.4f}")
        else:
            print(f"   Answer Relevancy:   не удалось вычислить")
    
    # Пояснения к метрикам
    print("\n" + "=" * 70)
    print("[INFO] ПОЯСНЕНИЯ К МЕТРИКАМ")
    print("=" * 70)
    print("""
Faithfulness (Точность ответа):
  Измеряет, насколько ответ соответствует предоставленному контексту.
  Значения: 0.0 - 1.0 (1.0 = полное соответствие контексту)

Context Precision (Точность контекста):
  Измеряет качество извлечённого контекста для ответа на вопрос.
  Значения: 0.0 - 1.0 (1.0 = идеальный контекст)

Answer Relevancy (Релевантность ответа):
  Измеряет, насколько ответ прямо отвечает на вопрос пользователя.
  Значения: 0.0 - 1.0 (1.0 = максимально релевантный ответ)

ПРИМЕЧАНИЕ:
  Если OpenAI embeddings недоступны в вашем регионе, используйте:
  RAGAS_EMBEDDINGS_PROVIDER=huggingface
    """)
    
    print("=" * 70)
    print("[OK] Оценка завершена!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    evaluate_rag_system()

