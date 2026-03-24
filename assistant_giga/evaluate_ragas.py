"""
Оценка качества RAG системы через RAGAS для assistant_giga.
Использует GigaChat для RAG и OpenAI/HuggingFace embeddings для метрик RAGAS.
"""

import os
import sys
import math
from typing import Any, List
from pathlib import Path
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

try:
    from datasets import Dataset
    from ragas import RunConfig, evaluate
except ModuleNotFoundError as exc:
    missing_module = getattr(exc, "name", "unknown")
    print(f"[ОШИБКА] Не найден модуль: {missing_module}")
    print("\nСкорее всего, файл запущен НЕ из виртуального окружения проекта.")
    print("Запустите так (из корня проекта):")
    print(r"  .\venv\Scripts\python.exe assistant_giga\evaluate_ragas.py")
    print("\nЕсли нужно, переустановите зависимости в venv:")
    print(r"  .\venv\Scripts\python.exe -m pip install -r requirements.txt")
    sys.exit(1)

# Импорт метрик RAGAS (как готовых объектов метрик, без вызова ())
try:
    from ragas.metrics import faithfulness, context_precision, answer_relevancy
except ImportError:
    from ragas.metrics import faithfulness, context_precision, answer_relevance as answer_relevancy

from rag_pipeline import RAGPipeline


EVALUATION_QUESTIONS = [
    "Когда выплачивается премия за успешную рекомендацию кандидата?",
    "Какие сотрудники не могут претендовать на премию за рекомендацию?",
    "Какая одежда считается недопустимой по дресс-коду?",
    "В каких случаях допускается использование корпоративного такси?",
    "Можно ли использовать корпоративное такси для поездок из дома на работу?",
    "Какие данные должны быть в заявке на корпоративное такси?",
]


def _create_ragas_embeddings():
    """
    Создание embeddings для метрики Answer Relevancy.

    Источник задаётся через переменную окружения:
    - RAGAS_EMBEDDINGS_PROVIDER=openai
    - RAGAS_EMBEDDINGS_PROVIDER=huggingface
    """
    # Для GigaChat-проекта безопаснее default=HuggingFace,
    # чтобы не зависать на OpenAI в неподдерживаемом регионе.
    provider = os.getenv("RAGAS_EMBEDDINGS_PROVIDER", "huggingface").strip().lower()

    if provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        model_name = os.getenv(
            "RAGAS_HF_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        print(f"   [+] Embeddings provider: HuggingFace ({model_name})")
        return HuggingFaceEmbeddings(model_name=model_name)

    from langchain_openai import OpenAIEmbeddings

    model_name = os.getenv("RAGAS_OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")
    print(f"   [+] Embeddings provider: OpenAI ({model_name})")
    return OpenAIEmbeddings(model=model_name)


def _get_metric_values(result: Any, key: str) -> List[float]:
    """Безопасное извлечение значений метрики из результата RAGAS."""
    values = None

    try:
        values = result[key]
    except Exception:
        values = None

    if values is None:
        values = getattr(result, key, None)

    if values is None:
        scores = getattr(result, "scores", None)
        if isinstance(scores, list):
            collected = []
            for row in scores:
                if isinstance(row, dict) and key in row:
                    collected.append(row[key])
            if collected:
                values = collected

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


def _preflight_check_ragas_llm() -> None:
    """
    Быстрая проверка доступности LLM для RAGAS-метрик.

    В используемой конфигурации метрики Faithfulness/Context Precision/Answer Relevancy
    требуют LLM-оценщик. На практике здесь используется OpenAI через ragas.
    Если OpenAI недоступен (например, 403 по региону), завершаем заранее,
    чтобы не "висеть" много минут на ретраях.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("[ОШИБКА] Для запуска RAGAS-метрик нужен OPENAI_API_KEY (LLM-оценщик).")
        print("Сейчас ключ отсутствует, поэтому оценка не может быть выполнена.")
        sys.exit(1)

    model_name = os.getenv("RAGAS_OPENAI_EVAL_MODEL", "gpt-4o-mini").strip()

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=12, max_retries=0)
        client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
    except Exception as exc:
        error_text = str(exc)
        if "unsupported_country_region_territory" in error_text:
            print("[ОШИБКА] OpenAI недоступен в вашем регионе (403).")
            print("Из-за этого RAGAS-оценка не запускается и раньше выглядела как зависание.")
            print("Что можно сделать:")
            print("  1) Запускать оценку там, где OpenAI доступен;")
            print("  2) Или оставить только ручное тестирование без RAGAS.")
            sys.exit(1)
        print(f"[ОШИБКА] Предпроверка OpenAI для RAGAS не прошла: {exc}")
        sys.exit(1)


def prepare_dataset(pipeline: RAGPipeline, questions: list) -> Dataset:
    """Подготовка датасета для RAGAS из вопросов."""
    questions_list = []
    answers_list = []
    contexts_list = []
    ground_truths_list = []

    print("[*] Получение ответов от RAG системы...\n")

    for i, question in enumerate(questions, 1):
        print(f"  {i}/{len(questions)}: {question}")
        result = pipeline.query(question, use_cache=False)

        questions_list.append(question)
        answers_list.append(result["answer"])
        contexts_list.append([doc["text"] for doc in result["context_docs"]])
        ground_truths_list.append(result["answer"][:100])

        print("     [+] Ответ получен от GigaChat")

    print()

    dataset_dict = {
        "question": questions_list,
        "answer": answers_list,
        "contexts": contexts_list,
        "ground_truth": ground_truths_list,
    }
    return Dataset.from_dict(dataset_dict)


def evaluate_rag_system():
    """Основная функция оценки RAG-системы через RAGAS."""
    print("=" * 70)
    print("ОЦЕНКА КАЧЕСТВА RAG-СИСТЕМЫ (GIGACHAT MODE) ЧЕРЕЗ RAGAS")
    print("=" * 70)
    print()

    # Проверка ключей для самого GigaChat проекта
    if not os.getenv("GIGACHAT_AUTH_KEY") or not os.getenv("GIGACHAT_RQUID"):
        print("[ОШИБКА] GIGACHAT_AUTH_KEY/GIGACHAT_RQUID не установлены")
        print("Укажите переменные в .env и повторите запуск.")
        sys.exit(1)

    # Для GigaChat-проекта default по embeddings: huggingface
    provider = os.getenv("RAGAS_EMBEDDINGS_PROVIDER", "huggingface").strip().lower()
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("[ОШИБКА] Для RAGAS embeddings provider=openai нужен OPENAI_API_KEY")
        print("Либо добавьте OPENAI_API_KEY в .env, либо переключите provider:")
        print("  PowerShell: $env:RAGAS_EMBEDDINGS_PROVIDER='huggingface'")
        sys.exit(1)

    # Предпроверка LLM-оценщика RAGAS, чтобы избежать долгого "зависания" на ретраях.
    print("[*] Быстрая предпроверка доступа к LLM-оценщику RAGAS...")
    _preflight_check_ragas_llm()
    print("[OK] Предпроверка LLM пройдена\n")

    try:
        print("[*] Инициализация RAG системы (GigaChat mode)...\n")
        pipeline = RAGPipeline(
            collection_name="gigachat_rag_collection",
            cache_db_path="gigachat_rag_cache.db",
            data_file="data",
            model="GigaChat",
        )
        print("\n[OK] RAG система готова к оценке\n")
    except Exception as e:
        print(f"[ОШИБКА] Ошибка инициализации RAG pipeline: {e}")
        sys.exit(1)

    print("=" * 70)
    dataset = prepare_dataset(pipeline, EVALUATION_QUESTIONS)
    print("=" * 70)

    print("\n[*] Запуск оценки метрик RAGAS...")
    print("   Метрики: Faithfulness, Context Precision, Answer Relevancy")
    print("   (это займёт 1-2 минуты)\n")

    print("   [+] Подготовка метрик RAGAS")
    metrics_to_use = [faithfulness, context_precision, answer_relevancy]
    embeddings = _create_ragas_embeddings()
    print()

    try:
        # Уменьшаем число ретраев RAGAS, чтобы не было "зависаний" на сетевых ошибках.
        # В дефолте ragas делает до 10 ретраев и это может занимать много минут.
        run_config = RunConfig(timeout=45, max_retries=1, max_wait=3, max_workers=4)
        result = evaluate(
            dataset=dataset,
            metrics=metrics_to_use,
            embeddings=embeddings,
            run_config=run_config,
        )
    except Exception as e:
        print(f"[ОШИБКА] Ошибка при оценке: {e}")
        print("\nПодсказка:")
        print("  Попробуйте локальные embeddings:")
        print("  PowerShell: $env:RAGAS_EMBEDDINGS_PROVIDER='huggingface'")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 70)

    faithfulness_values = _get_metric_values(result, "faithfulness")
    context_precision_values = _get_metric_values(result, "context_precision")
    answer_relevancy_values = _get_metric_values(result, "answer_relevancy")
    if not answer_relevancy_values:
        answer_relevancy_values = _get_metric_values(result, "answer_relevance")

    avg_faithfulness = (
        sum(faithfulness_values) / len(faithfulness_values) if faithfulness_values else 0
    )
    avg_context_precision = (
        sum(context_precision_values) / len(context_precision_values) if context_precision_values else 0
    )
    avg_answer_relevancy = (
        sum(answer_relevancy_values) / len(answer_relevancy_values) if answer_relevancy_values else 0
    )

    print()
    print("[МЕТРИКИ] Средние значения:")
    print(f"   Faithfulness (точность ответа):          {avg_faithfulness:.4f}")
    print(f"   Context Precision (точность контекста):  {avg_context_precision:.4f}")
    print(f"   Answer Relevancy (релевантность ответа): {avg_answer_relevancy:.4f}")

    avg_score = (avg_faithfulness + avg_context_precision + avg_answer_relevancy) / 3
    print(f"\n{'-'*70}")
    print(f"[ИТОГО] Средний балл: {avg_score:.4f}")

    if avg_score >= 0.7:
        print("   Оценка: Отличное качество! [OK]")
    elif avg_score >= 0.5:
        print("   Оценка: Удовлетворительное качество [!]")
    else:
        print("   Оценка: Требует значительного улучшения [X]")

    print("\n" + "=" * 70)
    print("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ПО ВОПРОСАМ")
    print("=" * 70)

    ar_values = answer_relevancy_values
    if not ar_values:
        ar_values = _get_metric_values(result, "answer_relevance")

    for i, question in enumerate(EVALUATION_QUESTIONS):
        print(f"\n{i+1}. {question}")

        faith_val = faithfulness_values[i] if i < len(faithfulness_values) else float("nan")
        cp_val = context_precision_values[i] if i < len(context_precision_values) else float("nan")
        ar_val = ar_values[i] if i < len(ar_values) else float("nan")

        print(
            f"   Faithfulness:       {faith_val:.4f}"
            if not (isinstance(faith_val, float) and math.isnan(faith_val))
            else "   Faithfulness:       не удалось вычислить"
        )
        print(
            f"   Context Precision:  {cp_val:.4f}"
            if not (isinstance(cp_val, float) and math.isnan(cp_val))
            else "   Context Precision:  не удалось вычислить"
        )
        print(
            f"   Answer Relevancy:   {ar_val:.4f}"
            if not (isinstance(ar_val, float) and math.isnan(ar_val))
            else "   Answer Relevancy:   не удалось вычислить"
        )

    print("\n" + "=" * 70)
    print("[OK] Оценка завершена!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    evaluate_rag_system()
