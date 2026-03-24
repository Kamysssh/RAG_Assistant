"""
Скрипт принудительной переиндексации векторной базы знаний.

Примеры запуска:
  python reindex.py --project api
  python reindex.py --project giga
  python reindex.py --project both
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent


def _load_env() -> None:
    """Загружает переменные окружения из .env, если файл существует."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()


def _require_env_vars(var_names: Iterable[str], project_label: str) -> None:
    """Проверяет обязательные переменные окружения."""
    missing = [name for name in var_names if not os.getenv(name)]
    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(
            f"Для проекта '{project_label}' не заданы переменные: {missing_list}. "
            "Проверьте файл .env."
        )


def _load_pipeline_class(project_dir: Path, module_tag: str):
    """Динамически загружает класс RAGPipeline из нужной папки проекта."""
    pipeline_path = project_dir / "rag_pipeline.py"
    spec = importlib.util.spec_from_file_location(module_tag, pipeline_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Не удалось загрузить модуль из {pipeline_path}")

    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(project_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)

    if not hasattr(module, "RAGPipeline"):
        raise RuntimeError(f"В модуле {pipeline_path} не найден класс RAGPipeline")
    return module.RAGPipeline


def reindex_api() -> None:
    """Переиндексация проекта assistant_api."""
    _require_env_vars(["OPENAI_API_KEY"], "assistant_api")
    project_dir = PROJECT_ROOT / "assistant_api"
    RAGPipeline = _load_pipeline_class(project_dir, "assistant_api_rag_pipeline")

    print("\n=== Переиндексация assistant_api ===")
    os.environ["RAG_FORCE_REINDEX"] = "1"
    RAGPipeline(
        collection_name="api_rag_collection",
        cache_db_path="api_rag_cache.db",
        data_file="data",
        model="gpt-4o-mini",
    )
    print("assistant_api: готово")


def reindex_giga() -> None:
    """Переиндексация проекта assistant_giga."""
    _require_env_vars(["GIGACHAT_AUTH_KEY", "GIGACHAT_RQUID"], "assistant_giga")
    project_dir = PROJECT_ROOT / "assistant_giga"
    RAGPipeline = _load_pipeline_class(project_dir, "assistant_giga_rag_pipeline")

    print("\n=== Переиндексация assistant_giga ===")
    os.environ["RAG_FORCE_REINDEX"] = "1"
    RAGPipeline(
        collection_name="gigachat_rag_collection",
        cache_db_path="gigachat_rag_cache.db",
        data_file="data",
        model="GigaChat",
    )
    print("assistant_giga: готово")


def main() -> None:
    """Точка входа CLI."""
    parser = argparse.ArgumentParser(
        description="Принудительная переиндексация RAG-коллекций."
    )
    parser.add_argument(
        "--project",
        choices=["api", "giga", "both"],
        default="both",
        help="Какой проект переиндексировать (по умолчанию: both).",
    )
    args = parser.parse_args()

    _load_env()
    try:
        if args.project in ("api", "both"):
            reindex_api()
        if args.project in ("giga", "both"):
            reindex_giga()
    except Exception as exc:
        print(f"\nОшибка: {exc}")
        raise SystemExit(1) from exc

    print("\nПереиндексация завершена.")


if __name__ == "__main__":
    main()
