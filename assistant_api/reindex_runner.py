"""
Переиндексация ChromaDB по ролям. Используется из корневого reindex.py и из app.py.
"""

from __future__ import annotations

import logging
import os

from config import ASSISTANT_ROLES
from rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


def reindex_role(role: str) -> None:
    """Принудительная переиндексация одной роли."""
    if role not in ASSISTANT_ROLES:
        raise ValueError(f"Неизвестная роль: {role}. Ожидается одна из {ASSISTANT_ROLES}")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Задайте OPENAI_API_KEY в .env")

    os.environ["RAG_FORCE_REINDEX"] = "1"
    try:
        print(f"\n=== Переиндексация роли: {role} ===")
        logger.info("Переиндексация: %s", role)
        RAGPipeline(role=role)
        print(f"=== Готово: {role} ===\n")
    finally:
        os.environ["RAG_FORCE_REINDEX"] = "0"


def reindex_all_roles() -> None:
    """Переиндексация всех коллекций (hr, post_sales, sales)."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Задайте OPENAI_API_KEY в .env")

    os.environ["RAG_FORCE_REINDEX"] = "1"
    try:
        for role in ASSISTANT_ROLES:
            print(f"\n=== Переиндексация роли: {role} ===")
            logger.info("Переиндексация: %s", role)
            RAGPipeline(role=role)
            print(f"=== Готово: {role} ===\n")
        print("Переиндексация завершена.")
    finally:
        os.environ["RAG_FORCE_REINDEX"] = "0"
