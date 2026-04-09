"""
Общие настройки корпоративных ассистентов (OpenAI + RAG).
"""

from __future__ import annotations

import os

# Роли ассистентов и подписи в интерфейсе
ASSISTANT_ROLES: tuple[str, ...] = ("hr", "post_sales", "sales")

ROLE_LABELS: dict[str, str] = {
    "hr": "HR — внутренние политики и вопросы сотрудников",
    "post_sales": "Постпродажа — гарантия и сервисное обслуживание",
    "sales": "Продажи — модельный ряд, цены и характеристики",
}

# LLM
DEFAULT_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip()
LLM_TEMPERATURE = 0.4
LLM_MAX_TOKENS = 500

# Embeddings (для RAG и семантического кеша)
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()

# openai — только OpenAI Embeddings API; local — sentence-transformers на ПК (если 403 region на Embeddings)
_eb = os.getenv("EMBEDDINGS_BACKEND", "local").strip().lower()
EMBEDDINGS_BACKEND = _eb if _eb in ("openai", "local") else "local"

LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
).strip()

# Семантический кеш: порог косинусного сходства (0..1)
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.88"))

# Путь к SQLite относительно каталога assistant_api
DEFAULT_CACHE_DB = "corporate_rag_cache.db"

# Корневая папка локальной базы знаний (подпапки: hr, post_sales, sales).
# В перспективе тот же контент можно синхронизировать из Google Docs — достаточно
# выгрузить текст в эти папки и снова запустить reindex.py.
KNOWLEDGE_ROOT = "knowledge"


def collection_name_for_role(role: str) -> str:
    """Имя коллекции ChromaDB для роли."""
    if role not in ASSISTANT_ROLES:
        raise ValueError(f"Неизвестная роль: {role}. Допустимо: {ASSISTANT_ROLES}")
    return f"corp_{role}"


def knowledge_dir_for_role(role: str) -> str:
    """Относительный путь к папке с .txt для роли."""
    return f"{KNOWLEDGE_ROOT}/{role}"
