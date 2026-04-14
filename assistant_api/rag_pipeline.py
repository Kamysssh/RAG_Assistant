"""
RAG pipeline: семантический кеш → поиск в Chroma → OpenAI Chat (только одна роль за раз).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from cache import SemanticRAGCache
from config import (
    ASSISTANT_ROLES,
    DEFAULT_CACHE_DB,
    DEFAULT_CHAT_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    collection_name_for_role,
    knowledge_dir_for_role,
)
from embeddings import embed_text
from openai_client import create_openai_client
from google_docs_knowledge import get_extra_sources_for_role
from prompts import build_system_prompt
from vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Один экземпляр на выбранную роль (hr / post_sales / sales)."""

    def __init__(
        self,
        role: str,
        collection_name: str | None = None,
        cache_db_path: str = DEFAULT_CACHE_DB,
        data_dir: str | None = None,
        model: str | None = None,
    ):
        if role not in ASSISTANT_ROLES:
            raise ValueError(f"Роль должна быть одной из {ASSISTANT_ROLES}, получено: {role}")

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY не установлен")

        self.role = role
        self.model = model or DEFAULT_CHAT_MODEL
        self.openai_client = create_openai_client()

        project_dir = Path(__file__).resolve().parent
        self._collection_name = collection_name or collection_name_for_role(role)
        data_rel = data_dir or knowledge_dir_for_role(role)
        self._data_path = str((project_dir / data_rel).resolve())
        resolved_cache = str((project_dir / cache_db_path).resolve()) if not Path(cache_db_path).is_absolute() else cache_db_path

        force_reindex = os.getenv("RAG_FORCE_REINDEX", "0").strip().lower() in {"1", "true", "yes", "y"}

        logger.info("Инициализация векторного хранилища: коллекция %s", self._collection_name)
        self.vector_store = VectorStore(collection_name=self._collection_name)

        chunk_count = self.vector_store.collection.count()
        # Индексация (эмбеддинги + Chroma) только при явном запросе: ответ «y» при старте
        # (reindex_all_roles) или команда `python reindex.py`. Без этого пустая коллекция — ошибка.
        if chunk_count == 0 and not force_reindex:
            raise RuntimeError(
                "Векторная база для этой роли пуста. Перезапустите приложение и на вопрос о "
                "переиндексации ответьте «y», либо выполните из корня проекта:\n"
                "  python reindex.py --role all\n"
                "или для одной роли: python reindex.py --role hr"
            )

        if chunk_count == 0 or force_reindex:
            extras = get_extra_sources_for_role(role)
            if not extras:
                raise RuntimeError(
                    f"Нет ссылок на Google Docs для роли «{role}». Задайте в .env "
                    f"KNOWLEDGE_{role.upper()}_GOOGLE_DOCS или проверьте config.DEFAULT_KNOWLEDGE_GOOGLE_DOCS."
                )
            logger.info(
                "Индексация из Google Docs (%s док.); локальный каталог %s не обязателен",
                len(extras),
                self._data_path,
            )
            self.vector_store.load_documents(
                self._data_path,
                force_reload=force_reindex,
                extra_sources=extras,
            )

        def _embed_for_cache(q: str) -> list[float]:
            return embed_text(q, self.openai_client)

        self.cache = SemanticRAGCache(db_path=resolved_cache, embed_fn=_embed_for_cache)
        self._system_prompt = build_system_prompt(role)

        logger.info("RAG pipeline готов: роль=%s, модель=%s", role, self.model)

    def _user_message(self, query: str, context_docs: list[dict[str, Any]]) -> str:
        parts = []
        for i, doc in enumerate(context_docs, 1):
            parts.append(f"Фрагмент {i}:\n{doc['text']}\n")
        context_block = "\n".join(parts) if parts else "(контекст пуст — в базе ничего не найдено)"
        return f"""Контекст из базы знаний:
{context_block}

Вопрос пользователя: {query}

Ответь по правилам из системной инструкции, опираясь на контекст."""

    def _generate_answer(self, user_message: str) -> str:
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        content = response.choices[0].message.content
        return (content or "").strip()

    def query(self, user_query: str, use_cache: bool = True) -> dict[str, Any]:
        logger.info("Запрос (role=%s): %s", self.role, user_query[:200])

        if use_cache:
            cached = self.cache.get(user_query, self.role)
            if cached:
                raw_ctx = cached.get("context")
                if raw_ctx and isinstance(raw_ctx, list) and raw_ctx and isinstance(raw_ctx[0], str):
                    context_docs = [{"text": t} for t in raw_ctx]
                else:
                    context_docs = raw_ctx or []
                return {
                    "query": user_query,
                    "answer": cached["answer"],
                    "from_cache": True,
                    "cache_hit": cached.get("cache_hit"),
                    "similarity": cached.get("similarity"),
                    "context_docs": context_docs,
                    "cached_at": cached.get("created_at"),
                    "role": self.role,
                    "model": self.model,
                }

        context_docs = self.vector_store.search(user_query, top_k=3)
        logger.info("Найдено фрагментов контекста: %s", len(context_docs))

        user_message = self._user_message(user_query, context_docs)
        answer = self._generate_answer(user_message)
        logger.info("Ответ сгенерирован моделью")

        if use_cache:
            ctx_list = [d["text"] for d in context_docs]
            self.cache.set(user_query, self.role, answer, ctx_list)

        return {
            "query": user_query,
            "answer": answer,
            "from_cache": False,
            "context_docs": context_docs,
            "model": self.model,
            "role": self.role,
            "mode": "openai",
        }

    def get_stats(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "vector_store": self.vector_store.get_collection_stats(),
            "cache": self.cache.get_stats(),
            "model": self.model,
            "mode": "openai",
        }
