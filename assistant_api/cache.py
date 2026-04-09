"""
Кеш ответов RAG: точное совпадение вопроса + семантический поиск по embedding (cosine similarity).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Any, Callable, Optional, cast

import numpy as np

from config import SEMANTIC_CACHE_THRESHOLD

logger = logging.getLogger(__name__)

NormalizeEmbedFn = Callable[[str], list[float]]


def _normalize_query(query: str) -> str:
    return " ".join(query.lower().strip().split())


def _query_hash(normalized: str) -> str:
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


class SemanticRAGCache:
    """
    SQLite-кеш: для каждой роли хранятся вопрос, ответ, embedding вопроса и контекст.
    Сначала ищется точное совпадение нормализованного текста, затем семантическое.
    """

    def __init__(
        self,
        db_path: str,
        embed_fn: NormalizeEmbedFn,
        semantic_threshold: float | None = None,
    ):
        self.db_path = db_path
        self.embed_fn = embed_fn
        self.semantic_threshold = (
            semantic_threshold if semantic_threshold is not None else SEMANTIC_CACHE_THRESHOLD
        )
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS response_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    query_hash TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    context_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_cache_role_hash
                ON response_cache(role, query_hash)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_cache_role
                ON response_cache(role)
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uq_cache_role_query_hash
                ON response_cache(role, query_hash)
                """
            )
            conn.commit()

    def get(self, query: str, role: str) -> Optional[dict[str, Any]]:
        """Возвращает запись из кеша или None."""
        normalized = _normalize_query(query)
        qh = _query_hash(normalized)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                """
                SELECT query_text, answer, context_json, created_at
                FROM response_cache
                WHERE role = ? AND query_hash = ?
                """,
                (role, qh),
            )
            row = cur.fetchone()

        if row:
            logger.info("Кеш: точное совпадение (role=%s)", role)
            return {
                "query": row["query_text"],
                "answer": row["answer"],
                "context": json.loads(row["context_json"]) if row["context_json"] else None,
                "created_at": row["created_at"],
                "cache_hit": "exact",
            }

        # Семантический поиск
        try:
            query_embedding = self.embed_fn(query)
        except Exception as exc:
            logger.exception("Кеш: не удалось вычислить embedding: %s", exc)
            return None

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                """
                SELECT query_text, answer, context_json, created_at, embedding_json
                FROM response_cache
                WHERE role = ?
                """,
                (role,),
            )
            rows = cur.fetchall()

        best_sim = -1.0
        best: Optional[Any] = None
        for r in rows:
            try:
                stored = json.loads(r["embedding_json"])
                sim = _cosine_similarity(query_embedding, stored)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if sim > best_sim:
                best_sim = sim
                best = r

        if best is not None and best_sim >= self.semantic_threshold:
            row = cast(sqlite3.Row, best)
            logger.info(
                "Кеш: семантическое совпадение (role=%s, similarity=%.4f, threshold=%.4f)",
                role,
                best_sim,
                self.semantic_threshold,
            )
            return {
                "query": row["query_text"],
                "answer": row["answer"],
                "context": json.loads(row["context_json"]) if row["context_json"] else None,
                "created_at": row["created_at"],
                "cache_hit": "semantic",
                "similarity": best_sim,
            }

        logger.debug(
            "Кеш: промах (role=%s, best_similarity=%.4f)",
            role,
            best_sim if rows else 0.0,
        )
        return None

    def set(
        self,
        query: str,
        role: str,
        answer: str,
        context: Optional[list[str]] = None,
    ) -> None:
        """Сохраняет ответ и embedding исходного вопроса."""
        normalized = _normalize_query(query)
        qh = _query_hash(normalized)
        emb = self.embed_fn(query)
        context_json = json.dumps(context) if context else None
        embedding_json = json.dumps(emb)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO response_cache
                (role, query_text, query_hash, embedding_json, answer, context_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(role, query_hash) DO UPDATE SET
                    query_text = excluded.query_text,
                    embedding_json = excluded.embedding_json,
                    answer = excluded.answer,
                    context_json = excluded.context_json,
                    created_at = excluded.created_at
                """,
                (
                    role,
                    query,
                    qh,
                    embedding_json,
                    answer,
                    context_json,
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
        logger.info("Кеш: сохранена запись (role=%s)", role)

    def clear(self, role: str | None = None) -> None:
        """Очистка всего кеша или только выбранной роли."""
        with sqlite3.connect(self.db_path) as conn:
            if role:
                conn.execute("DELETE FROM response_cache WHERE role = ?", (role,))
            else:
                conn.execute("DELETE FROM response_cache")
            conn.commit()
        logger.info("Кеш очищен (role=%s)", role or "все")

    def get_stats(self) -> dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT COUNT(*) FROM response_cache")
            count = cur.fetchone()[0]
            cur = conn.execute("SELECT MIN(created_at), MAX(created_at) FROM response_cache")
            dates = cur.fetchone()

        size_mb = os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0.0
        return {
            "total_entries": count,
            "oldest_entry": dates[0] if dates and dates[0] else None,
            "newest_entry": dates[1] if dates and dates[1] else None,
            "db_size_mb": size_mb,
            "semantic_threshold": self.semantic_threshold,
        }
