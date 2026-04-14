"""
Единая фабрика клиента OpenAI: ключ, опционально OPENAI_BASE_URL (прокси / совместимый API).
Таймауты HTTP: OPENAI_TIMEOUT, OPENAI_CONNECT_TIMEOUT (.env) — при медленном TLS/сети.
"""

from __future__ import annotations

import os

import httpx
from openai import OpenAI


def _openai_http_timeout() -> httpx.Timeout:
    # connect: TLS handshake to api.openai.com; read/write: ответ модели
    connect = float(os.getenv("OPENAI_CONNECT_TIMEOUT", "120"))
    rw = float(os.getenv("OPENAI_READ_TIMEOUT", "180"))
    return httpx.Timeout(connect=connect, read=rw, write=rw, pool=rw)


def create_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY не установлен")

    timeout = _openai_http_timeout()
    base = os.getenv("OPENAI_BASE_URL", "").strip()
    if base:
        return OpenAI(api_key=api_key, base_url=base, timeout=timeout)
    return OpenAI(api_key=api_key, timeout=timeout)
