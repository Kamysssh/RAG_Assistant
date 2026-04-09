"""
Единая фабрика клиента OpenAI: ключ, опционально OPENAI_BASE_URL (прокси / совместимый API).
"""

from __future__ import annotations

import os

from openai import OpenAI


def create_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY не установлен")

    base = os.getenv("OPENAI_BASE_URL", "").strip()
    if base:
        return OpenAI(api_key=api_key, base_url=base)
    return OpenAI(api_key=api_key)
