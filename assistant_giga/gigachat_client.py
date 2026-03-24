"""
Клиент для работы с GigaChat API от Сбера.
Управляет авторизацией и запросами к API.
"""

import requests
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import urllib3


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class GigaChatClient:
    """Клиент для работы с GigaChat API."""
    REQUEST_TIMEOUT_SECONDS = 30
    
    def __init__(self, auth_key: str = None, rq_uid: str = None):
        """
        Инициализация GigaChat клиента.
        
        Args:
            auth_key: ключ авторизации (Basic token)
            rq_uid: уникальный идентификатор запроса
        """
        self.auth_key = auth_key or os.getenv("GIGACHAT_AUTH_KEY")
        self.rq_uid = rq_uid or os.getenv("GIGACHAT_RQUID")
        
        if not self.auth_key:
            raise ValueError("GIGACHAT_AUTH_KEY не установлен")
        if not self.rq_uid:
            raise ValueError("GIGACHAT_RQUID не установлен")
        
        self.oauth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        self.api_url = "https://gigachat.devices.sberbank.ru/api/v1"
        
        self.access_token = None
        self.token_expires_at = None
        self._fallback_embedder = None
        self._embeddings_provider = os.getenv("GIGACHAT_EMBEDDINGS_PROVIDER", "auto").strip().lower()
        self._remote_embeddings_enabled = self._embeddings_provider != "local"
        self._remote_embeddings_warned = False
        
        # Получаем токен при инициализации
        self._refresh_token()
    
    def _refresh_token(self):
        """Получение нового access token."""
        payload = {'scope': 'GIGACHAT_API_PERS'}
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
            'RqUID': self.rq_uid,
            'Authorization': f'Basic {self.auth_key}'
        }
        
        try:
            response = requests.post(
                self.oauth_url,
                headers=headers,
                data=payload,
                timeout=self.REQUEST_TIMEOUT_SECONDS,
                verify=False  # Для работы с сертификатом Сбера
            )
            response.raise_for_status()
            
            data = response.json()
            self.access_token = data['access_token']
            # Токен действует 30 минут, обновляем за 1 минуту до истечения
            self.token_expires_at = datetime.now() + timedelta(minutes=29)
            
            print("[OK] GigaChat access token получен")
            
        except Exception as e:
            raise Exception(f"Ошибка получения access token: {e}")
    
    def _ensure_token_valid(self):
        """Проверка валидности токена и обновление при необходимости."""
        if not self.access_token or datetime.now() >= self.token_expires_at:
            print("[*] Обновление access token...")
            self._refresh_token()
    
    def _get_headers(self) -> Dict[str, str]:
        """Получение заголовков для запросов."""
        self._ensure_token_valid()
        return {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       model: str = "GigaChat",
                       temperature: float = 0.3,
                       max_tokens: int = 500) -> str:
        """
        Отправка запроса к чат-модели GigaChat.
        
        Args:
            messages: список сообщений в формате [{"role": "user", "content": "..."}]
            model: название модели
            temperature: температура генерации
            max_tokens: максимальное количество токенов в ответе
            
        Returns:
            текст ответа от модели
        """
        url = f"{self.api_url}/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(
                url,
                headers=self._get_headers(),
                json=payload,
                timeout=self.REQUEST_TIMEOUT_SECONDS,
                verify=False
            )
            response.raise_for_status()
            
            data = response.json()
            return data['choices'][0]['message']['content']
            
        except Exception as e:
            raise Exception(f"Ошибка запроса к GigaChat: {e}")
    
    def get_embeddings(self, texts: List[str], model: str = "Embeddings") -> List[List[float]]:
        """
        Получение векторных представлений текстов.
        
        Args:
            texts: список текстов для векторизации
            model: модель для embeddings
            
        Returns:
            список векторов
        """
        if not self._remote_embeddings_enabled:
            return self._get_fallback_embeddings(texts)

        url = f"{self.api_url}/embeddings"
        
        payload = {
            "model": model,
            "input": texts
        }
        
        try:
            response = requests.post(
                url,
                headers=self._get_headers(),
                json=payload,
                timeout=self.REQUEST_TIMEOUT_SECONDS,
                verify=False
            )
            response.raise_for_status()
            
            data = response.json()
            return [item['embedding'] for item in data['data']]
            
        except Exception as e:
            # Если embeddings API недоступен, используем локальный fallback.
            # При 402/403 отключаем удалённые embeddings до конца процесса.
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            if status_code in (402, 403):
                self._remote_embeddings_enabled = False
            if not self._remote_embeddings_warned:
                print(f"[WARN] Embeddings API недоступен, используется fallback: {e}")
                if status_code in (402, 403):
                    print("[WARN] Удалённые embeddings отключены до перезапуска приложения.")
                self._remote_embeddings_warned = True
            return self._get_fallback_embeddings(texts)

    def _get_fallback_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Локальный fallback embeddings через sentence-transformers.
        Если модель недоступна, используем простой hash-based fallback.
        """
        try:
            from sentence_transformers import SentenceTransformer

            if self._fallback_embedder is None:
                model_name = os.getenv(
                    "GIGACHAT_FALLBACK_EMBED_MODEL",
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
                print(f"[*] Инициализация fallback-эмбеддингов: {model_name}")
                self._fallback_embedder = SentenceTransformer(model_name)

            vectors = self._fallback_embedder.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return vectors.tolist()
        except Exception as fallback_error:
            print(f"[WARN] Локальный fallback embeddings недоступен: {fallback_error}")
            # Последний резервный fallback: token hash embeddings.
            # Работает без внешних зависимостей и лучше whole-text hash
            # для простого лексического поиска.
            import hashlib
            import math
            import re

            dim = 768
            embeddings = []
            for text in texts:
                vector = [0.0] * dim
                tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
                if not tokens:
                    tokens = [text.lower()]

                for token in tokens:
                    digest = hashlib.sha1(token.encode("utf-8")).digest()
                    index = int.from_bytes(digest[:4], "little") % dim
                    sign = 1.0 if (digest[4] % 2 == 0) else -1.0
                    vector[index] += sign

                norm = math.sqrt(sum(v * v for v in vector))
                if norm > 0:
                    vector = [v / norm for v in vector]
                embeddings.append(vector)
            return embeddings
    
    def get_models(self) -> List[Dict[str, Any]]:
        """
        Получение списка доступных моделей.
        
        Returns:
            список моделей
        """
        url = f"{self.api_url}/models"
        
        try:
            response = requests.get(
                url,
                headers=self._get_headers(),
                timeout=self.REQUEST_TIMEOUT_SECONDS,
                verify=False
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get('data', [])
            
        except Exception as e:
            print(f"Ошибка получения списка моделей: {e}")
            return []


if __name__ == "__main__":
    # Тестирование клиента
    import sys
    from pathlib import Path
    from dotenv import load_dotenv
    
    # Загружаем .env
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    
    try:
        client = GigaChatClient()
        
        # Тест получения моделей
        print("\n=== Доступные модели ===")
        models = client.get_models()
        for model in models:
            print(f"- {model.get('id', 'unknown')}")
        
        # Тест чата
        print("\n=== Тест чата ===")
        response = client.chat_completion(
            messages=[
                {"role": "user", "content": "Когда сотруднику могут отказать в использовании корпоративного такси? Ответь кратко."}
            ]
        )
        print(f"Ответ: {response}")
        
        # Тест embeddings
        print("\n=== Тест embeddings ===")
        embeddings = client.get_embeddings(["Тестовый текст"])
        print(f"Размерность вектора: {len(embeddings[0])}")
        
        print("\n[OK] Все тесты пройдены")
        
    except Exception as e:
        print(f"[ERROR] Ошибка: {e}")
        sys.exit(1)

