# RAG Assistant Final Project

Учебный проект с двумя отдельными RAG-приложениями:

- `assistant_api` — RAG на OpenAI API.
- `assistant_giga` — RAG на GigaChat API.

Оба проекта используют:

- локальное векторное хранилище ChromaDB;
- локальный кеш ответов в SQLite;
- общую базу зависимостей из `requirements.txt`.

---

## 1) Требования

- Python `3.12`
- Windows PowerShell (примеры ниже для PowerShell)

Проверка версии:

```powershell
python --version
```

---

## 2) Установка зависимостей

Из корня проекта:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Если обычная установка медленная/падает по сети, можно использовать mirror:

```powershell
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --timeout 120 --retries 10
```

---

## 3) Настройка `.env`

1. Скопируйте шаблон:

```powershell
Copy-Item .env.example .env
```

2. Откройте `.env` и заполните ключи:

- `OPENAI_API_KEY` — для `assistant_api` и (опционально) RAGAS;
- `GIGACHAT_AUTH_KEY`, `GIGACHAT_RQUID` — для `assistant_giga`.

Важно:

- `.env` не должен попадать в Git (уже добавлено в `.gitignore`).
- В репозиторий добавляйте только `.env.example`.

---

## 4) Запуск приложений

### OpenAI-проект

```powershell
.\venv\Scripts\python.exe assistant_api\app.py
```

### GigaChat-проект

```powershell
.\venv\Scripts\python.exe assistant_giga\app.py
```

Команды внутри CLI:

- `stats` — статистика кеша;
- `clear` — очистка кеша;
- `exit` или `quit` — выход.

---

## 5) Переиндексация базы знаний

Если вы обновили файлы в `assistant_api/data` или `assistant_giga/data`, выполните переиндексацию:

```powershell
.\venv\Scripts\python.exe reindex.py --project api
.\venv\Scripts\python.exe reindex.py --project giga
```

Или сразу оба:

```powershell
.\venv\Scripts\python.exe reindex.py --project both
```

---

## 6) Оценка качества через RAGAS

### OpenAI-проект

```powershell
.\venv\Scripts\python.exe assistant_api\evaluate_ragas.py
```

### GigaChat-проект

```powershell
.\venv\Scripts\python.exe assistant_giga\evaluate_ragas.py
```

Примечание по `assistant_giga/evaluate_ragas.py`:

- для embeddings по умолчанию используется `huggingface`;
- при недоступности OpenAI (например, `403 unsupported_country_region_territory`) скрипт завершится быстро с понятным сообщением, без долгого "зависания".

Опциональные переменные:

- `RAGAS_EMBEDDINGS_PROVIDER` = `huggingface` или `openai`
- `RAGAS_HF_MODEL` (модель локальных embeddings)
- `RAGAS_OPENAI_EMBEDDINGS_MODEL`

---

## 7) Где хранятся данные

- Кеш ответов:
  - `assistant_api/api_rag_cache.db`
  - `assistant_giga/gigachat_rag_cache.db`
- Индексы Chroma:
  - `assistant_api/chroma_db`
  - `assistant_giga/chroma_db`
- Документы базы знаний:
  - `assistant_api/data`
  - `assistant_giga/data`

Эти файлы/папки игнорируются Git.

---

## 8) Подготовка к GitHub

Перед коммитом проверьте:

```powershell
git status
```

В `git status` не должно быть:

- `.env`
- `*.db`
- `chroma_db/*`
- содержимого `assistant_api/data/*`, `assistant_giga/data/*`
- `venv/*`

---

## 9) Быстрый старт (кратко)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# заполнить ключи в .env
.\venv\Scripts\python.exe reindex.py --project both
.\venv\Scripts\python.exe assistant_giga\app.py
```

