# Корпоративные нейроассистенты (OpenAI + RAG)

Учебный CLI-проект: три роли ассистента для автодилера — **HR**, **постпродажа**, **продажи**. База знаний подтягивается из **Google Docs** (просмотр по ссылке; URL по умолчанию в `assistant_api/config.py`, переопределение в `.env`), с **семантическим кешированием** повторяющихся вопросов. Интеграция только с **OpenAI**.

## Возможности

- Три изолированные коллекции ChromaDB (`corp_hr`, `corp_post_sales`, `corp_sales`); источник текста при индексации — Google Docs по роли.
- Промпты вынесены в `assistant_api/prompts.py` (цели, сценарии, ограничения, примеры FAQ).
- **Семантический кеш** (SQLite): сначала точное совпадение вопроса, затем сравнение embedding вопроса с сохранёнными (cosine similarity), порог по умолчанию `0.88` (`SEMANTIC_CACHE_THRESHOLD` в `.env`).
- **Логи**: консоль + файл `assistant_api/logs/app.log`.
- Параметры LLM: температура **0.4**, **max_tokens 500** (`assistant_api/config.py`).

## Стек

| Компонент | Технологии |
|-----------|------------|
| Язык | Python 3.12 |
| Чат (LLM) | OpenAI API (при необходимости — `OPENAI_BASE_URL` в `.env`) |
| Эмбеддинги RAG | OpenAI Embeddings **или** локально `sentence-transformers` (`EMBEDDINGS_BACKEND=local`, по умолчанию — удобно при 403 region на Embeddings) |
| Векторный поиск | ChromaDB (локально) |
| Кеш | SQLite |

## Установка

```powershell
cd "путь\к\проекту"
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
# Укажите OPENAI_API_KEY в .env
```

Опционально: установить Python 3.12 через Windows — `winget install Python.Python.3.12`. Если `pypi.org` недоступен, можно указать зеркало, например:

`python -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com`

## Переиндексация базы знаний

После изменения текста в Google Docs (или ссылок в `config` / `.env`) выполните (из корня репозитория):

```powershell
.\venv\Scripts\python.exe reindex.py --role hr
.\venv\Scripts\python.exe reindex.py --role post_sales
.\venv\Scripts\python.exe reindex.py --role sales
```

Все роли сразу:

```powershell
.\venv\Scripts\python.exe reindex.py --role all
```

## Запуск ассистента в терминале

```powershell
.\venv\Scripts\python.exe assistant_api\app.py
```

Дальше выберите роль (1 — HR, 2 — постпродажа, 3 — продажи). Команды: `stats`, `clear`, `role`, `help`, `exit`.

## Оценка качества (RAGAS, опционально)

Для роли HR (тот же источник, что в `DEFAULT_KNOWLEDGE_GOOGLE_DOCS`):

```powershell
.\venv\Scripts\python.exe assistant_api\evaluate_ragas.py
```

## Локальные данные (не в Git)

- `assistant_api/chroma_db/`, `assistant_api/*.db`, `assistant_api/logs/`, `.env`, `venv/`

Папки `assistant_api/knowledge/<роль>/` в репозитории пустые (заглушки `.gitkeep`). Папка `assistant_api/data/` (если создадите локально) в Git не отслеживается.

## Лицензия

Укажите лицензию при публикации репозитория, если планируете открытый код.
