"""
Переиндексация ChromaDB по ролям (текст подтягивается из Google Docs, см. config).

Примеры:
  python reindex.py --role hr
  python reindex.py --role post_sales
  python reindex.py --role sales
  python reindex.py --role all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent


def _load_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()


def _import_runner():
    """Импорт reindex_runner из каталога assistant_api."""
    assistant_dir = PROJECT_ROOT / "assistant_api"
    if str(assistant_dir) not in sys.path:
        sys.path.insert(0, str(assistant_dir))
    from reindex_runner import reindex_all_roles, reindex_role

    return reindex_role, reindex_all_roles


def main() -> None:
    parser = argparse.ArgumentParser(description="Переиндексация ChromaDB по роли ассистента.")
    parser.add_argument(
        "--role",
        choices=["hr", "post_sales", "sales", "all"],
        required=True,
        help="Какую базу знаний переиндексировать.",
    )
    args = parser.parse_args()
    _load_env()

    reindex_role, reindex_all_roles = _import_runner()
    try:
        if args.role == "all":
            reindex_all_roles()
        else:
            reindex_role(args.role)
    except Exception as exc:
        print(f"\nОшибка: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
