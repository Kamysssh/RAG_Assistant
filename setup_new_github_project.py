# All ASCII: avoids Windows cp1251 vs utf-8 issues when running the script.
"""Copy tracked project files into a new folder and run git init (fresh repo, no old remote).

Run from project root:
    python setup_new_github_project.py

Default destination: ../Corporate_RAG_Assistant
Custom: python setup_new_github_project.py "D:\\path\\to\\NewFolder"
"""
from __future__ import annotations

import argparse
import io
import subprocess
import sys
import tarfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export git snapshot (git archive HEAD) and git init in a new folder",
    )
    parser.add_argument(
        "dest",
        nargs="?",
        default=None,
        help="Destination folder (default: sibling Corporate_RAG_Assistant)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    if args.dest:
        dest = Path(args.dest).expanduser().resolve()
    else:
        dest = root.parent / "Corporate_RAG_Assistant"

    if dest.exists() and any(dest.iterdir()):
        print(f"Folder already exists and is not empty: {dest}")
        print("Delete it or pass a different path.")
        return 1

    dest.mkdir(parents=True, exist_ok=True)

    # Only tracked files from last commit (no venv, .env, DBs, etc.)
    archive = subprocess.check_output(["git", "archive", "HEAD"], cwd=root)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tf:
        try:
            tf.extractall(dest, filter="data")
        except TypeError:
            tf.extractall(dest)

    subprocess.run(["git", "init", "-b", "main"], cwd=dest, check=True)
    subprocess.run(["git", "add", "-A"], cwd=dest, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit: corporate RAG assistant"],
        cwd=dest,
        check=True,
    )

    print("Done.")
    print(f" Folder: {dest}")
    print(" Next: on GitHub create an empty repo (no README). Copy its HTTPS URL.")
    print(" Then in that folder run:")
    print(f'    cd "{dest}"')
    print("    git remote add origin <YOUR_REPO_URL>")
    print("    git push -u origin main")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}", file=sys.stderr)
        raise SystemExit(e.returncode or 1)
