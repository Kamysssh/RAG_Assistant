# ASCII-only source file (avoids cp1251 vs utf-8 issues on Windows).
"""Fetch Google Docs as plain text via public export (?format=txt).

Requires link sharing with viewer access. Default URLs: config.DEFAULT_KNOWLEDGE_GOOGLE_DOCS.
Override: KNOWLEDGE_<ROLE>_GOOGLE_DOCS in .env.

SSL errors (e.g. DECRYPTION_FAILED_OR_BAD_RECORD_MAC): often antivirus HTTPS scanning,
VPN, or bad network path. Try: pause VPN, disable HTTPS inspection in AV, or set
GOOGLE_DOCS_SSL_VERIFY=0 in .env (insecure; debugging only).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request

import requests

from config import DEFAULT_KNOWLEDGE_GOOGLE_DOCS

logger = logging.getLogger(__name__)

_EXPORT_TIMEOUT_SEC = 120
_SSL_RETRIES = 3


def _ssl_verify_enabled() -> bool:
    return os.getenv("GOOGLE_DOCS_SSL_VERIFY", "true").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _export_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/plain,*/*;q=0.9",
    }


def _is_html_login_wall(text: str) -> bool:
    head = (text or "")[:800].lower()
    return "<!doctype html" in head or "<html" in head


def _normalize_export_text(text: str) -> str | None:
    text = (text or "").strip()
    if not text:
        return None
    if _is_html_login_wall(text):
        logger.warning("Google Docs export: got HTML (login wall?), not plain text")
        return None
    return text


def _requests_verify_arg(verify_ssl: bool) -> bool | str:
    if not verify_ssl:
        return False
    try:
        import certifi

        return certifi.where()
    except ImportError:
        return True


def _fetch_via_requests(doc_id: str, verify_ssl: bool) -> str | None:
    if not verify_ssl:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    url = f"https://docs.google.com/document/d/{doc_id}/export"
    verify = _requests_verify_arg(verify_ssl)
    last_err: Exception | None = None
    for attempt in range(_SSL_RETRIES):
        try:
            r = requests.get(
                url,
                params={"format": "txt"},
                timeout=_EXPORT_TIMEOUT_SEC,
                headers=_export_headers(),
                verify=verify,
            )
        except requests.exceptions.SSLError as exc:
            last_err = exc
            logger.warning(
                "Google Docs export: SSL error (attempt %s/%s): %s",
                attempt + 1,
                _SSL_RETRIES,
                exc,
            )
            time.sleep(1.0 * (attempt + 1))
            continue
        except requests.RequestException as exc:
            logger.warning("Google Docs export: network error: %s", exc)
            return None

        if r.status_code != 200:
            logger.warning(
                "Google Docs export: HTTP %s for %s", r.status_code, doc_id
            )
            return None
        return _normalize_export_text(r.text or "")

    if last_err is not None:
        logger.warning(
            "Google Docs export: giving up after SSL errors: %s", last_err
        )
    return None


def _ssl_context_for_urllib(verify_ssl: bool):
    import ssl

    if verify_ssl:
        return ssl.create_default_context()
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _fetch_via_curl(doc_id: str, verify_ssl: bool) -> str | None:
    """Use OS curl (on Windows often Schannel) when Python OpenSSL fails."""
    curl_exe = shutil.which("curl")
    if not curl_exe:
        return None
    url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    ua = _export_headers()["User-Agent"]
    cmd: list[str] = [curl_exe]
    if os.name == "nt":
        cmd.append("--ssl-no-revoke")
    if not verify_ssl:
        cmd.append("-k")
    cmd.extend(
        [
            "-L",
            "-sS",
            "--max-time",
            str(_EXPORT_TIMEOUT_SEC),
            "-A",
            ua,
            url,
        ]
    )
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_EXPORT_TIMEOUT_SEC + 15,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("Google Docs export (curl): %s", exc)
        return None
    if result.returncode != 0:
        err = (result.stderr or "").strip()[:300]
        logger.warning(
            "Google Docs export (curl): exit %s %s",
            result.returncode,
            err or "",
        )
        return None
    return _normalize_export_text(result.stdout or "")


def _fetch_via_urllib(doc_id: str, verify_ssl: bool) -> str | None:
    url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    req = urllib.request.Request(url, headers=_export_headers())
    ctx = _ssl_context_for_urllib(verify_ssl)
    try:
        with urllib.request.urlopen(
            req, timeout=_EXPORT_TIMEOUT_SEC, context=ctx
        ) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        logger.warning("Google Docs export (urllib): HTTP %s", exc.code)
        return None
    except OSError as exc:
        logger.warning("Google Docs export (urllib): %s", exc)
        return None

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
    return _normalize_export_text(text)


def fetch_google_doc_public_export(doc_id: str) -> str | None:
    verify_ssl = _ssl_verify_enabled()

    text = _fetch_via_requests(doc_id, verify_ssl)
    if text:
        return text

    if os.name == "nt":
        logger.info("Google Docs export: trying curl (Windows / Schannel) for %s", doc_id)
        text = _fetch_via_curl(doc_id, verify_ssl)
        if text:
            return text

    logger.info("Google Docs export: trying urllib fallback for %s", doc_id)
    text = _fetch_via_urllib(doc_id, verify_ssl)
    if text:
        return text

    if os.name != "nt":
        logger.info("Google Docs export: trying curl fallback for %s", doc_id)
        return _fetch_via_curl(doc_id, verify_ssl)
    return None


def fetch_google_doc_text(doc_id: str) -> str:
    logger.info("Google Docs: export txt, id=%s", doc_id)
    public = fetch_google_doc_public_export(doc_id)
    if public:
        return public
    verify_on = _ssl_verify_enabled()
    ssl_hint = (
        " This is usually a TLS/network issue (not Google sharing), if the doc is "
        '"Anyone with the link". Try: VPN off, antivirus HTTPS scan off, '
        "ensure `curl` works in terminal, pip install -U certifi requests urllib3, "
        "or GOOGLE_DOCS_SSL_VERIFY=0 in .env (insecure, debug only)."
    )
    access_hint = "" if not verify_on else (
        ' If the doc is not public: set "Anyone with the link" as viewer.'
    )
    raise RuntimeError(
        f"Could not download document {doc_id} as text.{access_hint}{ssl_hint}"
    )


def google_doc_ids_for_role(role: str) -> list[str]:
    key = f"KNOWLEDGE_{role.upper()}_GOOGLE_DOCS"
    raw = os.getenv(key, "").strip()
    if raw:
        return [parse_doc_id(part) for part in raw.split(",") if part.strip()]
    default = (DEFAULT_KNOWLEDGE_GOOGLE_DOCS.get(role) or "").strip()
    if not default:
        return []
    return [parse_doc_id(part) for part in default.split(",") if part.strip()]


def parse_doc_id(raw: str) -> str:
    """Document ID or full Google Docs URL."""
    raw = raw.strip()
    if "/document/d/" in raw:
        m = re.search(r"/document/d/([a-zA-Z0-9_-]+)", raw)
        if m:
            return m.group(1)
    return raw


def get_extra_sources_for_role(role: str) -> list[tuple[str, str]]:
    """Pairs (source_label, text) for VectorStore.load_documents."""
    out: list[tuple[str, str]] = []
    for doc_id in google_doc_ids_for_role(role):
        text = fetch_google_doc_text(doc_id).strip()
        if not text:
            logger.warning("Google Docs %s: empty text, skip", doc_id)
            continue
        label = f"google_doc_{doc_id}.txt"
        out.append((label, text))
    return out
