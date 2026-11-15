"""HTTP minimale basato su urllib, con retry e gzip. Nessuna dipendenza esterna."""
from __future__ import annotations

import gzip
import json
import ssl
import time
import urllib.error
import urllib.request
import zlib

UA = "Mozilla/5.0 (compatible; ai-radar/1.0; personal news digest)"


def _ssl_context() -> ssl.SSLContext:
    try:  # alcune installazioni di Python su macOS non hanno i certificati di sistema
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


_CTX = _ssl_context()


class HttpError(Exception):
    def __init__(self, msg: str, status: int | None = None, body: str = ""):
        super().__init__(msg)
        self.status = status
        self.body = body


def _read_with_deadline(r, deadline: float | None) -> bytes:
    if deadline is None:
        return r.read()
    chunks = []
    while True:
        if time.time() > deadline:
            raise TimeoutError("tempo massimo superato")
        chunk = r.read1(16384) if hasattr(r, "read1") else r.read(16384)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def fetch(url: str, *, headers: dict | None = None, data=None, timeout: float = 30,
          retries: int = 2, method: str | None = None, max_seconds: float | None = None) -> bytes:
    """max_seconds: durata massima totale della richiesta (alcuni server tengono viva la connessione
    inviando spazi, quindi il timeout del socket da solo non basta)."""
    hdrs = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}
    hdrs.update(headers or {})
    body = None
    if data is not None:
        body = json.dumps(data).encode() if isinstance(data, (dict, list)) else data
        hdrs.setdefault("Content-Type", "application/json")
    last: HttpError | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=body, headers=hdrs, method=method)
        deadline = time.time() + max_seconds if max_seconds else None
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as r:
                raw = _read_with_deadline(r, deadline)
                enc = (r.headers.get("Content-Encoding") or "").lower()
                if enc == "gzip":
                    raw = gzip.decompress(raw)
                elif enc == "deflate":
                    raw = zlib.decompress(raw)
                return raw
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read()[:2000].decode("utf-8", "replace")
            except Exception:
                err_body = ""
            last = HttpError(f"HTTP {e.code} su {url}", e.code, err_body)
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                retry_after = e.headers.get("Retry-After") if e.headers else None
                wait = float(retry_after) if retry_after and retry_after.isdigit() else 4 * (attempt + 1)
                time.sleep(min(wait, 30))
                continue
            raise last
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = HttpError(f"{type(e).__name__} su {url}: {e}")
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
            raise last
    assert last is not None
    raise last


def get_text(url: str, **kw) -> str:
    return fetch(url, **kw).decode("utf-8", "replace")


def get_json(url: str, **kw):
    return json.loads(fetch(url, **kw))
