"""Utility per testo, date, URL e parsing RSS/Atom (solo stdlib)."""
from __future__ import annotations

import hashlib
import html
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def strip_html(s: str | None, limit: int | None = None) -> str:
    if not s:
        return ""
    s = re.sub(r"<(script|style|svg|nav|footer|header|noscript)\b[^>]*>.*?</\1>", " ", s, flags=re.S | re.I)
    s = re.sub(r"</t[dh]>", " | ", s, flags=re.I)
    s = re.sub(r"<br\s*/?>|</p>|</li>|</h\d>|</tr>|</div>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t\r\f\v ]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s).strip()
    if limit and len(s) > limit:
        s = s[:limit].rsplit(" ", 1)[0] + "…"
    return s


def one_line(s: str, limit: int | None = None) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    if limit and len(s) > limit:
        s = s[:limit].rsplit(" ", 1)[0] + "…"
    return s


def parse_date(s) -> datetime | None:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return datetime.fromtimestamp(s, tz=timezone.utc)
    s = str(s).strip()
    if not s:
        return None
    dt = None
    try:
        dt = parsedate_to_datetime(s)
    except Exception:
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            m = re.match(r"(\d{4}-\d{2}-\d{2})", s)
            if m:
                dt = datetime.fromisoformat(m.group(1))
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def iso(dt: datetime | None) -> str | None:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if dt else None


_TRACKING = re.compile(r"^(utm_|ref$|ref_src$|source$|fbclid$|gclid$|mc_)")


def normalize_url(url: str) -> str:
    try:
        p = urlsplit(url.strip())
    except Exception:
        return url.strip()
    q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if not _TRACKING.match(k)]
    scheme = "https" if p.scheme in ("http", "https") else p.scheme
    host = p.netloc.lower().removeprefix("www.").removeprefix("old.")
    path = p.path.rstrip("/") or "/"
    return urlunsplit((scheme, host, path, urlencode(q), ""))


def item_id(url: str, title: str = "") -> str:
    key = normalize_url(url) if url else one_line(title).lower()
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def norm_name(s: str) -> str:
    """Normalizza il nome di un modello per confronti fuzzy: 'Claude Fable 5.2' -> 'claudefable52'."""
    s = (s or "").lower()
    s = s.split(":", 1)[-1] if ":" in s and not s.startswith("http") else s  # "Anthropic: Claude X" -> "Claude X"
    s = s.split("/")[-1]
    return re.sub(r"[^a-z0-9]", "", s)


# ───────────────────────── RSS / Atom ─────────────────────────

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower() if isinstance(tag, str) else ""


def parse_feed(text: str) -> list[dict]:
    """Restituisce [{title, url, published(datetime|None), summary}] da RSS 2.0, Atom o RDF."""
    text = text.lstrip("﻿").strip()
    try:
        root = ET.fromstring(text.encode("utf-8"))
    except ET.ParseError:
        return _regex_feed(text)
    out = []
    for el in root.iter():
        if _local(el.tag) not in ("item", "entry"):
            continue
        d = {"title": "", "url": "", "published": None, "summary": ""}
        updated = None
        content = ""
        for ch in el:
            n = _local(ch.tag)
            txt = "".join(ch.itertext()).strip()
            if n == "title":
                d["title"] = one_line(strip_html(txt))
            elif n == "link":
                href = ch.get("href")
                if href:
                    if ch.get("rel", "alternate") == "alternate" or not d["url"]:
                        d["url"] = href
                elif txt and not d["url"]:
                    d["url"] = txt
            elif n in ("pubdate", "published", "date", "issued", "created") and not d["published"]:
                d["published"] = parse_date(txt)
            elif n in ("updated", "modified"):
                updated = parse_date(txt)
            elif n in ("description", "summary") and not d["summary"]:
                d["summary"] = txt
            elif n in ("encoded", "content") and not content:
                content = txt
        d["published"] = d["published"] or updated
        d["summary"] = strip_html(d["summary"] or content, 1500)
        if d["title"] or d["url"]:
            out.append(d)
    return out


def _regex_feed(text: str) -> list[dict]:
    out = []
    for block in re.findall(r"<(?:item|entry)\b.*?</(?:item|entry)>", text, flags=re.S | re.I):
        def grab(tag):
            m = re.search(rf"<{tag}\b[^>]*>(.*?)</{tag}>", block, flags=re.S | re.I)
            if not m:
                return ""
            v = m.group(1).strip()
            v = re.sub(r"^<!\[CDATA\[(.*)\]\]>$", r"\1", v, flags=re.S)
            return v
        url = grab("link")
        if not url:
            m = re.search(r"<link[^>]+href=\"([^\"]+)\"", block)
            url = m.group(1) if m else ""
        out.append({
            "title": one_line(strip_html(grab("title"))),
            "url": html.unescape(url),
            "published": parse_date(grab("pubDate") or grab("published") or grab("dc:date") or grab("updated")),
            "summary": strip_html(grab("description") or grab("summary") or grab("content"), 1500),
        })
    return out
