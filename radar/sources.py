"""Raccolta notizie da fonti eterogenee. Ogni fonte restituisce una lista di item (dict)."""
from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

from . import net
from .textutil import item_id, iso, one_line, parse_date, parse_feed, strip_html

AI_TITLE_RE = re.compile(
    # nomi di modelli/aziende: niente \b finale, così matchano anche "Qwen4", "GPT-6", "Gemma4"
    r"\b(gpt|claude|fable|mythos|opus|sonnet|haiku|gemini|gemma|llama|qwen|deepseek|mistral|kimi|glm|minimax|grok|"
    r"nemotron|phi-?\d|openai|anthropic|deepmind|hugging ?face|ollama|vllm|gguf|quantiz|fine-?tun|embedding)"
    r"|\b(ai|a\.i\.|llms?|rag|agents?|agentic|inference|lora|diffusion|neural|transformers?|model weights|"
    r"open[- ]weights?|reasoning models?|language models?|tokens?|cuda|gpu|rtx|copilot|cursor|codex|chatbot|"
    r"multimodal|vision models?|tts|speech models?|benchmarks?)\b",
    re.I,
)


def make_item(*, source: str, source_name: str, kind: str, title: str, url: str,
              published: datetime | None, summary: str = "", engagement: float = 0,
              weight: float = 1.0, extra: dict | None = None) -> dict:
    return {
        "id": item_id(url, title),
        "source": source,
        "source_name": source_name,
        "kind": kind,
        "title": one_line(title, 300),
        "url": url,
        "published": iso(published),
        "summary": strip_html(summary, 900),
        "engagement": engagement,
        "weight": weight,
        "extra": extra or {},
    }


# ───────────────────────── RSS ─────────────────────────

def _one_feed(feed: dict, since: datetime) -> list[dict]:
    text = net.get_text(feed["url"], timeout=30, retries=1)
    entries = parse_feed(text)
    items = []
    undated = 0
    for e in entries:
        pub = e["published"]
        if pub is None:
            undated += 1
            if undated > 5:
                continue
        elif pub < since:
            continue
        if not e["url"]:
            continue
        items.append(make_item(
            source="rss", source_name=feed["name"], kind=feed.get("kind", "blog"),
            title=e["title"], url=e["url"], published=pub, summary=e["summary"],
            weight=float(feed.get("weight", 1.0)),
        ))
    return items


def rss_feeds(cfg: dict, since: datetime, status: dict) -> list[dict]:
    feeds = cfg["sources"].get("feeds", [])
    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_one_feed, f, since): f for f in feeds}
        for fut in as_completed(futs):
            f = futs[fut]
            try:
                items = fut.result()
                out.extend(items)
                status[f"rss:{f['name']}"] = {"ok": True, "n": len(items)}
            except Exception as e:  # una fonte rotta non deve fermare la rassegna
                status[f"rss:{f['name']}"] = {"ok": False, "error": str(e)[:200]}
    return out


# ───────────────────────── Hacker News ─────────────────────────

def hackernews(cfg: dict, since: datetime, status: dict) -> list[dict]:
    min_pts = cfg["sources"].get("hackernews_min_points", 40)
    ts = int(since.timestamp())
    url = ("https://hn.algolia.com/api/v1/search?tags=story&hitsPerPage=1000"
           f"&numericFilters=created_at_i>{ts},points>={min_pts}")
    data = net.get_json(url, timeout=30)
    items = []
    for h in data.get("hits", []):
        title = h.get("title") or ""
        if not AI_TITLE_RE.search(title):
            continue
        hn_url = f"https://news.ycombinator.com/item?id={h['objectID']}"
        items.append(make_item(
            source="hn", source_name="Hacker News", kind="discussion",
            title=title, url=h.get("url") or hn_url, published=parse_date(h.get("created_at")),
            summary=strip_html(h.get("story_text") or ""), engagement=h.get("points") or 0, weight=1.1,
            extra={"comments_url": hn_url, "comments": h.get("num_comments") or 0},
        ))
    status["hackernews"] = {"ok": True, "n": len(items)}
    return items


# ───────────────────────── Hugging Face Daily Papers ─────────────────────────

def hf_papers(cfg: dict, since: datetime, status: dict) -> list[dict]:
    min_up = cfg["sources"].get("papers_min_upvotes", 8)
    today = datetime.now(timezone.utc).date()
    days = max(1, (today - since.date()).days + 1)
    items, seen = [], set()
    for d in range(min(days, 10)):
        day = today - timedelta(days=d)
        try:
            data = net.get_json(f"https://huggingface.co/api/daily_papers?date={day.isoformat()}&limit=100", timeout=30)
        except net.HttpError:
            continue
        for p in data:
            paper = p.get("paper") or {}
            pid = paper.get("id")
            up = int(paper.get("upvotes") or 0)
            if not pid or pid in seen or up < min_up:
                continue
            seen.add(pid)
            when = parse_date(paper.get("submittedOnDailyAt") or p.get("publishedAt")) or datetime.combine(day, datetime.min.time(), timezone.utc)
            summary = paper.get("ai_summary") or p.get("summary") or paper.get("summary") or ""
            items.append(make_item(
                source="hf_papers", source_name="HF Daily Papers", kind="paper",
                title=p.get("title") or paper.get("title") or pid, url=f"https://huggingface.co/papers/{pid}",
                published=when, summary=summary, engagement=up, weight=1.0,
                extra={"arxiv": f"https://arxiv.org/abs/{pid}", "github": paper.get("githubRepo"),
                       "project": paper.get("projectPage"), "keywords": (paper.get("ai_keywords") or [])[:8]},
            ))
    status["hf_papers"] = {"ok": True, "n": len(items)}
    return items


# ───────────────────────── GitHub (nuovi repository in crescita) ─────────────────────────

def github_repos(cfg: dict, since: datetime, status: dict) -> list[dict]:
    src = cfg["sources"]
    min_stars = src.get("github_min_stars", 80)
    created = (datetime.now(timezone.utc) - timedelta(days=10)).date().isoformat()
    headers = {"Accept": "application/vnd.github+json"}
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    items, seen = [], set()
    errors = []
    for topic in src.get("github_topics", []):
        q = quote(f"topic:{topic} created:>={created} stars:>={min_stars}")
        try:
            data = net.get_json(f"https://api.github.com/search/repositories?q={q}&sort=stars&order=desc&per_page=30",
                                headers=headers, timeout=30, retries=1)
        except net.HttpError as e:
            errors.append(str(e))
            continue
        for r in data.get("items", []):
            if r["full_name"] in seen:
                continue
            seen.add(r["full_name"])
            items.append(make_item(
                source="github", source_name="GitHub", kind="repo",
                title=f"{r['full_name']}: {r.get('description') or ''}".strip(": "), url=r["html_url"],
                published=parse_date(r.get("created_at")), summary=r.get("description") or "",
                engagement=r.get("stargazers_count") or 0, weight=0.9,
                extra={"stars": r.get("stargazers_count"), "language": r.get("language"), "topics": r.get("topics", [])[:8]},
            ))
    status["github"] = {"ok": not errors or bool(items), "n": len(items), **({"error": errors[0][:200]} if errors else {})}
    return items


# ───────────────────────── orchestrazione ─────────────────────────

def collect_news(cfg: dict, since: datetime, status: dict) -> list[dict]:
    jobs = {"rss": rss_feeds, "hackernews": hackernews, "github": github_repos}
    items: list[dict] = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(fn, cfg, since, status): name for name, fn in jobs.items()}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                items.extend(fut.result())
            except Exception as e:
                status[name] = {"ok": False, "error": str(e)[:200]}
    # dedup per id (stesso URL da più fonti): tieni quello con più engagement e somma i segnali
    by_id: dict[str, dict] = {}
    title_to_id: dict[str, str] = {}
    for it in items:
        tkey = re.sub(r"[^a-z0-9]", "", it["title"].lower())
        if len(tkey) >= 20 and tkey in title_to_id:  # stessa notizia con URL diversi
            it["id"] = title_to_id[tkey]
        elif len(tkey) >= 20:
            title_to_id[tkey] = it["id"]
        cur = by_id.get(it["id"])
        if cur is None:
            by_id[it["id"]] = it
        else:
            cur["engagement"] = max(cur["engagement"], it["engagement"])
            cur["weight"] = max(cur["weight"], it["weight"]) + 0.2
            cur.setdefault("also_on", []).append(it["source_name"])
            if it["extra"].get("comments_url"):
                cur["extra"].setdefault("comments_url", it["extra"]["comments_url"])
    return list(by_id.values())
