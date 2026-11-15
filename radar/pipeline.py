"""Esecuzione giornaliera: raccolta → valutazione → modelli → approfondimenti → editoriale → file JSON del sito."""
from __future__ import annotations

import json
import math
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from . import models as M
from .llm import LLM
from .sources import collect_news
from .textutil import norm_name, one_line

CATS = ("modello", "feature", "tool", "paper", "guida", "industria", "altro")
RELEASE_RE = re.compile(r"\b(releas\w*|introduc\w*|launch\w*|announc\w*|now available|now in|is out|now live|"
                        r"rolling out|preview|open[- ]sourc\w*|open[- ]weights?|unveil\w*|debut\w*|rilasciat\w*|"
                        r"disponibile)\b", re.I)
VERSIONED_MODEL_RE = re.compile(r"\b(qwen|llama|gemma|gemini|claude|gpt|deepseek|mistral|kimi|glm|grok|nemotron|phi|"
                                r"fable|opus|sonnet|haiku|mythos|minimax|muse|ernie|hunyuan|granite|olmo)[\s-]?v?\d", re.I)
MODEL_WORD_RE = re.compile(r"\b(model|llm|weights|checkpoint|\d+(\.\d+)?b\b|gguf|qwen|llama|gemma|mistral|deepseek|"
                           r"claude|gpt|gemini|kimi|glm|nemotron|phi)\b", re.I)
ORG_HINTS = [
    (r"claude|fable|mythos|opus|sonnet|haiku", "Anthropic"), (r"\bgpt|\bo\d\b|sora|codex", "OpenAI"),
    (r"gemini|gemma|veo|imagen|lyria", "Google"), (r"qwen", "Qwen (Alibaba)"), (r"deepseek", "DeepSeek"),
    (r"llama|muse", "Meta"), (r"mistral|magistral|devstral|codestral", "Mistral"), (r"kimi", "Moonshot AI"),
    (r"glm", "Z.ai"), (r"grok", "xAI"), (r"minimax", "MiniMax"), (r"nemotron", "NVIDIA"), (r"\bphi", "Microsoft"),
]


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def save_json(path: Path, data, indent=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=indent))


# ───────────────────────── prompt ─────────────────────────

def system_prompt(cfg: dict, role: str) -> str:
    return (f"{role}\n\nProfilo del lettore:\n{cfg['profile'].get('description', '').strip()}\n\n"
            "Scrivi sempre in italiano. Rispondi SOLO con JSON valido, senza testo prima o dopo.")


TRIAGE_RULES = """Valuta questi {n} elementi. Per ognuno restituisci un oggetto:
{{"i": <numero>, "cat": "modello|feature|tool|paper|guida|industria|altro", "rel": <0-10>, "title": "...", "summary": "...", "why": "...", "models": ["..."], "release": true|false}}

Regole:
- cat: "modello" = rilascio/annuncio di un nuovo modello AI (LLM, embedding, immagini, voce…); "feature" = nuova funzionalità di un prodotto, API o tool esistente; "tool" = libreria, repo o app; "paper" = ricerca; "guida" = tutorial, analisi, benchmark indipendenti, esperienze d'uso; "industria" = business, policy, persone; "altro" = il resto.
- rel: 9-10 imperdibile per QUESTO lettore; 7-8 molto utile; 4-6 interessante; 0-3 rumore, meme, clickbait, fuori tema.
- title: titolo italiano chiaro e informativo (max 90 caratteri), conserva nomi propri e numeri.
- summary: 1-2 frasi italiane con i fatti concreti (cosa, chi, numeri chiave). Non aggiungere nulla che non sia nel testo.
- why: se rel >= 6, una frase su perché interessa a questo lettore; altrimenti "".
- models: nomi esatti dei modelli AI NUOVI annunciati o rilasciati nell'elemento (es. "Qwen4-Max Preview"); [] se nessuno. Non elencare modelli citati solo come confronto.
- release: true solo se l'elemento annuncia direttamente il rilascio di un nuovo modello o versione.

Rispondi con {{"items": [ ... ]}} contenente tutti gli elementi.

Elementi:
{block}"""

DEEP_RULES = """Modello da analizzare: {name} ({org})
Dati già noti: {known}

Testo (annuncio ufficiale o model card):
\"\"\"
{text}
\"\"\"

Restituisci:
{{"tldr": "2-3 frasi: cos'è, cosa ha di nuovo, per cosa è adatto",
 "features": ["fino a 6 punti concreti: architettura, contesto, multimodalità, reasoning, tool use, licenza, novità tecniche"],
 "benchmarks": [{{"name": "benchmark", "score": "valore di {name} come scritto nel testo", "vs": [{{"model": "concorrente", "score": "valore"}}]}}],
 "params": "parametri dichiarati (es. '27B dense', '35B totali / 3B attivi') o ''",
 "context": "contesto massimo dichiarato o ''",
 "availability": "dove e come si usa: pesi aperti, API, prezzo, piattaforme; o ''",
 "local_tips": "consigli per l'esecuzione locale presenti nel testo (sampling, framework, quantizzazioni ufficiali) o ''"}}
Usa solo dati presenti nel testo. Massimo 12 benchmark, preferendo i più noti (SWE-bench, GPQA, HLE, AIME, LiveCodeBench, MMLU-Pro, Terminal-Bench, ARC-AGI, MTEB…); in "vs" al massimo 3 concorrenti della stessa tabella. Se il testo riguarda più modelli, concentrati su {name}."""

DIGEST_RULES = """Data: {date}.

Notizie più rilevanti di oggi (rel = rilevanza 0-10):
{news}

Modelli nuovi o aggiornati oggi:
{models}

Restituisci:
{{"headline": "titolo della giornata (max 80 caratteri)",
 "bullets": ["3-6 punti di 1-2 frasi: i fatti più importanti per il lettore, con nomi e numeri"],
 "gpu_note": "una frase su cosa c'è di nuovo da far girare sulla sua {gpu}, o '' se niente"}}"""


# ───────────────────────── valutazione notizie ─────────────────────────

ENGAGEMENT_SCALE = {"hn": 30, "github": 150, "hf_papers": 8, "rss": 1}


def heuristic(it: dict, kws: list[str]) -> float:
    text = (it["title"] + " " + it["summary"][:400]).lower()
    hits = sum(1 for k in kws if k in text)
    eng = math.log10(1 + it["engagement"] / ENGAGEMENT_SCALE.get(it["source"], 1))
    s = it["weight"] * (1 + 0.5 * min(hits, 6) + 1.2 * eng)
    if RELEASE_RE.search(it["title"]):
        s += 1.5
    if VERSIONED_MODEL_RE.search(it["title"]):
        s += 1.5
    return round(s, 2)


def guess_cat(it: dict) -> str:
    if it["kind"] == "paper":
        return "paper"
    if it["kind"] == "repo":
        return "tool"
    t = it["title"].lower()
    if RELEASE_RE.search(t) and MODEL_WORD_RE.search(t):
        return "modello"
    if re.search(r"\b(v\d+\.\d+|update|now supports|support for|feature|api|release)\b", t):
        return "feature"
    if re.search(r"\b(how|guide|tutorial|benchmark|tested|tried|running|vs\.?)\b", t):
        return "guida"
    return "altro"


def fallback_fields(it: dict):
    it.setdefault("cat", guess_cat(it))
    it.setdefault("rel", min(10, max(0, round(it["h"] * 1.3))))
    it.setdefault("llm", False)


def triage(llm: LLM, cfg: dict, items: list[dict]) -> int:
    """Valuta gli item a lotti con l'LLM. Restituisce quanti item sono stati valutati dall'LLM."""
    if not items or not llm.available:
        return 0
    bs = int(cfg["llm"]["triage_batch"])
    sysmsg = system_prompt(cfg, "Sei l'editor di una rassegna quotidiana su intelligenza artificiale scritta per UN solo lettore. "
                                "Valuti notizie grezze (blog, Hacker News, Reddit, paper, repository) per utilità e rilevanza.")
    done = 0
    for start in range(0, len(items), bs):
        if not llm.available:
            break
        batch = items[start:start + bs]
        lines = []
        for i, it in enumerate(batch):
            eng = f", {int(it['engagement'])} punti" if it["engagement"] else ""
            lines.append(f"[{i}] ({it['source_name']}{eng}) {it['title']}\n    {one_line(it['summary'], 420)}")
        user = TRIAGE_RULES.format(n=len(batch), block="\n".join(lines))
        res = llm.chat_json(sysmsg, user, max_tokens=6000,
                            validate=lambda r: isinstance(r, dict) and isinstance(r.get("items"), list) and r["items"])
        if not res:
            continue
        for row in res["items"]:
            try:
                it = batch[int(row.get("i"))]
            except (TypeError, ValueError, IndexError):
                continue
            cat = str(row.get("cat", "")).lower().strip()
            it["cat"] = cat if cat in CATS else guess_cat(it)
            try:
                it["rel"] = max(0, min(10, round(float(row.get("rel", 0)))))
            except (TypeError, ValueError):
                it["rel"] = 0
            it["title_it"] = one_line(str(row.get("title") or ""), 140) or None
            it["summary_it"] = one_line(str(row.get("summary") or ""), 500) or None
            it["why"] = one_line(str(row.get("why") or ""), 240)
            ms = row.get("models") or []
            it["model_names"] = [one_line(str(m), 80) for m in ms if isinstance(m, str) and m.strip()][:5]
            it["release"] = bool(row.get("release"))
            it["llm"] = True
            done += 1
    return done


# ───────────────────────── modelli ─────────────────────────

def merge_or_fields(hfe: dict, ore: dict) -> dict:
    for k in ("or_id", "pricing", "free_on_openrouter", "modality", "knowledge_cutoff", "description"):
        if ore.get(k) not in (None, "", {}):
            hfe[k] = ore[k]
    if ore.get("context") and not hfe.get("context"):
        hfe["context"] = ore["context"]
    hfe.setdefault("links", {}).update(ore.get("links", {}))
    return hfe


def guess_org(name: str) -> str:
    for pat, org in ORG_HINTS:
        if re.search(pat, name, re.I):
            return org
    return ""


def resolve_mention(name: str, it: dict, reg: M.Registry, or_index: list, cfg: dict, today: str) -> dict | None:
    hw = cfg["hardware"]
    e = reg.find(name)
    if e:
        return e
    m = M.match_openrouter(name, or_index)
    if m:
        ore = M.entry_from_or(m)
        if ore.get("hf_id"):
            api = M.hf_model(ore["hf_id"])
            if api:
                return reg.add(merge_or_fields(M.entry_from_hf(api, hw), ore), today)
        return reg.add(ore, today)
    api, others = M.search_hf(name, cfg)
    if api:
        return reg.add(M.entry_from_hf(api, hw), today)
    if not it.get("release"):
        return None
    e = {
        "id": f"ann:{norm_name(name)}",
        "name": name,
        "org": guess_org(name),
        "kind": "announced",
        "released": (it.get("published") or today)[:10],
        "links": {"announcement": it["url"]},
    }
    others = others or M.family_placeholders(name)
    if others:
        e["note"] = ("Nessun peso ufficiale su Hugging Face: esistono solo repo di terze parti o segnaposto ("
                     + ", ".join(others[:3]) + ").")
    return reg.add(e, today)


def refresh_pending(reg: M.Registry, or_index: list, cfg: dict, today: str):
    """Ricontrolla modelli 'annunciati' o 'segnaposto' recenti: se escono i pesi o l'API, aggiorna la scheda."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=45)).date().isoformat()
    hw = cfg["hardware"]
    for e in list(reg.models.values()):
        if (e.get("first_seen") or "") < cutoff:
            continue
        if e["kind"] == "placeholder" and e.get("hf_id"):
            api = M.hf_model(e["hf_id"])
            if api and M.has_weights(api.get("siblings")):
                new = M.entry_from_hf(api, hw)
                new["note"] = "Pesi pubblicati: il repository ora contiene il modello."
                reg.add(new, today)
                e["updated"] = today
                reg.touched.add(e["id"])
        elif e["kind"] == "announced":
            m = M.match_openrouter(e["name"], or_index)
            if m:
                ore = M.entry_from_or(m)
                for k in ("or_id", "pricing", "context", "modality", "free_on_openrouter", "knowledge_cutoff", "description"):
                    if ore.get(k) not in (None, "", {}):
                        e[k] = ore[k]
                e["links"].update(ore["links"])
                e["kind"] = "open" if ore.get("hf_id") else "closed"
                e["updated"] = today
                reg.touched.add(e["id"])


def discover_models(reg: M.Registry, or_index: list, cfg: dict, status: dict, today: str):
    hw = cfg["hardware"]
    for m in M.discover_openrouter(or_index, cfg):
        ore = M.entry_from_or(m)
        if ore["id"] in reg.models and "arch" in reg.models[ore["id"]]:
            merge_or_fields(reg.models[ore["id"]], ore)
            continue
        reg.add(ore, today)

    hf_cands = {m["id"]: m for m in M.discover_hf(cfg, status)}
    # modelli OpenRouter con pesi su HF non ancora arricchiti
    for e in list(reg.models.values()):
        if e.get("hf_id") and "arch" not in e and e["hf_id"] not in hf_cands:
            hf_cands[e["hf_id"]] = {"id": e["hf_id"], "_need_api": True}
    todo = [m for m in hf_cands.values()
            if m["id"] not in reg.models or "arch" not in reg.models[m["id"]]]

    def enrich(m):
        api = M.hf_model(m["id"]) if m.get("_need_api") or "safetensors" not in m else m
        if not api:
            return None
        try:
            e = M.entry_from_hf(api, hw)
        except Exception as ex:
            log(f"  modello ignorato {m['id']}: {type(ex).__name__}: {ex}")
            return None
        if m.get("_trend_rank"):
            e["signals"]["trend_rank"] = m["_trend_rank"]
        if m.get("_watched"):
            e["signals"]["watched_org"] = True
        return e

    with ThreadPoolExecutor(max_workers=6) as ex:
        for m, e in zip(todo, ex.map(enrich, todo)):
            if not e:
                cur = reg.models.get(m["id"])
                if cur and cur.get("or_id") and cur["kind"] == "open":
                    cur["kind"] = "closed"
                    cur["note"] = ("OpenRouter indica pesi su Hugging Face, ma il repository non è (ancora) pubblico: "
                                   "verrà ricontrollato ogni giorno.")
                continue
            existing = reg.models.get(e["id"])
            if existing and existing.get("or_id"):
                merge_or_fields(e, existing)
            reg.add(e, today)
    # aggiorna i segnali di popolarità dei modelli già noti
    for mid, m in hf_cands.items():
        if mid in reg.models and "likes" in m:
            reg.models[mid].setdefault("signals", {}).update(
                {"likes": m.get("likes", 0), "downloads": m.get("downloads", 0), "trending": m.get("trendingScore")})


def normalize_deep(d: dict) -> dict:
    def s(x, n=600):
        return one_line(str(x), n) if x not in (None, [], {}) else ""
    benches = []
    for b in (d.get("benchmarks") or [])[:14]:
        if not isinstance(b, dict) or not b.get("name") or b.get("score") in (None, ""):
            continue
        vs = [{"model": s(v.get("model"), 60), "score": s(v.get("score"), 30)}
              for v in (b.get("vs") or [])[:3] if isinstance(v, dict) and v.get("model")]
        benches.append({"name": s(b["name"], 80), "score": s(b["score"], 30), "vs": vs})
    return {
        "tldr": s(d.get("tldr"), 700),
        "features": [s(f, 260) for f in (d.get("features") or [])[:7] if isinstance(f, str) and f.strip()],
        "benchmarks": benches,
        "params": s(d.get("params"), 120),
        "context": s(d.get("context"), 80),
        "availability": s(d.get("availability"), 400),
        "local_tips": s(d.get("local_tips"), 500),
    }


def deep_dives(llm: LLM, reg: M.Registry, cfg: dict, items_by_id: dict, today: str, candidates: list[dict]) -> int:
    n = 0
    sysmsg = system_prompt(cfg, "Sei un analista tecnico di modelli AI. Estrai informazioni SOLO dal testo fornito; "
                                "se un dato manca lascialo vuoto. Non inventare numeri.")
    for e in candidates[: int(cfg["llm"]["max_deep_dives"])]:
        if not llm.available:
            break
        text, src = M.fetch_source_text(e, items_by_id)
        if len(text) < 200:
            continue
        known = {k: e.get(k) for k in ("params_b", "active_b", "context", "license", "pricing", "pipeline") if e.get(k)}
        user = DEEP_RULES.format(name=e["name"], org=e.get("org") or "", known=json.dumps(known, ensure_ascii=False), text=text)
        res = llm.chat_json(sysmsg, user, max_tokens=3500, validate=lambda r: isinstance(r, dict) and r.get("tldr"))
        if res:
            deep = normalize_deep(res)
            deep.update({"source": src, "date": today, "by": llm.last_model})
            e["deep"] = deep
            n += 1
            log(f"  approfondito: {e['name']} ({len(deep['benchmarks'])} benchmark)")
    return n


# ───────────────────────── output ─────────────────────────

def item_out(it: dict) -> dict:
    ex = it.get("extra", {})
    return {
        "id": it["id"],
        "title": it.get("title_it") or it["title"],
        "title_orig": it["title"],
        "url": it["url"],
        "source": it["source_name"],
        "also_on": it.get("also_on", []),
        "kind": it["kind"],
        "cat": it.get("cat", "altro"),
        "rel": it.get("rel", 0),
        "summary": it.get("summary_it") or one_line(it["summary"], 300),
        "why": it.get("why", ""),
        "published": it.get("published"),
        "engagement": it["engagement"],
        "comments_url": ex.get("comments_url"),
        "links": {k: ex[k] for k in ("github", "arxiv", "project") if ex.get(k)},
        "models": it.get("model_ids", []),
        "llm": it.get("llm", False),
    }


def make_digest(llm: LLM, cfg: dict, date: str, items: list[dict], models_today: list[dict]):
    if not llm.available or not items:
        return None
    top = sorted(items, key=lambda i: (-i["rel"], -i["engagement"]))[:25]
    news = "\n".join(f"- [rel {i['rel']}] {i['title']} — {i['summary']}" for i in top)
    mods = "\n".join(
        f"- {e['name']} ({e.get('org', '')}, {e['kind']}"
        + (f", {e['params_b']:g}B" if e.get("params_b") else "")
        + (f", verdetto GPU: {e['vram']['text']}" if e.get("vram") else "")
        + (f", prezzo ${e['pricing']['in']}/${e['pricing']['out']} per M token" if (e.get("pricing") or {}).get("in") else "")
        + ")" for e in models_today[:12]) or "(nessuno)"
    sysmsg = system_prompt(cfg, "Scrivi l'apertura della rassegna AI quotidiana per questo lettore: concreta, senza enfasi.")
    res = llm.chat_json(sysmsg, DIGEST_RULES.format(date=date, news=news, models=mods, gpu=cfg["hardware"]["gpu_name"]),
                        max_tokens=2000, validate=lambda r: isinstance(r, dict) and r.get("bullets"))
    if not res:
        return None
    return {
        "headline": one_line(str(res.get("headline") or ""), 120),
        "bullets": [one_line(str(b), 400) for b in res.get("bullets", [])[:7] if b],
        "gpu_note": one_line(str(res.get("gpu_note") or ""), 300),
        "by": llm.last_model,
    }


def run(root: Path, cfg: dict, *, use_llm: bool = True, bootstrap_days: int | None = None) -> dict:
    t0 = time.time()
    tz = ZoneInfo(cfg["run"]["timezone"])
    now = datetime.now(timezone.utc)
    today = now.astimezone(tz).date().isoformat()
    data_dir, state_dir = root / "docs" / "data", root / "state"
    seen = load_json(state_dir / "seen.json", {"items": {}})
    first_run = not seen["items"]
    hours = (bootstrap_days or cfg["run"]["bootstrap_days"]) * 24 if (first_run or bootstrap_days) else cfg["run"]["lookback_hours"]
    since = now - timedelta(hours=hours)
    status: dict = {}
    llm = LLM(cfg, enabled=use_llm)
    reg = M.Registry(data_dir / "models.json")
    log(f"AI Radar {today} — finestra {hours}h, LLM {'attivo' if llm.available else 'non configurato'}")

    # 1. notizie
    items = collect_news(cfg, since, status)
    new = [i for i in items if i["id"] not in seen["items"]]
    kws = [k.lower() for k in cfg["profile"].get("keywords", [])]
    for it in new:
        it["h"] = heuristic(it, kws)
    new.sort(key=lambda i: -i["h"])
    log(f"Notizie raccolte: {len(items)} (nuove: {len(new)})")

    # 2. valutazione con LLM
    to_triage = new[: int(cfg["llm"]["triage_items"])]
    n_triaged = triage(llm, cfg, to_triage)
    for it in new:
        fallback_fields(it)
    log(f"Valutate dall'LLM: {n_triaged}/{len(to_triage)} (chiamate finora: {llm.calls})")

    # 3. modelli
    or_index = M.openrouter_index(status)
    discover_models(reg, or_index, cfg, status, today)
    items_by_id = {i["id"]: i for i in new}
    for it in new:
        if not it.get("model_names") or it["rel"] < 5:
            continue
        for name in it["model_names"]:
            e = resolve_mention(name, it, reg, or_index, cfg, today)
            if not e:
                continue
            it.setdefault("model_ids", []).append(e["id"])
            if not any(m["item"] == it["id"] for m in e.setdefault("mentions", [])):
                e["mentions"].insert(0, {"item": it["id"], "rel": it["rel"], "date": (it.get("published") or today)[:10],
                                         "title": it.get("title_it") or it["title"], "url": it["url"],
                                         "source": it["source_name"]})
                e["mentions"] = e["mentions"][:8]
    refresh_pending(reg, or_index, cfg, today)
    for e in reg.models.values():
        e["score"] = M.priority(e, cfg)
    mentioned_today = {mid for it in new if it["rel"] >= 6 for mid in it.get("model_ids", [])}
    today_ids = (reg.touched | mentioned_today) & set(reg.models)
    models_today = sorted((reg.models[i] for i in today_ids), key=lambda e: -e["score"])
    log(f"Modelli: {len(reg.models)} nel registro, {len(reg.touched)} nuovi/aggiornati, {len(mentioned_today)} citati oggi")

    # 4. approfondimenti (benchmark, feature) sui modelli più importanti
    cands = [e for e in models_today if not e.get("deep") and e["kind"] != "placeholder"]
    n_deep = deep_dives(llm, reg, cfg, items_by_id, today, cands)
    M.attach_aa(reg, M.aa_models(status))

    # 5. pagina del giorno
    day_path = data_dir / "days" / f"{today}.json"
    prev = load_json(day_path, None)
    if n_triaged:
        triaged_ids = {i["id"] for i in to_triage}
        keep = [i for i in new if (i["llm"] and i["rel"] >= cfg["run"]["min_relevance"])
                or (not i["llm"] and i["id"] in triaged_ids)]
    else:
        keep = new
    out_items = [item_out(i) for i in keep]
    if prev:
        known = {i["id"] for i in out_items}
        out_items += [i for i in prev.get("items", []) if i["id"] not in known]
        today_ids |= set(prev.get("models", [])) & set(reg.models)
        models_today = sorted((reg.models[i] for i in today_ids), key=lambda e: -e["score"])
    out_items.sort(key=lambda i: (-i["rel"], -i["engagement"]))
    out_items = out_items[: int(cfg["run"]["max_items_per_day"])]
    digest = make_digest(llm, cfg, today, out_items, models_today) if (new or not prev) else None
    if digest is None and prev:
        digest = prev.get("digest")

    day = {
        "date": today,
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "digest": digest,
        "items": out_items,
        "models": [e["id"] for e in models_today],
        "stats": {"collected": len(items), "new": len(new), "triaged": n_triaged, "deep_dives": n_deep},
    }
    save_json(day_path, day)
    for mid in reg.touched:
        reg.models[mid].setdefault("updated", today)
    reg.save()

    for it in new:
        seen["items"][it["id"]] = today
    cutoff = (now - timedelta(days=120)).date().isoformat()
    seen["items"] = {k: v for k, v in seen["items"].items() if v >= cutoff}
    save_json(state_dir / "seen.json", seen)

    index = load_json(data_dir / "index.json", {"days": []})
    index["days"] = [d for d in index.get("days", []) if d["date"] != today]
    index["days"].insert(0, {"date": today, "headline": (digest or {}).get("headline", ""),
                             "n_items": len(out_items), "n_models": len(day["models"])})
    index["days"].sort(key=lambda d: d["date"], reverse=True)
    index.update({
        "updated": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hardware": cfg["hardware"],
        "status": {
            "sources": status,
            "llm": {"enabled": bool(llm.providers and any(p.key for p in llm.providers)), "calls": llm.calls,
                    "failures": llm.failures, "models": llm.used_models, "log": llm.log[-15:]},
            "duration_s": round(time.time() - t0),
        },
    })
    save_json(data_dir / "index.json", index, indent=1)
    log(f"Fatto in {round(time.time() - t0)}s: {len(out_items)} notizie, {len(day['models'])} modelli, "
        f"{llm.calls} chiamate LLM ({llm.failures} fallite)")
    for line in llm.log[-8:]:
        log(f"  llm: {line}")
    bad = [k for k, v in status.items() if not v.get("ok")]
    if bad:
        log(f"Fonti con errori: {', '.join(bad)}")
    return day
