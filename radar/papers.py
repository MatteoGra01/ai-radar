"""Paper per argomento: raccolta (HF Daily Papers, arXiv per argomento, feed come Alignment Forum),
classificazione con LLM (argomenti, importanza, riassunto in italiano) e archivio a finestra mobile."""
from __future__ import annotations

import json
import math
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

from . import net
from .textutil import iso, one_line, parse_date, parse_feed, strip_html

ATOM = "{http://www.w3.org/2005/Atom}"


def _log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _arxiv_id(s: str) -> str | None:
    m = re.search(r"(\d{4}\.\d{4,5})", s or "")
    return m.group(1) if m else None


def _paper(pid, title, summary, published, url, source, **kw) -> dict:
    return {"id": pid, "title": one_line(title, 300), "abstract": one_line(strip_html(summary), 1600),
            "published": iso(published), "url": url, "source": source, "upvotes": 0, "authors": [],
            "links": {}, **kw}


# ───────────────────────── raccolta ─────────────────────────

def hf_daily(since: datetime) -> list[dict]:
    today = datetime.now(timezone.utc).date()
    out = []
    for d in range(min(10, max(1, (today - since.date()).days + 1))):
        day = today - timedelta(days=d)
        try:
            data = net.get_json(f"https://huggingface.co/api/daily_papers?date={day.isoformat()}&limit=100", timeout=30)
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for p in data:
            pp = p.get("paper") or {}
            pid = pp.get("id")
            if not pid:
                continue
            authors = [a.get("name") for a in (pp.get("authors") or [])[:4] if isinstance(a, dict) and a.get("name")]
            when = parse_date(pp.get("submittedOnDailyAt") or p.get("publishedAt"))
            orig = parse_date(pp.get("publishedAt"))  # data del paper (arXiv); "when" = comparsa su HF Papers
            org = p.get("organization") if isinstance(p.get("organization"), dict) else {}
            out.append(_paper(
                pid, p.get("title") or pp.get("title") or pid, pp.get("summary") or p.get("summary") or "", orig or when,
                f"https://huggingface.co/papers/{pid}", "HF Papers", hf_date=iso(when),
                upvotes=int(pp.get("upvotes") or 0), authors=authors,
                org=org.get("fullname") or org.get("name"),
                links={k: v for k, v in {"arxiv": f"https://arxiv.org/abs/{pid}", "pdf": f"https://arxiv.org/pdf/{pid}",
                                         "github": pp.get("githubRepo"), "project": pp.get("projectPage")}.items() if v},
            ))
    return out


def arxiv_topic(topic: dict, n: int, since: datetime) -> list[dict]:
    phrases = " OR ".join(f'abs:"{p}"' for p in topic.get("arxiv", []))
    q = f"({phrases}) AND (cat:cs.LG OR cat:cs.CL OR cat:cs.AI OR cat:cs.CV OR cat:cs.RO OR cat:stat.ML)"
    url = (f"https://export.arxiv.org/api/query?search_query={quote(q)}"
           f"&sortBy=submittedDate&sortOrder=descending&max_results={n}")
    root = ET.fromstring(net.fetch(url, timeout=40, retries=2))
    out = []
    for e in root.findall(f"{ATOM}entry"):
        pid = _arxiv_id(e.findtext(f"{ATOM}id") or "")
        pub = parse_date(e.findtext(f"{ATOM}published"))
        if not pid or (pub and pub < since):
            continue
        authors = [a.findtext(f"{ATOM}name") for a in e.findall(f"{ATOM}author")][:4]
        out.append(_paper(pid, e.findtext(f"{ATOM}title") or "", e.findtext(f"{ATOM}summary") or "", pub,
                          f"https://arxiv.org/abs/{pid}", "arXiv", authors=[a for a in authors if a],
                          links={"arxiv": f"https://arxiv.org/abs/{pid}", "pdf": f"https://arxiv.org/pdf/{pid}"},
                          hint=topic["id"]))
    return out


def feed_posts(feed: dict, since: datetime) -> list[dict]:
    out = []
    for e in parse_feed(net.get_text(feed["url"], timeout=30, retries=1)):
        if not e["url"] or (e["published"] and e["published"] < since):
            continue
        pid = "post:" + re.sub(r"[^a-z0-9]+", "-", e["url"].lower())[-80:]
        out.append(_paper(pid, e["title"], e["summary"], e["published"], e["url"], feed["name"], kind="post"))
    return out


def collect(cfg: dict, since: datetime, status: dict) -> list[dict]:
    pc = cfg.get("papers", {})
    topics = pc.get("topics", [])
    found: dict[str, dict] = {}

    def add(p):
        cur = found.get(p["id"])
        if cur is None:
            found[p["id"]] = p
            return
        # stesso paper da più fonti: tieni i dati più ricchi (HF ha upvote e link GitHub)
        cur["upvotes"] = max(cur["upvotes"], p["upvotes"])
        cur["links"] = {**p["links"], **cur["links"]}
        if p["source"] == "HF Papers":
            cur.update({"url": p["url"], "source": "HF Papers", "org": p.get("org"), "hf_date": p.get("hf_date")})
        if p.get("hint"):
            cur.setdefault("hints", []).append(p["hint"])

    try:
        hf = hf_daily(since)
        for p in hf:
            add(p)
        status["papers:hf"] = {"ok": True, "n": len(hf)}
    except Exception as e:
        status["papers:hf"] = {"ok": False, "error": str(e)[:200]}

    n_ax, errs = 0, []
    for t in topics:  # arXiv chiede di non superare una richiesta ogni 3 secondi
        try:
            ps = arxiv_topic(t, int(pc.get("arxiv_per_topic", 30)), since)
            for p in ps:
                p.setdefault("hints", [p.pop("hint")])
                add(p)
            n_ax += len(ps)
        except Exception as e:
            errs.append(f"{t['id']}: {str(e)[:80]}")
        time.sleep(3.1)
    status["papers:arxiv"] = {"ok": len(errs) < len(topics) or not topics, "n": n_ax, **({"error": "; ".join(errs)[:200]} if errs else {})}

    for f in pc.get("feeds", []):
        try:
            ps = feed_posts(f, since)
            for p in ps:
                add(p)
            status[f"papers:{f['name']}"] = {"ok": True, "n": len(ps)}
        except Exception as e:
            status[f"papers:{f['name']}"] = {"ok": False, "error": str(e)[:200]}
    return list(found.values())


# ───────────────────────── classificazione ─────────────────────────

PAPER_RULES = """Classifica questi {n} paper/post di ricerca. Argomenti possibili (id: descrizione):
{topics}

Per ognuno restituisci:
{{"i": <numero>, "topics": ["id", ...], "sci": <0-10>, "imp": <0-10>, "title": "...", "summary": "...", "why": "..."}}
- topics: 1-3 id dalla lista sopra (solo quelli davvero centrali nel lavoro); [] se nessuno è pertinente.
- sci: valore scientifico NEL SUO CAMPO, indipendente dal lettore: 9-10 risultato di rilievo o idea nuova che sposta
  il campo; 7-8 contributo solido (metodo nuovo, evidenze rigorose, risorsa importante); 4-6 incrementale; 0-3 marginale
  o poco chiaro. Giudica novità, rigore e portata; un paper teorico o di interpretabilità vale anche senza applicazioni.
- imp: rilevanza per QUESTO lettore (profilo e argomenti sopra): quanto è vicino ai suoi interessi e quanto può usarlo.
- title: titolo in italiano, breve e chiaro (max 100 caratteri), mantieni i nomi propri dei metodi.
- summary: 1-2 frasi in italiano: cosa fanno e il risultato principale (con numeri se presenti). Solo fatti dal testo.
- why: se sci >= 7 o imp >= 7, una frase sul perché conta: il contributo nel suo campo e, se c'è, l'uso pratico.
  Cita GPU, VRAM o esecuzione locale SOLO se il paper riguarda davvero efficienza, quantizzazione o inferenza;
  altrimenti "".
Rispondi con {{"items": [ ... ]}} contenente tutti gli elementi.

Paper:
{block}"""


def classify(llm, cfg: dict, papers: list[dict], system_prompt) -> int:
    if not papers or not llm.available:
        return 0
    topics = cfg["papers"]["topics"]
    ids = {t["id"] for t in topics}
    tlist = "\n".join(f"- {t['id']}: {t['label']} — {t['desc']}" for t in topics)
    sysmsg = system_prompt(cfg, "Sei un ricercatore che seleziona la letteratura AI per un collega: preciso, sintetico, "
                                "senza enfasi. Valuti paper e post di ricerca.")
    bs = 12
    batches = [papers[s:s + bs] for s in range(0, len(papers), bs)]

    def run(arg) -> int:
        k, batch = arg
        if not llm.available:
            return 0
        block = "\n".join(f"[{i}] ({p['source']}{', ' + str(p['upvotes']) + ' voti' if p['upvotes'] else ''}) "
                          f"{p['title']}\n    {p['abstract'][:900]}" for i, p in enumerate(batch))
        res = llm.chat_json(sysmsg, PAPER_RULES.format(n=len(batch), topics=tlist, block=block), max_tokens=5000,
                            label=f"paper {k + 1}/{len(batches)}",
                            validate=lambda r: isinstance(r, dict) and isinstance(r.get("items"), list) and r["items"])
        if not res:
            return 0
        done = 0
        for row in res["items"]:
            if not isinstance(row, dict):
                continue
            try:
                p = batch[int(row.get("i"))]
            except (TypeError, ValueError, IndexError):
                continue
            p["topics"] = [t for t in (row.get("topics") or []) if t in ids][:3]
            for k in ("imp", "sci"):
                try:
                    p[k] = max(0, min(10, round(float(row.get(k, 0)))))
                except (TypeError, ValueError):
                    p[k] = 0
            p["title_it"] = one_line(str(row.get("title") or ""), 160) or None
            p["summary_it"] = one_line(str(row.get("summary") or ""), 500) or None
            p["why"] = one_line(str(row.get("why") or ""), 240)
            p["llm"] = True
            done += 1
        return done

    with ThreadPoolExecutor(max_workers=int(cfg["llm"].get("parallel", 4))) as ex:
        return sum(ex.map(run, enumerate(batches)))


OVERVIEW_RULES = """Per ognuno di questi {n} paper scrivi una overview in italiano SEMPLICE, per un lettore tecnico ma non
specialista di quel sotto-campo. Niente gergo non spiegato: se usi un termine tecnico, chiariscilo tra parentesi in poche
parole. Frasi brevi e concrete; un'analogia solo se aiuta davvero.

Per ognuno restituisci:
{{"i": <numero>,
 "problema": "1-2 frasi: che problema affrontano e perché è importante o difficile",
 "idea": "2-3 frasi: come lo risolvono, l'intuizione chiave del metodo",
 "risultati": "1-2 frasi con i risultati principali e i numeri presenti nel testo",
 "limiti": "1 frase: limiti, condizioni o dubbi che emergono dal testo, oppure ''",
 "glossario": [{{"termine": "...", "spiegazione": "max 20 parole"}}]}}
Nel glossario 0-3 termini chiave necessari per capire il paper. Usa SOLO informazioni del testo, non inventare numeri.
Rispondi con {{"items": [ ... ]}} contenente tutti gli elementi.

Paper:
{block}"""


def _ov_clean(row: dict) -> dict | None:
    ov = {k: one_line(str(row.get(k) or ""), 600) for k in ("problema", "idea", "risultati", "limiti")}
    if not ov["problema"] or not ov["idea"]:
        return None
    gl = []
    for g in (row.get("glossario") or [])[:3]:
        if isinstance(g, dict) and g.get("termine") and g.get("spiegazione"):
            gl.append({"t": one_line(str(g["termine"]), 60), "d": one_line(str(g["spiegazione"]), 200)})
    ov["glossario"] = gl
    return ov


def overviews(llm, cfg: dict, papers: list[dict], system_prompt) -> dict[str, dict]:
    """Overview più dettagliata (problema, idea, risultati, limiti, glossario) per i paper passati, che devono
    avere il campo "abstract". Restituisce {id: overview}."""
    papers = [p for p in papers if (p.get("abstract") or "").strip()]
    if not papers or not llm.available:
        return {}
    sysmsg = system_prompt(cfg, "Sei un divulgatore scientifico rigoroso: spieghi la ricerca AI in modo chiaro e semplice, "
                                "senza perdere precisione e senza aggiungere informazioni che non sono nel testo.")
    bs = 6
    batches = [papers[i:i + bs] for i in range(0, len(papers), bs)]
    out: dict[str, dict] = {}

    def run(arg):
        k, batch = arg
        block = "\n\n".join(f"[{i}] {p['title']}\n{p['abstract'][:1500]}" for i, p in enumerate(batch))
        res = llm.chat_json(sysmsg, OVERVIEW_RULES.format(n=len(batch), block=block), max_tokens=4500,
                            label=f"overview {k + 1}/{len(batches)}",
                            validate=lambda r: isinstance(r, dict) and isinstance(r.get("items"), list) and r["items"])
        got = {}
        for row in (res or {}).get("items", []):
            try:
                pp = batch[int(row.get("i"))]
            except (TypeError, ValueError, IndexError, AttributeError):
                continue
            ov = _ov_clean(row) if isinstance(row, dict) else None
            if ov:
                got[pp["id"]] = ov
        return got

    with ThreadPoolExecutor(max_workers=int(cfg["llm"].get("parallel", 4))) as ex:
        for got in ex.map(run, enumerate(batches)):
            out.update(got)
    return out


def details_path(root: Path) -> Path:
    return root / "docs" / "data" / "paper_details.json"


def load_details(root: Path) -> dict:
    try:
        return json.loads(details_path(root).read_text()).get("papers", {})
    except Exception:
        return {}


def save_details(root: Path, details: dict, papers: dict):
    """Abstract e overview in un file a parte, caricato dal sito solo quando apri un paper."""
    keep = {k: v for k, v in details.items() if k in papers}
    details_path(root).write_text(json.dumps({"papers": keep}, ensure_ascii=False))


def fill_overviews(root: Path, cfg: dict, llm, system_prompt, papers: dict, details: dict, limit: int) -> int:
    """Genera le overview mancanti per i paper con punteggio sufficiente, dai più rilevanti."""
    min_score = float(cfg["papers"].get("overview_min_score", 6))
    todo = sorted((p for p in papers.values() if p.get("score", 0) >= min_score
                   and not (details.get(p["id"]) or {}).get("ov") and (details.get(p["id"]) or {}).get("abs")),
                  key=lambda p: -p["score"])[:limit]
    got = overviews(llm, cfg, [{"id": p["id"], "title": p["title"], "abstract": details[p["id"]]["abs"]} for p in todo],
                    system_prompt)
    for pid, ov in got.items():
        details[pid]["ov"] = ov
    for pid, p in papers.items():
        if (details.get(pid) or {}).get("ov"):
            p["has_ov"] = True
    return len(got)


def when(p: dict) -> str:
    """Data più recente in cui il paper è "uscito" per il lettore: pubblicazione o comparsa su HF Papers."""
    return max((p.get("published") or ""), (p.get("hf_date") or "")) or p.get("first_seen") or ""


def score(p: dict, now: datetime) -> float:
    """Punteggio per l'ordinamento: rilevanza per il lettore e valore scientifico, voti della community, freschezza."""
    age = max(0.0, (now - (parse_date(when(p)) or now)).total_seconds() / 86400)
    imp = p.get("imp", 3)
    base = 0.55 * imp + 0.45 * p["sci"] if p.get("sci") is not None else imp
    return round(base + math.log10(1 + p.get("upvotes", 0)) * 2.2 - min(age, 30) * 0.05, 2)


def arxiv_abstracts(ids: list[str], dates: dict | None = None) -> dict[str, str]:
    """Abstract da arXiv per una lista di id (a blocchi di 50, rispettando i 3 s tra le richieste).
    Se passi `dates`, ci mette anche la data di pubblicazione originale di ogni paper."""
    out = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        try:
            root = ET.fromstring(net.fetch(f"https://export.arxiv.org/api/query?id_list={','.join(chunk)}&max_results=50",
                                           timeout=60, retries=2))
            for e in root.findall(f"{ATOM}entry"):
                pid = _arxiv_id(e.findtext(f"{ATOM}id") or "")
                if pid:
                    out[pid] = one_line(e.findtext(f"{ATOM}summary") or "", 1600)
                    if dates is not None and e.findtext(f"{ATOM}published"):
                        dates[pid] = iso(parse_date(e.findtext(f"{ATOM}published")))
        except Exception as ex:
            _log(f"  arXiv: blocco {i // 50 + 1} non scaricato ({str(ex)[:80]})")
        time.sleep(3.1)
    return out


def backfill_overviews(root: Path, cfg: dict, llm, system_prompt, limit: int = 250) -> int:
    """Scarica gli abstract mancanti e genera le overview per i paper più rilevanti già in archivio."""
    path = root / "docs" / "data" / "papers.json"
    store = json.loads(path.read_text())
    papers = store["papers"]
    details = load_details(root)
    missing = [pid for pid in papers if not (details.get(pid) or {}).get("abs") and _arxiv_id(pid) == pid]
    for pid, ab in arxiv_abstracts(missing).items():
        details.setdefault(pid, {})["abs"] = ab
    n = fill_overviews(root, cfg, llm, system_prompt, papers, details, limit)
    path.write_text(json.dumps(store, ensure_ascii=False))
    save_details(root, details, papers)
    _log(f"Overview generate: {n} ({len(missing)} abstract scaricati)")
    return n


def reclassify(root: Path, cfg: dict, llm, system_prompt) -> int:
    """Riassegna argomenti/importanza/riassunti a tutto l'archivio (es. dopo aver cambiato gli argomenti)."""
    path = root / "docs" / "data" / "papers.json"
    store = json.loads(path.read_text())
    papers = list(store["papers"].values())
    dates: dict = {}
    abstracts = arxiv_abstracts([p["id"] for p in papers if _arxiv_id(p["id"]) == p["id"]], dates)
    for p in papers:
        # archivio vecchio: per i paper di HF "published" era la data di comparsa su HF Papers
        if p.get("source") == "HF Papers" and not p.get("hf_date") and p.get("published"):
            p["hf_date"] = p["published"]
        if dates.get(p["id"]):
            p["published"] = dates[p["id"]]
        p["abstract"] = abstracts.get(p["id"]) or p.get("summary_it") or ""
        p.pop("llm", None)
    n = classify(llm, cfg, papers, system_prompt)
    now = datetime.now(timezone.utc)
    details = load_details(root)
    for p in papers:
        if abstracts.get(p["id"]):
            details.setdefault(p["id"], {})["abs"] = abstracts[p["id"]]
        p.pop("abstract", None)
        p["score"] = score(p, now)
    store["topics"] = [{"id": t["id"], "label": t["label"], "desc": t["desc"]} for t in cfg["papers"]["topics"]]
    store["default_topics"] = cfg["papers"].get("default_topics", [])
    path.write_text(json.dumps(store, ensure_ascii=False))
    save_details(root, details, store["papers"])
    _log(f"Riclassificati {n}/{len(papers)} paper ({len(abstracts)} abstract riscaricati da arXiv)")
    return n


# ───────────────────────── archivio ─────────────────────────

def update(root: Path, cfg: dict, llm, system_prompt, status: dict, since: datetime, today: str) -> dict:
    path = root / "docs" / "data" / "papers.json"
    try:
        store = json.loads(path.read_text())
    except Exception:
        store = {"papers": {}}
    papers: dict = store.get("papers", {})
    now = datetime.now(timezone.utc)
    found = collect(cfg, since, status)
    fresh = []
    for p in found:
        cur = papers.get(p["id"])
        if cur:  # già noto: aggiorna i segnali che crescono nel tempo
            cur["upvotes"] = max(cur.get("upvotes", 0), p["upvotes"])
            cur["links"] = {**p["links"], **cur.get("links", {})}
            if p["source"] == "HF Papers" and cur.get("source") != "HF Papers":
                cur.update({"url": p["url"], "source": "HF Papers"})
            if p.get("hf_date") and not cur.get("hf_date"):
                cur["hf_date"] = p["hf_date"]
        else:
            fresh.append(p)
    # prima i più promettenti (voti su HF, poi più recenti) se superano il budget
    # budget distribuito: prima i paper votati su HF, poi a turno tra gli argomenti (così le nicchie,
    # es. interpretabilità o allineamento, non vengono schiacciate dagli argomenti più affollati)
    fresh.sort(key=lambda p: p.get("published") or "", reverse=True)
    voted = sorted((p for p in fresh if p["upvotes"] >= 3), key=lambda p: -p["upvotes"])
    queues: dict[str, list] = {}
    for p in fresh:
        if p["upvotes"] < 3:
            queues.setdefault((p.get("hints") or ["altro"])[0], []).append(p)
    rr = []
    while any(queues.values()):
        for q in queues.values():
            if q:
                rr.append(q.pop(0))
    todo = (voted + rr)[: int(cfg["papers"].get("max_per_run", 400))]
    n = classify(llm, cfg, todo, system_prompt)
    details = load_details(root)
    for p in todo:
        if (p.get("abstract") or "").strip():
            details.setdefault(p["id"], {})["abs"] = p["abstract"]
    for p in todo:
        if not p.get("llm"):  # senza LLM: argomento dall'indizio della ricerca arXiv, importanza neutra
            p["topics"] = list(dict.fromkeys(p.get("hints", [])))[:3]
            p["imp"] = 4
        p.pop("hints", None)
        p["first_seen"] = today
        if not p.get("summary_it"):
            p["summary_it"] = one_line(p["abstract"], 320)
        p.pop("abstract", None)  # l'abstract completo appesantirebbe troppo il file del sito
        papers[p["id"]] = p
    cutoff = (now - timedelta(days=int(cfg["papers"].get("days_kept", 60)))).date().isoformat()
    week = (now - timedelta(days=7)).date().isoformat()
    for p in papers.values():
        p["score"] = score(p, now)
    # ultima settimana: tutto; prima: solo i paper con punteggio decente (tiene leggero il file del sito)
    papers = {k: v for k, v in papers.items()
              if (d := when(v)[:10]) >= cutoff and (d >= week or v["score"] >= 6)}
    n_ov = fill_overviews(root, cfg, llm, system_prompt, papers, details, int(cfg["papers"].get("overview_per_run", 60)))
    save_details(root, details, papers)
    topics = [{"id": t["id"], "label": t["label"], "desc": t["desc"]} for t in cfg["papers"]["topics"]]
    out = {"updated": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "topics": topics,
           "default_topics": cfg["papers"].get("default_topics", []), "papers": papers}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False))
    _log(f"Paper: {len(found)} raccolti, {len(fresh)} nuovi, {n} classificati, {n_ov} overview, {len(papers)} in archivio")
    return out
