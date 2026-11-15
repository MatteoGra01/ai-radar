"""Registro dei modelli: scoperta (Hugging Face, OpenRouter), risoluzione dei nomi citati nelle notizie,
arricchimento (parametri, architettura, VRAM, quantizzazioni, prezzi, benchmark)."""
from __future__ import annotations

import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

from . import net
from .hardware import active_from_arch, arch_from_config, estimate, params_from_arch, params_from_name
from .textutil import norm_name, strip_html

HF = "https://huggingface.co"
VARIANT_RE = re.compile(
    r"(uncensor|abliterat|heretic|merge|gguf|awq|gptq|exl[23]|nvfp4|mxfp4|fp8|fp4|int4|int8|w4a16|w8a8|mlx|"
    r"lora|bnb|4bit|8bit|imatrix|tiny-random|placeholder|-test\b|dummy)", re.I)
WEIGHT_EXT = (".safetensors", ".gguf", ".bin", ".pt", ".pth", ".ckpt", ".onnx", ".npz", ".msgpack", ".h5", ".nemo")
MAJOR_API_ORGS = {"anthropic", "openai", "google", "x-ai", "qwen", "deepseek", "mistralai", "meta", "meta-llama",
                  "moonshotai", "z-ai", "minimax", "amazon", "microsoft", "nvidia", "cohere", "xiaomi"}
NON_PRIMARY_URL = re.compile(r"(news\.ycombinator\.com|reddit\.com|x\.com|twitter\.com|youtube\.com|bsky\.app)")


def hf_headers() -> dict:
    tok = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


# ───────────────────────── Registro ─────────────────────────

class Registry:
    def __init__(self, path: Path):
        self.path = path
        self.models: dict[str, dict] = {}
        if path.exists():
            self.models = json.loads(path.read_text()).get("models", {})
        self.touched: set[str] = set()  # nuovi o aggiornati in questa esecuzione

    def add(self, e: dict, today: str, touch: bool = True) -> dict:
        """touch=False: aggiunge al registro senza segnarlo come novità del giorno (es. modelli della classifica)."""
        cur = self.models.get(e["id"])
        if cur:
            for k, v in e.items():
                if v in (None, "", [], {}) or k in ("first_seen", "deep", "mentions"):
                    continue
                if k in ("links", "signals") and isinstance(cur.get(k), dict) and isinstance(v, dict):
                    cur[k].update(v)
                else:
                    cur[k] = v
            return cur
        e.setdefault("first_seen", today)
        e.setdefault("mentions", [])
        self.models[e["id"]] = e
        if touch:
            self.touched.add(e["id"])
        return e

    def find(self, name: str) -> dict | None:
        n = norm_name(name)
        if len(n) < 3:
            return None
        exact = [e for e in self.models.values()
                 if n in (norm_name(e["name"]), norm_name(e["id"]), norm_name(e.get("or_id") or ""))]
        if exact:
            return max(exact, key=lambda e: e.get("released") or "")
        if len(n) >= 5:
            loose = [e for e in self.models.values() if norm_name(e["name"]).startswith(n) or
                     (len(norm_name(e["name"])) >= 5 and n.startswith(norm_name(e["name"])))]
            if loose:
                return max(loose, key=lambda e: e.get("released") or "")
        return None

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        slim = {}
        for k, e in self.models.items():
            e = dict(e)
            if isinstance(e.get("vram"), dict):
                e["vram"] = {kk: vv for kk, vv in e["vram"].items() if kk != "table"}
            slim[k] = e
        data = {"updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "models": slim}
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=1))


# ───────────────────────── Hugging Face ─────────────────────────

def has_weights(siblings) -> bool | None:
    if not siblings:
        return None
    names = [s.get("rfilename", "").lower() for s in siblings]
    return any(n.endswith(WEIGHT_EXT) for n in names)


def is_variant(m: dict) -> bool:
    """True per finetune/quantizzazioni/merge di terzi o varianti quantizzate: non sono 'nuovi modelli'."""
    author = m["id"].split("/")[0].lower()
    for t in m.get("tags", []):
        if t.startswith("base_model:"):
            base = t.split(":")[-1]
            if "/" in base and base.split("/")[0].lower() != author:
                return True
    return bool(VARIANT_RE.search(m["id"].split("/")[-1]))


def hf_config(repo: str) -> dict | None:
    try:
        return net.get_json(f"{HF}/{repo}/resolve/main/config.json", headers=hf_headers(), timeout=20, retries=1)
    except Exception:
        return None


QUANT_KEEP = {"GGUF": 3, "NVFP4": 2, "FP8": 2, "MXFP4": 1, "AWQ": 1, "GPTQ": 1, "MLX": 1, "altro": 1}


def hf_quants(repo: str) -> list[dict] | None:
    """Versioni quantizzate pronte su HF, le migliori per ogni formato (prima quelle ufficiali). None = errore."""
    try:
        data = net.get_json(f"{HF}/api/models?filter=base_model:quantized:{quote(repo)}&sort=downloads&limit=60",
                            headers=hf_headers(), timeout=20, retries=1)
    except Exception:
        return None
    base_org = repo.split("/")[0].lower()
    out = []
    for m in data:
        tags = [t.lower() for t in m.get("tags", [])]
        low = m["id"].lower()
        nvfp4 = "nvfp4" in low or "nvfp4" in tags
        fmt = ("GGUF" if "gguf" in tags else "MLX" if "mlx" in tags else "NVFP4" if nvfp4 else
               "MXFP4" if "mxfp4" in low else "FP8" if ("fp8" in low or "fp8" in tags) else
               "AWQ" if "awq" in tags or "awq" in low else "GPTQ" if "gptq" in tags or "gptq" in low else "altro")
        out.append({"id": m["id"], "format": fmt, "nvfp4": nvfp4, "official": m["id"].split("/")[0].lower() == base_org,
                    "downloads": m.get("downloads", 0)})
    out.sort(key=lambda q: (not q["official"], -q["downloads"]))
    kept, count = [], {}
    for q in out:
        if count.get(q["format"], 0) < QUANT_KEEP.get(q["format"], 1):
            kept.append(q)
            count[q["format"]] = count.get(q["format"], 0) + 1
    order = list(QUANT_KEEP)
    kept.sort(key=lambda q: (order.index(q["format"]) if q["format"] in order else 99, not q["official"], -q["downloads"]))
    return kept


def native_ready(e: dict, fmt: str) -> dict | None:
    """Un checkpoint già pronto nel formato nativo (es. NVFP4, anche come GGUF NVFP4)?"""
    qs = e.get("quants") or []
    exact = [q for q in qs if q["format"] == fmt]  # prima un checkpoint nativo vero e proprio (vLLM/SGLang/TRT-LLM)
    if exact:
        return exact[0]
    return next((q for q in qs if fmt == "NVFP4" and q.get("nvfp4")), None)  # poi una GGUF NVFP4 (llama.cpp)


def entry_from_hf(api: dict, hw: dict, with_quants: bool = True) -> dict:
    repo = api["id"]
    org, name = repo.split("/", 1)
    cfgjson = hf_config(repo) or api.get("config")
    arch = arch_from_config(cfgjson)
    st_total = (api.get("safetensors") or {}).get("total")
    params_b, source = (st_total / 1e9, "safetensors") if st_total else (None, None)
    n_total, n_active = params_from_name(repo)
    if not params_b and n_total:
        params_b, source = n_total, "nome"
    if not params_b and arch:
        params_b = params_from_arch(arch)[0]
        source = "stima da config" if params_b else None
    active_b = n_active or (active_from_arch(arch, params_b) if params_b else None)
    weights = has_weights(api.get("siblings"))
    card = api.get("cardData") or {}
    lic = card.get("license") or next((t.split(":", 1)[1] for t in api.get("tags", []) if t.startswith("license:")), None)
    if isinstance(lic, list):
        lic = ", ".join(map(str, lic))
    e = {
        "id": repo,
        "name": name,
        "org": org,
        "kind": "placeholder" if weights is False else "open",
        "released": (api.get("createdAt") or "")[:10] or None,
        "hf_id": repo,
        "pipeline": api.get("pipeline_tag"),
        "license": lic,
        "gated": bool(api.get("gated")),
        "params_b": round(params_b, 2) if params_b else None,
        "params_source": source,
        "active_b": round(active_b, 2) if active_b else None,
        "context": int(arch["max_ctx"]) if arch and arch.get("max_ctx") else None,
        "arch": arch,
        "weights_available": weights,
        "signals": {"likes": api.get("likes", 0), "downloads": api.get("downloads", 0),
                    "trending": api.get("trendingScore")},
        "links": {"hf": f"{HF}/{repo}"},
    }
    e["vram"] = estimate(e["params_b"], e["active_b"], arch, hw, e["pipeline"]) if weights is not False else None
    if weights is False:
        e["note"] = "Repository senza file di pesi (segnaposto o rilascio non ancora avvenuto)."
    if with_quants and weights:
        q = hf_quants(repo)
        if q is not None:
            e["quants"] = q
            e["quants_checked"] = datetime.now(timezone.utc).date().isoformat()
    return e


def hf_model(repo: str) -> dict | None:
    try:
        return net.get_json(f"{HF}/api/models/{repo}", headers=hf_headers(), timeout=20, retries=1)
    except Exception:
        return None


def discover_hf(cfg: dict, status: dict) -> list[dict]:
    src = cfg["sources"]
    cutoff = (datetime.now(timezone.utc) - timedelta(days=cfg["run"]["new_model_days"])).isoformat()
    cands: dict[str, dict] = {}
    try:
        trending = net.get_json(f"{HF}/api/models?sort=trendingScore&limit={src.get('hf_trending_limit', 100)}&full=true",
                                headers=hf_headers(), timeout=30)
        for rank, m in enumerate(trending, 1):
            m["_trend_rank"] = rank
            if m.get("likes", 0) >= src.get("hf_min_likes", 20):
                cands[m["id"]] = m
        status["hf_trending"] = {"ok": True, "n": len(trending)}
    except Exception as e:
        status["hf_trending"] = {"ok": False, "error": str(e)[:200]}

    def org_models(org):
        return net.get_json(f"{HF}/api/models?author={quote(org)}&sort=createdAt&limit=8&full=true",
                            headers=hf_headers(), timeout=20, retries=1)

    orgs = src.get("hf_orgs", [])
    ok = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for org, res in zip(orgs, ex.map(lambda o: _safe(org_models, o), orgs)):
            if res is None:
                continue
            ok += 1
            fresh = 0
            for m in res:
                recent = (m.get("createdAt") or "") >= (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
                if (m.get("likes", 0) >= 5 or recent) and fresh < 4:
                    m["_watched"] = True
                    cands.setdefault(m["id"], m)
                    cands[m["id"]]["_watched"] = True
                    fresh += 1
    status["hf_orgs"] = {"ok": ok > 0, "n": ok}
    out = [m for m in cands.values() if (m.get("createdAt") or "") >= cutoff and not is_variant(m)]
    return out


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception:
        return None


# ───────────────────────── OpenRouter ─────────────────────────

def openrouter_index(status: dict) -> list[dict]:
    try:
        data = net.get_json("https://openrouter.ai/api/v1/models", timeout=30)["data"]
        status["openrouter"] = {"ok": True, "n": len(data)}
    except Exception as e:
        status["openrouter"] = {"ok": False, "error": str(e)[:200]}
        return []
    by_base: dict[str, dict] = {}
    free_bases = set()
    for m in data:
        base = m["id"].split(":")[0]
        if base.startswith("openrouter/"):
            continue
        if m["id"].endswith(":free"):
            free_bases.add(base)
        if base not in by_base or ":" not in m["id"]:
            by_base[base] = m
    for base, m in by_base.items():
        m["_base"] = base
        m["_has_free"] = base in free_bases
    return list(by_base.values())


def _price(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v < 0 else round(v * 1e6, 4)


def entry_from_or(m: dict) -> dict:
    full_name = m.get("name") or m["_base"]
    org_label, _, model_name = full_name.partition(": ")
    if not model_name:
        org_label, model_name = m["_base"].split("/")[0], full_name
    p = m.get("pricing") or {}
    created = datetime.fromtimestamp(m.get("created", 0), tz=timezone.utc).date().isoformat() if m.get("created") else None
    hf_id = m.get("hugging_face_id") or None
    return {
        "id": hf_id or f"or:{m['_base']}",
        "name": model_name.strip(),
        "org": org_label.strip(),
        "kind": "open" if hf_id else "closed",
        "released": created,
        "or_id": m["_base"],
        "hf_id": hf_id,
        "context": m.get("context_length"),
        "modality": (m.get("architecture") or {}).get("modality"),
        "pricing": {"in": _price(p.get("prompt")), "out": _price(p.get("completion"))},
        "free_on_openrouter": m.get("_has_free", False),
        "knowledge_cutoff": m.get("knowledge_cutoff"),
        "description": strip_html(m.get("description") or "", 700),
        "links": {"openrouter": f"https://openrouter.ai/{m['_base']}"},
    }


def discover_openrouter(index: list[dict], cfg: dict) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=cfg["run"]["new_model_days"])).timestamp()
    return [m for m in index if (m.get("created") or 0) >= cutoff]


def match_openrouter(name: str, index: list[dict]) -> dict | None:
    n = norm_name(name)
    if len(n) < 4:
        return None
    best = None
    for m in index:
        cand = {norm_name(m.get("name") or ""), norm_name(m["_base"])}
        if n in cand:
            return m
        if any(len(c) >= 5 and (c.startswith(n) or n.endswith(c)) for c in cand):
            if best is None or (m.get("created") or 0) > (best.get("created") or 0):
                best = m
    return best


def search_hf(name: str, cfg: dict) -> tuple[dict | None, list[str]]:
    """Cerca su HF un repo ufficiale con pesi per il nome citato. Restituisce (api_model|None, repo_terzi)."""
    n = norm_name(name)
    q = re.sub(r"\s+", "-", name.strip())
    q = re.sub(r"(?i)^(claude|gpt|gemini)-", r"\1-", q)
    try:
        res = net.get_json(f"{HF}/api/models?search={quote(q)}&sort=likes&limit=25&full=true",
                           headers=hf_headers(), timeout=20, retries=1)
    except Exception:
        return None, []
    orgs = {o.lower() for o in cfg["sources"].get("hf_orgs", [])}
    good, official_variants, others = [], [], []
    for m in res:
        if n not in norm_name(m["id"]):
            continue
        author = m["id"].split("/")[0].lower()
        official = author in orgs or (len(norm_name(author)) >= 3 and norm_name(author) in n)
        weights = has_weights(m.get("siblings"))
        if is_variant(m) or weights is False:
            # un produttore può pubblicare direttamente solo la versione quantizzata (es. Edge0/Edge0-145B-NVFP4)
            if official and weights:
                official_variants.append(m)
            elif not official:
                others.append(m["id"])
            continue
        good.append(m)
    good.sort(key=lambda m: (m["id"].split("/")[0].lower() not in orgs, -m.get("likes", 0)))
    if good and (good[0]["id"].split("/")[0].lower() in orgs or good[0].get("likes", 0) >= 50):
        return good[0], others
    if official_variants:
        return max(official_variants, key=lambda m: m.get("likes", 0)), others
    return None, others


def family_placeholders(name: str) -> list[str]:
    """Per un modello annunciato ma non trovato (es. 'Qwen4-Max Preview') cerca la famiglia ('Qwen4') su HF:
    se esistono solo repo di terzi/segnaposto, è un'informazione utile da mostrare."""
    fam = re.split(r"[-\s]", name.strip())[0]
    if len(fam) < 4 or not re.search(r"\d", fam):
        return []
    try:
        res = net.get_json(f"{HF}/api/models?search={quote(fam)}&sort=likes&limit=30&full=true",
                           headers=hf_headers(), timeout=20, retries=1)
    except Exception:
        return []
    pat = re.compile(rf"(?i){re.escape(fam)}(?![0-9.]|\s*b\b|b[-_.])")
    return [m["id"] for m in res if pat.search(m["id"].split("/")[-1])][:5]


# ───────────────────────── Benchmark indipendenti (Artificial Analysis + OpenRouter) ─────────────────────────

def fetch_benchmarks(status: dict) -> tuple[dict, dict]:
    """Endpoint Benchmarks di OpenRouter: indici di Artificial Analysis (intelligence, coding, agentic) e valutazioni
    proprie di OpenRouter (GPQA, tau-bench…). Restituisce ({permaslug: dati}, meta)."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return {}, {}
    out: dict[str, dict] = {}
    meta = {}
    for source in ("artificial-analysis", "openrouter"):
        try:
            data = net.get_json(f"https://openrouter.ai/api/v1/benchmarks?source={source}",
                                headers={"Authorization": f"Bearer {key}"}, timeout=40, retries=1)
        except Exception as e:
            status[f"benchmarks:{source}"] = {"ok": False, "error": str(e)[:200]}
            continue
        rows = data.get("data") or []
        status[f"benchmarks:{source}"] = {"ok": True, "n": len(rows)}
        if source == "artificial-analysis":
            meta = data.get("meta") or {}
        for r in rows:
            slug = r.get("model_permaslug")
            if not slug:
                continue
            b = out.setdefault(slug, {"slug": slug, "name": r.get("display_name")})
            if source == "artificial-analysis":
                cur = b.get("aa")
                if cur is None or (r.get("intelligence_index") or -1) > (cur.get("intelligence") or -1):
                    m = re.search(r"\(([^)]*)\)\s*$", r.get("display_name") or "")
                    b["aa"] = {"intelligence": r.get("intelligence_index"), "coding": r.get("coding_index"),
                               "agentic": r.get("agentic_index"), "variant": m.group(1) if m else None}
                    b["name"] = re.sub(r"\s*\([^)]*\)\s*$", "", r.get("display_name") or "") or b["name"]
                elif cur:  # varianti con lo stesso slug: completa gli indici mancanti
                    for k, src in (("coding", "coding_index"), ("agentic", "agentic_index")):
                        if cur.get(k) is None and r.get(src) is not None:
                            cur[k] = r[src]
            else:
                bt = r.get("benchmark_type") or ""
                if bt.startswith("search") or r.get("accuracy") is None:
                    continue
                b.setdefault("or_evals", {})[bt] = round(float(r["accuracy"]) * 100, 1)
    return out, meta


def sync_benchmarks(reg: Registry, or_index: list[dict], bench: dict, cfg: dict, today: str) -> int:
    """Collega i punteggi ai modelli del registro (via slug OpenRouter ↔ id Hugging Face). I modelli in classifica
    non ancora nel registro vengono aggiunti (senza segnarli come novità), così si possono confrontare."""
    if not bench:
        return 0
    hw = cfg["hardware"]
    by_slug = {}
    for m in or_index:
        for s in (m.get("canonical_slug"), m["_base"]):
            if s:
                by_slug[s] = m
    org_labels = {}  # "qwen" -> "Qwen", "z-ai" -> "Z.ai" (dai nomi visualizzati di OpenRouter)
    for m in or_index:
        label = (m.get("name") or "").partition(": ")[0]
        if label and ": " in (m.get("name") or ""):
            org_labels.setdefault(m["_base"].split("/")[0], label)
    for e in reg.models.values():  # i punteggi sono ricalcolati ogni giorno
        e.pop("aa", None)
        e.pop("or_evals", None)
    to_enrich, linked = [], 0
    for slug, b in bench.items():
        if not b.get("aa") and not b.get("or_evals"):
            continue
        m = by_slug.get(slug)
        if m:
            ore = entry_from_or(m)
            e = reg.models.get(ore["id"])
            if e is None:
                e = reg.add(ore, today, touch=False)
                if ore.get("hf_id"):
                    to_enrich.append(e)
        else:
            org, _, rest = slug.partition("/")
            dm = re.search(r"-(\d{4})(\d{2})(\d{2})$", rest)
            eid = f"aa:{slug}"
            nb = norm_name(b["name"] or "")
            # stesso modello già nel registro con un altro id (es. da Hugging Face)? Unisci invece di duplicare
            same = next((x for x in reg.models.values() if not x["id"].startswith("aa:") and nb and
                         nb in (norm_name(x["name"]), norm_name(x.get("hf_id") or ""))), None)
            if same:
                reg.models.pop(eid, None)
                e = same
            else:
                e = reg.models.get(eid) or reg.add({
                    "id": eid, "name": b["name"] or rest, "org": org_labels.get(org, org), "kind": "closed",
                    "released": f"{dm.group(1)}-{dm.group(2)}-{dm.group(3)}" if dm else None,
                    "links": {}}, today, touch=False)
                e["org"] = org_labels.get(org, e.get("org") or org)
        if b.get("aa"):
            if not e.get("aa") or (b["aa"].get("intelligence") or -1) > (e["aa"].get("intelligence") or -1):
                e["aa"] = {**b["aa"], "slug": slug}
        if b.get("or_evals"):
            e.setdefault("or_evals", {}).update(b["or_evals"])
        linked += 1

    def enrich(e):
        api = hf_model(e["hf_id"])
        if not api:
            return
        try:
            hfe = entry_from_hf(api, hw, with_quants=False)
        except Exception:
            return
        for k in ("params_b", "params_source", "active_b", "arch", "vram", "license", "pipeline", "weights_available"):
            if hfe.get(k) is not None:
                e[k] = hfe[k]
        e.setdefault("links", {})["hf"] = hfe["links"]["hf"]
        e.setdefault("signals", {}).update(hfe.get("signals", {}))
    with ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(enrich, to_enrich))
    return linked


# ───────────────────────── versioni più recenti ─────────────────────────

_SIZE = re.compile(r"(?i)(?<![a-z0-9])a?\d+(?:\.\d+)?[bmt](?![a-z])")
_TAGS = re.compile(r"(?i)\b(it|instruct|chat|base|hf|preview|exp|latest|\d{4,8})\b")


def parse_version(name: str) -> dict | None:
    """'granite-4.1-8b' -> famiglia 'granite', versione (4,1), taglia '8b'. None se il nome non ha una versione."""
    base = (name or "").split("/")[-1].split(":")[-1].strip()
    m = re.match(r"^(.*?[A-Za-z][A-Za-z._\- ]*?)[\s_\-]?[vV]?(\d+(?:\.\d+)*)(.*)$", base)
    if not m:
        return None
    fam, ver, suf = m.group(1), m.group(2), m.group(3)
    if re.match(r"^[bBmMtT](?![a-zA-Z])", suf):  # "gpt-oss-120b": è una taglia, non una versione
        return None
    sizes = [s.lower() for s in _SIZE.findall(suf)]
    rest = _TAGS.sub(" ", _SIZE.sub(" ", suf))
    return {"fam": re.sub(r"[^a-z0-9]", "", fam.lower()), "fam_raw": fam.strip(" -_."),
            "ver": tuple(int(x) for x in ver.split(".")),
            "size": next((s for s in sizes if not s.startswith("a")), None),  # "A3B" = parametri attivi, non la taglia
            "suf": re.sub(r"[^a-z0-9]", "", rest.lower())}


def _is_newer(cur: dict, cand: dict) -> bool:
    if cand["fam"] != cur["fam"] or cand["suf"] != cur["suf"] or cand["ver"] <= cur["ver"]:
        return False
    if cur["size"] and cand["size"] != cur["size"]:
        return False
    return True


def find_newer(e: dict, or_index: list[dict], cache: dict) -> dict | None:
    """Cerca una versione più recente della stessa famiglia dello stesso produttore (stessa taglia)."""
    cur = parse_version(e.get("hf_id") or e["name"])
    if not cur or not cur["fam"]:
        return None
    best = None
    if e.get("hf_id"):
        org = e["hf_id"].split("/")[0]
        key = (org.lower(), cur["fam"])
        if key not in cache:
            try:
                cache[key] = net.get_json(f"{HF}/api/models?author={quote(org)}&search={quote(cur['fam_raw'])}"
                                          f"&sort=createdAt&limit=100", headers=hf_headers(), timeout=20, retries=1)
            except Exception:
                cache[key] = []
        for m in cache[key]:
            if is_variant(m):
                continue
            pv = parse_version(m["id"])
            if pv and _is_newer(cur, pv) and (best is None or pv["ver"] > best[0]):
                best = (pv["ver"], {"id": m["id"], "name": m["id"].split("/")[-1],
                                    "released": (m.get("createdAt") or "")[:10], "url": f"{HF}/{m['id']}"})
    if e.get("or_id"):
        org = e["or_id"].split("/")[0]
        for m in or_index:
            if m["_base"].split("/")[0] != org:
                continue
            nm = (m.get("name") or "").partition(": ")[2] or m["_base"]
            pv = parse_version(nm)
            if pv and _is_newer(cur, pv) and (best is None or pv["ver"] > best[0]):
                best = (pv["ver"], {"id": m.get("hugging_face_id") or f"or:{m['_base']}", "name": nm,
                                    "released": datetime.fromtimestamp(m.get("created", 0), tz=timezone.utc).date().isoformat(),
                                    "url": f"https://openrouter.ai/{m['_base']}"})
    return best[1] if best else None


# ───────────────────────── priorità ─────────────────────────

def priority(e: dict, cfg: dict) -> float:
    s = {"open": 2.0, "closed": 2.0, "announced": 1.5, "placeholder": 0.0}.get(e["kind"], 0)
    s += max([m.get("rel", 0) for m in e.get("mentions", [])] or [0]) * 0.8
    sig = e.get("signals") or {}
    s += math.log10(1 + (sig.get("likes") or 0)) * 0.9
    if sig.get("trending"):
        s += 1.0
    org = (e.get("org") or "").lower()
    if org in {o.lower() for o in cfg["sources"].get("hf_orgs", [])}:
        s += 1.5
    if (e.get("or_id") or "").split("/")[0] in MAJOR_API_ORGS:
        s += 3.0
    v = e.get("vram") or {}
    if v.get("level") == "ok":
        s += 2.0
    elif v.get("level") == "warn":
        s += 1.0
    if e.get("pipeline") in ("text-generation", "image-text-to-text", "any-to-any", "feature-extraction",
                             "sentence-similarity"):
        s += 1.0
    return round(s, 2)


def _org_tokens(e: dict) -> set[str]:
    toks = set()
    for s in (e.get("org"), (e.get("or_id") or "").split("/")[0], (e.get("hf_id") or "").split("/")[0]):
        for t in re.split(r"[^a-z0-9]+", (s or "").lower()):
            if len(t) >= 4:
                toks.add(t)
    if toks & {"google", "gemini", "deepmind"}:
        toks |= {"blog.google", "deepmind"}
    if toks & {"meta", "llama"}:
        toks |= {"ai.meta", "about.fb"}
    if toks & {"qwen", "alibaba"}:
        toks |= {"qwen", "alibabacloud"}
    return toks


def primary_urls(e: dict, items_by_id: dict) -> list[str]:
    """URL da cui estrarre benchmark e feature: prima i siti ufficiali dell'organizzazione, poi blog e articoli,
    per ultimi aggregatori e social."""
    urls = []
    for m in sorted(e.get("mentions", []), key=lambda m: -m.get("rel", 0)):
        it = items_by_id.get(m["item"])
        url = (it or {}).get("url") or m.get("url")
        if url:
            urls.append(url)
    toks = _org_tokens(e)

    def rank(u: str) -> int:
        host = re.sub(r"^https?://(www\.)?", "", u).split("/")[0].lower()
        if NON_PRIMARY_URL.search(u):
            return 2
        return 0 if any(t in host for t in toks) else 1
    urls.sort(key=rank)
    if e.get("links", {}).get("announcement"):
        urls.insert(0, e["links"]["announcement"])
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def fetch_source_text(e: dict, items_by_id: dict, limit: int = 14000) -> tuple[str, str | None]:
    """Testo su cui fare l'analisi approfondita: annuncio ufficiale, poi model card, poi descrizione."""
    for url in primary_urls(e, items_by_id)[:4]:
        if "news.ycombinator.com" in url:
            continue
        try:
            text = strip_html(net.get_text(url, timeout=25, retries=1))
        except Exception:
            continue
        # siti ufficiali e articoli: serve un testo ricco; post social (es. annuncio su X): basta un paragrafo
        if len(text) > (400 if NON_PRIMARY_URL.search(url) else 1500):
            return smart_truncate(text, limit), url
    if e.get("hf_id"):
        try:
            text = net.get_text(f"{HF}/{e['hf_id']}/raw/main/README.md", headers=hf_headers(), timeout=20, retries=1)
            text = re.sub(r"^---.*?---", "", text, count=1, flags=re.S)
            text = strip_html(text)
            if len(text) > 300:
                return smart_truncate(text, limit), f"{HF}/{e['hf_id']}"
        except Exception:
            pass
    if e.get("description"):
        return e["description"], e.get("links", {}).get("openrouter")
    return "", None


def smart_truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 3]
    rest = text[limit // 3:]
    windows = []
    for m in re.finditer(r"(?i)benchmark|evaluation|results|performance|swe-bench|gpqa|mmlu|aime|context|parameters|pricing", rest):
        windows.append((max(0, m.start() - 300), min(len(rest), m.start() + 1800)))
    merged: list[list[int]] = []
    for a, b in windows:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    body, budget = [], limit - len(head)
    for a, b in merged:
        chunk = rest[a:b]
        if len(chunk) > budget:
            chunk = chunk[:budget]
        body.append(chunk)
        budget -= len(chunk)
        if budget <= 0:
            break
    if not body:
        body = [rest[: limit - len(head)]]
    return head + "\n[…]\n" + "\n[…]\n".join(body)
