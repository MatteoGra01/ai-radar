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

    def add(self, e: dict, today: str) -> dict:
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
        data = {"updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "models": self.models}
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


def hf_quants(repo: str) -> list[dict]:
    try:
        data = net.get_json(f"{HF}/api/models?filter=base_model:quantized:{repo}&sort=downloads&limit=12",
                            headers=hf_headers(), timeout=20, retries=1)
    except Exception:
        return []
    out = []
    for m in data:
        tags = m.get("tags", [])
        fmt = "GGUF" if "gguf" in tags else "MLX" if "mlx" in tags else "AWQ" if "awq" in tags else \
              "GPTQ" if "gptq" in tags else "FP8" if "fp8" in m["id"].lower() else \
              "NVFP4" if "nvfp4" in m["id"].lower() else "altro"
        out.append({"id": m["id"], "format": fmt, "downloads": m.get("downloads", 0)})
    # prima i GGUF (llama.cpp / Ollama / LM Studio), poi gli altri
    out.sort(key=lambda q: (q["format"] != "GGUF", -q["downloads"]))
    return out[:6]


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
        e["quants"] = hf_quants(repo)
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
    good, others = [], []
    for m in res:
        if n not in norm_name(m["id"]):
            continue
        if is_variant(m) or has_weights(m.get("siblings")) is False:
            others.append(m["id"])
            continue
        good.append(m)
    good.sort(key=lambda m: (m["id"].split("/")[0].lower() not in orgs, -m.get("likes", 0)))
    if good and (good[0]["id"].split("/")[0].lower() in orgs or good[0].get("likes", 0) >= 50):
        return good[0], others
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


# ───────────────────────── Artificial Analysis (opzionale) ─────────────────────────

AA_KEYS = re.compile(r"(index|bench|gpqa|mmlu|hle|aime|swe|livecode|scicode|math|ifbench|lcr|tau|terminal|arena|elo)", re.I)


def aa_models(status: dict) -> list[dict]:
    key = os.environ.get("AA_API_KEY")
    if not key:
        return []
    try:
        data = net.get_json("https://artificialanalysis.ai/api/v2/language/models/free",
                            headers={"x-api-key": key}, timeout=30, retries=1)
    except Exception as e:
        status["artificial_analysis"] = {"ok": False, "error": str(e)[:200]}
        return []
    rows = data.get("data") if isinstance(data, dict) else data
    out = []
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        name = r.get("name") or r.get("model_name") or r.get("slug")
        ev = r.get("evaluations") if isinstance(r.get("evaluations"), dict) else \
            {k: v for k, v in r.items() if isinstance(v, (int, float)) and AA_KEYS.search(k)}
        ev = {k: v for k, v in (ev or {}).items() if isinstance(v, (int, float))}
        if name and ev:
            out.append({"name": name, "slug": r.get("slug"), "evals": ev})
    status["artificial_analysis"] = {"ok": True, "n": len(out)}
    return out


def attach_aa(registry: Registry, aa: list[dict]):
    if not aa:
        return
    for e in registry.models.values():
        if e["kind"] not in ("closed", "open", "announced"):
            continue
        n = norm_name(e["name"])
        if len(n) < 5:
            continue
        cands = [a for a in aa if norm_name(a["name"]) == n or norm_name(a["name"]).startswith(n)]
        if not cands:
            continue
        best = max(cands, key=lambda a: next((v for k, v in a["evals"].items() if "intelligence" in k), 0))
        e["aa"] = {"name": best["name"], "evals": best["evals"],
                   "url": f"https://artificialanalysis.ai/models/{best['slug']}" if best.get("slug") else None}


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
    for url in primary_urls(e, items_by_id)[:3]:
        if NON_PRIMARY_URL.search(url):
            continue
        try:
            text = strip_html(net.get_text(url, timeout=25, retries=1))
        except Exception:
            continue
        if len(text) > 1500:
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
