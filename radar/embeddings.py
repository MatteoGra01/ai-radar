"""Modelli di embedding: leaderboard MTEB ufficiale (API del backend della leaderboard), punteggio sulla lingua
scelta, versioni superate e nuovi modelli su Hugging Face non ancora valutati su MTEB."""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

from . import models as M
from . import net
from .hardware import _text_cfg

API = "https://mteb-leaderboard-backend.hf.space/v1"
PIPES = ("feature-extraction", "sentence-similarity", "text-ranking")


def _log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _pct(x):
    return round(x * 100, 2) if isinstance(x, (int, float)) else None


def fetch_benchmark(name: str) -> dict:
    return net.get_json(f"{API}/benchmarks/{quote(name)}/scores", timeout=60, retries=2)


def fetch_language(name: str, lang: str) -> dict[str, float]:
    try:
        d = net.get_json(f"{API}/benchmarks/{quote(name)}/per-language", timeout=60, retries=1)
    except Exception:
        return {}
    return {r["modelName"]: _pct(r["scoresByLanguage"][lang]) for r in d.get("rows", [])
            if lang in (r.get("scoresByLanguage") or {})}


def superseded(names: list[str]) -> dict[str, str]:
    """{modello: versione più recente della stessa famiglia e taglia dello stesso produttore}"""
    parsed = {}
    for n in names:
        pv = M.parse_version(n)
        if pv:
            parsed[n] = (n.split("/")[0].lower() if "/" in n else "", pv)
    out = {}
    for n, (org, pv) in parsed.items():
        best = None
        for m, (org2, pv2) in parsed.items():
            if m != n and org2 == org and M._is_newer(pv, pv2) and (best is None or pv2["ver"] > parsed[best][1]["ver"]):
                best = m
        if best:
            out[n] = best
    return out


def discover_new(cfg: dict, known: set[str]) -> list[dict]:
    """Modelli di embedding/reranker recenti su HF (trending o di organizzazioni seguite) non ancora su MTEB."""
    days = int(cfg.get("embeddings", {}).get("new_days", 45))
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    orgs = {o.lower() for o in cfg["sources"].get("hf_orgs", [])}
    cands = {}
    for pipe in PIPES:
        try:
            res = net.get_json(f"https://huggingface.co/api/models?pipeline_tag={pipe}&sort=trendingScore&limit=60&full=true",
                               headers=M.hf_headers(), timeout=30, retries=1)
        except Exception:
            continue
        for m in res:
            if (m.get("createdAt") or "") < cutoff or m["id"] in known or M.is_variant(m):
                continue
            if m.get("likes", 0) >= 10 or m["id"].split("/")[0].lower() in orgs:
                m["_pipe"] = pipe
                cands[m["id"]] = m
    out = []

    def detail(m):
        api = M.hf_model(m["id"])
        if not api:
            return None
        st = (api.get("safetensors") or {}).get("total")
        cfgjson = M.hf_config(m["id"]) or {}
        t = _text_cfg(cfgjson)
        return {"name": m["id"], "url": f"https://huggingface.co/{m['id']}", "kind": "reranker" if m["_pipe"] == "text-ranking" else "embedding",
                "params": round(st / 1e9, 3) if st else M.params_from_name(m["id"])[0],
                "dim": t.get("hidden_size") or t.get("d_model"), "maxTokens": t.get("max_position_embeddings"),
                "released": (m.get("createdAt") or "")[:10], "likes": m.get("likes", 0), "open": True,
                "license": (api.get("cardData") or {}).get("license")}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for r in ex.map(detail, list(cands.values())[:30]):
            if r:
                out.append(r)
    out.sort(key=lambda r: r["released"], reverse=True)
    return out


def update(root: Path, cfg: dict, status: dict, today: str) -> dict | None:
    ec = cfg.get("embeddings", {})
    benches = ec.get("benchmarks", [])
    if not benches:
        return None
    path = root / "docs" / "data" / "embeddings.json"
    top_n = int(ec.get("top_n", 150))
    lang = ec.get("language", "Italian")
    recent_cut = (datetime.now(timezone.utc) - timedelta(days=int(ec.get("new_days", 45)) * 2)).date().isoformat()
    meta: dict[str, dict] = {}
    out_b = []
    errors = []
    for b in benches:
        try:
            d = fetch_benchmark(b["name"])
        except Exception as e:
            errors.append(f"{b['name']}: {str(e)[:80]}")
            continue
        rows = []
        for r in d.get("rows", []):
            m = r.get("model") or {}
            name = m.get("name")
            if not name or r.get("meanTask") is None:
                continue
            keep = r["rank"] <= top_n or (m.get("releaseDate") or "") >= recent_cut
            if not keep:
                continue
            meta.setdefault(name, {
                "name": name, "url": m.get("url"), "params": m.get("totalParamsB"), "active": m.get("activeParamsB"),
                "dim": m.get("embeddingDim"), "maxTokens": int(m["maxTokens"]) if m.get("maxTokens") else None,
                "released": m.get("releaseDate"), "open": bool(m.get("openWeights")), "type": m.get("modelType"),
                "instruct": m.get("instructionTuned"), "st": m.get("sentenceTransformersCompatible"),
                "nLangs": len(m.get("languages") or []), "lang": lang in (m.get("languages") or []),
                "modalities": m.get("modalities") or ["text"],
            })
            rows.append({"m": name, "rank": r["rank"], "mean": _pct(r["meanTask"]), "meanType": _pct(r.get("meanTaskType")),
                         "zs": r.get("zeroShotPct"), "types": {k: _pct(v) for k, v in (r.get("scoresByTaskType") or {}).items()}})
        entry = {"name": b["name"], "label": b.get("label", b["name"]), "types": d.get("taskTypes", []), "rows": rows,
                 "total": len(d.get("rows", []))}
        if b.get("language_scores"):
            ls = fetch_language(b["name"], lang)
            for r in rows:
                if r["m"] in ls:
                    r["lang"] = ls[r["m"]]
            entry["language"] = lang
        out_b.append(entry)
    status["embeddings:mteb"] = {"ok": bool(out_b), "n": sum(len(b["rows"]) for b in out_b),
                                 **({"error": "; ".join(errors)[:200]} if errors else {})}
    if not out_b:
        return None
    sup = superseded(list(meta))
    for n, newer in sup.items():
        meta[n]["newer"] = newer
    try:
        fresh = discover_new(cfg, set(meta))
        status["embeddings:hf"] = {"ok": True, "n": len(fresh)}
    except Exception as e:
        fresh = []
        status["embeddings:hf"] = {"ok": False, "error": str(e)[:200]}
    data = {"updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "language": lang,
            "source": "MTEB leaderboard (mteb-leaderboard-backend.hf.space) · Hugging Face",
            "benchmarks": out_b, "models": meta, "new_unbenchmarked": fresh}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False))
    _log(f"Embedding: {len(meta)} modelli da {len(out_b)} benchmark MTEB, {len(sup)} superati, "
         f"{len(fresh)} nuovi non ancora su MTEB")
    return data
