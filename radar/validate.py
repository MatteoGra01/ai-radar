"""Controllo qualità delle schede modello prima della pubblicazione.

- unisce le schede duplicate (un modello "annunciato" che poi risulta essere un repo già in registro);
- ricontrolla su Hugging Face lo stato dei repository (esiste, gated, contiene i pesi) almeno una volta a settimana;
- ricava i parametri dai metadati safetensors quando prima erano solo dedotti dal nome;
- annota per ogni scheda gli avvisi sui dati mancanti o incoerenti (mostrati nel sito come "Qualità dei dati").
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

from . import models as M
from .hardware import estimate


def _merge_into(dst: dict, src: dict):
    """Fonde una scheda duplicata (src) in quella canonica (dst) senza perdere menzioni, analisi e link."""
    have = {m.get("item") for m in dst.get("mentions", [])}
    dst["mentions"] = (dst.get("mentions", []) + [m for m in src.get("mentions", []) if m.get("item") not in have])[:8]
    if src.get("deep") and not dst.get("deep"):
        dst["deep"] = src["deep"]
    dst["links"] = {**src.get("links", {}), **dst.get("links", {})}
    for k in ("first_seen", "released"):
        vals = [x for x in (dst.get(k), src.get(k)) if x]
        if vals:
            dst[k] = min(vals)
    dst["score"] = max(dst.get("score") or 0, src.get("score") or 0)


def dedupe(reg: M.Registry) -> int:
    n = 0
    for mid, e in list(reg.models.items()):
        target = e.get("hf_id")
        if mid.startswith("ann:") and target and target != mid and target in reg.models:
            _merge_into(reg.models[target], e)
            reg.aliases[mid] = target
            del reg.models[mid]
            if mid in reg.touched:
                reg.touched.discard(mid)
                reg.touched.add(target)
            n += 1
    # gli alias puntano sempre a una scheda esistente (anche dopo fusioni a catena)
    for a, t in list(reg.aliases.items()):
        while t in reg.aliases and reg.aliases[t] != t:
            t = reg.aliases[t]
        if t in reg.models:
            reg.aliases[a] = t
        else:
            del reg.aliases[a]
    return n


def recheck_hf(reg: M.Registry, cfg: dict, today: str, limit: int = 120) -> int:
    """Stato del repository su HF per le schede con pesi aperti non verificate da una settimana."""
    week = (date.fromisoformat(today) - timedelta(days=7)).isoformat()
    todo = sorted((e for e in reg.models.values() if e.get("hf_id") and e.get("kind") in ("open", "placeholder")
                   and (e.get("hf_checked") or "") < week), key=lambda e: e.get("hf_checked") or "")[:limit]

    def one(e):
        api = M.hf_model(e["hf_id"])
        return e, api

    n = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for e, api in ex.map(one, todo):
            if api is None:
                # repo rimosso, rinominato o reso privato (o errore di rete: si riprova la volta dopo)
                e["repo_missing"] = (e.get("repo_missing") or 0) + 1
                continue
            e.pop("repo_missing", None)
            e["gated"] = bool(api.get("gated"))
            w = M.has_weights(api.get("siblings"))
            if w is not None:
                e["weights_available"] = w
                if w is False and e["kind"] == "open":
                    e["kind"] = "placeholder"
                elif w and e["kind"] == "placeholder":
                    e["kind"] = "open"
            st = (api.get("safetensors") or {}).get("total")
            if st and e.get("params_source") != "safetensors":
                e["params_b"], e["params_source"] = round(st / 1e9, 2), "safetensors"
            e.setdefault("signals", {}).update({"likes": api.get("likes", 0), "downloads": api.get("downloads", 0)})
            e["hf_checked"] = today
            n += 1
    return n


ACTIVE_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*B\s*(?:di\s+parametri\s+)?(?:attivi|active)", re.I)


def declared_active(e: dict) -> bool:
    """Senza config.json leggibile: usa i parametri attivi dichiarati dal produttore (es. "145B totali / ~10B attivi")."""
    if e.get("active_b") or not e.get("params_b"):
        return False
    m = ACTIVE_RE.search((e.get("deep") or {}).get("params") or "")
    if not m:
        return False
    a = float(m.group(1).replace(",", "."))
    if not 0 < a < e["params_b"] * 0.7:
        return False
    e["active_b"], e["active_source"] = a, "dichiarato"
    return True


def checks(e: dict) -> list[str]:
    """Avvisi sulla scheda: cosa manca o non torna (non sono errori del modello, ma limiti dei dati)."""
    out = []
    local = e.get("kind") in ("open", "placeholder") or e.get("hf_id")
    if e.get("repo_missing", 0) >= 2:
        out.append("Il repository Hugging Face non risponde più (rimosso, rinominato o reso privato).")
    if local and e.get("kind") != "placeholder":
        if not e.get("params_b"):
            out.append("Numero di parametri non determinato: niente stima della VRAM.")
        elif e.get("params_source") == "nome":
            out.append("Parametri dedotti dal nome del repository (non dai metadati dei pesi).")
        if e.get("params_b") and not e.get("arch"):
            out.append("config.json non leggibile" + (" (repo gated: servono le condizioni accettate e un HF_TOKEN)" if e.get("gated") else "")
                       + ": KV cache, contesto massimo e parametri attivi non stimabili.")
        if e.get("active_source") == "dichiarato":
            out.append("Parametri attivi presi dalla dichiarazione del produttore, non dall'architettura.")
        if e.get("arch") and not e.get("active_b") and (e.get("arch") or {}).get("experts"):
            out.append("MoE senza numero di parametri attivi: velocità stimata come modello denso.")
    if e.get("kind") == "announced" and (e.get("links") or {}).get("hf"):
        out.append("Annunciato ma con un link Hugging Face: stato da ricontrollare.")
    return out


def validate(reg: M.Registry, cfg: dict, today: str, *, recheck: bool = True) -> dict:
    merged = dedupe(reg)
    rechecked = recheck_hf(reg, cfg, today) if recheck else 0
    hw = cfg["hardware"]
    for e in reg.models.values():
        declared_active(e)
        if e.get("kind") == "placeholder" and e.get("vram"):
            e["vram"] = None
        elif e.get("kind") == "open" and e.get("params_b") and not e.get("vram"):
            e["vram"] = estimate(e["params_b"], e.get("active_b"), e.get("arch"), hw, e.get("pipeline"))
        c = checks(e)
        if c:
            e["checks"] = c
        else:
            e.pop("checks", None)
    return {"merged": merged, "rechecked": rechecked, "with_warnings": sum(1 for e in reg.models.values() if e.get("checks"))}
