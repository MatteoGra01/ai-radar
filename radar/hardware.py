"""Stima VRAM / contesto massimo / token al secondo per un modello su una GPU.

Formula (identica a quella di docs/app.js, usata dal calcolatore interattivo):
  pesi     = parametri × bit_per_peso / 8
  KV cache = Σ layer di attenzione (2 × kv_heads × head_dim × 2 byte) × contesto
             (layer sliding-window limitati alla finestra, MLA = (kv_lora_rank + rope_dim) per token,
              layer lineari/SSM = stato costante)
  overhead = 0.75 GiB + 2% dei pesi (buffer di calcolo, contesto CUDA)
I bit/peso sono quelli effettivi dei formati GGUF di llama.cpp; FP8/NVFP4 includono le scale.
"""
from __future__ import annotations

import re

GiB = 2**30

QUANTS = [
    ("BF16", 16.0),
    ("Q8_0", 8.5),
    ("Q6_K", 6.56),
    ("Q5_K_M", 5.69),
    ("Q4_K_M", 4.85),
    ("IQ4_XS", 4.25),
    ("Q3_K_M", 3.91),
    ("IQ2_M", 2.7),
]
GOOD_QUANTS = {"BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "IQ4_XS", "FP8", "NVFP4"}


def _first(d: dict, *keys):
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
            return v
    return None


def params_from_name(name: str) -> tuple[float | None, float | None]:
    """'Qwen3.6-35B-A3B' -> (35.0, 3.0); 'Qwen3.8-2.4T-A95B' -> (2400.0, 95.0)."""
    base = (name or "").split("/")[-1]
    total = active = None
    m = re.search(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)\s?([BbTtMm])(?![A-Za-z0-9])", base)
    if m:
        v = float(m.group(1))
        u = m.group(2).upper()
        total = v * 1000 if u == "T" else v / 1000 if u == "M" else v
    m = re.search(r"(?<![A-Za-z0-9])[Aa](\d+(?:\.\d+)?)[Bb](?![A-Za-z0-9])", base)
    if m:
        active = float(m.group(1))
    return total, active


def _text_cfg(cfg: dict) -> dict:
    if not isinstance(cfg, dict):
        return {}
    t = dict(cfg)
    for k in ("text_config", "llm_config", "language_config", "thinker_config", "language_model_config"):
        sub = cfg.get(k)
        if isinstance(sub, dict):
            t.update(_text_cfg(sub))
    return t


def arch_from_config(cfg: dict | None) -> dict | None:
    """Estrae dal config.json di HF i soli campi che servono a stimare la memoria."""
    if not cfg:
        return None
    try:
        return _arch(cfg)
    except Exception:  # config non standard: meglio nessuna stima che un crash
        return None


def _int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _arch(cfg: dict) -> dict | None:
    t = _text_cfg(cfg)
    L = _first(t, "num_hidden_layers", "n_layer", "num_layers", "n_layers")
    if not L:
        return None
    L = int(L)
    hidden = _first(t, "hidden_size", "d_model", "n_embd", "dim")
    heads = _first(t, "num_attention_heads", "n_head", "n_heads", "num_heads")
    kv = _first(t, "num_key_value_heads", "num_kv_heads", "n_kv_heads", "multi_query_group_num") or heads
    head_dim = _first(t, "head_dim", "kv_channels") or (int(hidden // heads) if hidden and heads else None)

    full = sliding = linear = 0
    lt = t.get("layer_types")
    if isinstance(lt, list) and lt:
        for x in lt:
            x = str(x).lower()
            if any(s in x for s in ("sliding", "local", "chunk")):
                sliding += 1
            elif any(s in x for s in ("linear", "mamba", "ssm", "recurrent", "deltanet", "conv", "rwkv")):
                linear += 1
            else:
                full += 1
    elif _int(t.get("full_attention_interval")):
        n = _int(t["full_attention_interval"])
        full, linear = L // n, L - L // n
    elif _int(t.get("sliding_window_pattern")) and t.get("sliding_window"):
        p = _int(t["sliding_window_pattern"])
        full, sliding = L // p, L - L // p
    elif isinstance(t.get("hybrid_override_pattern"), str):  # Nemotron-H: M=mamba, *=attention
        pat = t["hybrid_override_pattern"]
        full, linear = pat.count("*"), pat.count("M")
    elif _int(t.get("attn_layer_period")):  # Jamba
        p = _int(t["attn_layer_period"])
        full, linear = L // p, L - L // p
    else:
        full = L
    window = _first(t, "sliding_window", "attention_window") if sliding else None
    if sliding and not window:
        full, sliding = full + sliding, 0

    mla_dim = None
    if _first(t, "kv_lora_rank"):
        mla_dim = int(t["kv_lora_rank"]) + int(_first(t, "qk_rope_head_dim") or 64)

    linear_state = 0
    if linear:
        lv = _first(t, "linear_num_value_heads")
        lk = _first(t, "linear_key_head_dim")
        lvd = _first(t, "linear_value_head_dim")
        mh = _first(t, "mamba_num_heads", "mamba_n_heads")
        mhd = _first(t, "mamba_head_dim")
        ss = _first(t, "ssm_state_size", "state_size", "mamba_d_state")
        if lv and lk and lvd:
            linear_state = int(lv * lk * lvd * 4)
        elif mh and mhd and ss:
            linear_state = int(mh * mhd * ss * 4)
        else:
            linear_state = 4 * 2**20

    experts = _first(t, "n_routed_experts", "num_local_experts", "num_experts", "moe_num_experts", "num_routed_experts")
    k_act = _first(t, "num_experts_per_tok", "moe_top_k", "num_experts_per_token", "top_k_experts", "moe_k")
    moe_inter = _first(t, "moe_intermediate_size", "expert_intermediate_size", "moe_ffn_hidden_size")
    return {
        "layers": L,
        "hidden": hidden,
        "heads": heads,
        "kv_heads": kv,
        "head_dim": head_dim,
        "full_layers": full,
        "sliding_layers": sliding,
        "window": window,
        "linear_layers": linear,
        "linear_state_bytes": linear_state,
        "mla_dim": mla_dim,
        "experts": int(experts) if experts and experts > 1 else None,
        "experts_active": int(k_act) if experts and k_act else None,
        "moe_inter": moe_inter if experts else None,
        "dense_layers": int(_first(t, "first_k_dense_replace") or 0),
        "inter": _first(t, "intermediate_size", "ffn_hidden_size"),
        "vocab": _first(t, "vocab_size", "padded_vocab_size"),
        "tied": bool(t.get("tie_word_embeddings", False)),
        "max_ctx": _first(t, "max_position_embeddings", "max_sequence_length", "seq_length", "n_positions"),
    }


def params_from_arch(arch: dict) -> tuple[float | None, float | None]:
    """Stima grezza (in miliardi) quando HF non espone il numero di parametri."""
    try:
        h, L = arch["hidden"], arch["layers"]
        heads, kv, hd = arch["heads"], arch["kv_heads"], arch["head_dim"]
        attn = h * heads * hd * 2 + h * kv * hd * 2
        emb = arch["vocab"] * h * (1 if arch["tied"] else 2)
        if arch.get("experts"):
            routed = arch["experts"] * 3 * h * (arch["moe_inter"] or arch["inter"])
            dense_mlp = 3 * h * (arch["inter"] or 0)
            n_moe = L - arch["dense_layers"]
            total = emb + L * attn + n_moe * routed + arch["dense_layers"] * dense_mlp
            active = total - n_moe * routed * (1 - arch["experts_active"] / arch["experts"])
            return total / 1e9, active / 1e9
        total = emb + L * (attn + 3 * h * arch["inter"])
        return total / 1e9, None
    except (KeyError, TypeError, ZeroDivisionError):
        return None, None


def active_from_arch(arch: dict | None, total_b: float) -> float | None:
    if not arch or not arch.get("experts") or not arch.get("experts_active"):
        return None
    try:
        inter = arch.get("moe_inter") or arch.get("inter")
        routed = (arch["layers"] - arch["dense_layers"]) * arch["experts"] * 3 * arch["hidden"] * inter
        ratio = arch["experts_active"] / arch["experts"]
        active = total_b * 1e9 - routed * (1 - ratio)
        # architetture non standard (tabelle di lookup, layer extra): stima non affidabile → meglio nessun numero
        if 0 < active < total_b * 1e9 and active / (total_b * 1e9) <= max(0.25, 3 * ratio + 0.15):
            return round(active / 1e9, 1)
    except (KeyError, TypeError, ZeroDivisionError):
        pass
    return None


def kv_bytes(arch: dict | None, ctx: int, bytes_per_el: float = 2.0) -> float:
    if not arch:
        return 0.0
    if arch.get("mla_dim"):
        per = arch["mla_dim"] * bytes_per_el
    elif arch.get("kv_heads") and arch.get("head_dim"):
        per = 2 * arch["kv_heads"] * arch["head_dim"] * bytes_per_el
    else:
        return 0.0
    b = arch.get("full_layers", 0) * per * ctx
    if arch.get("sliding_layers"):
        b += arch["sliding_layers"] * per * min(ctx, arch.get("window") or ctx)
    b += arch.get("linear_layers", 0) * arch.get("linear_state_bytes", 0)
    return b


def _fmt_ctx(c: int) -> str:
    return f"{c // 1024}k" if c >= 1024 else str(c)


TEXT_PIPES = {None, "text-generation", "image-text-to-text", "any-to-any", "text2text-generation", "conversational"}


def _tok(t: int | None) -> str:
    return "500+" if t and t >= 500 else f"~{t}"


NATIVE = {"FP8": 8.25, "NVFP4": 4.5}  # bit effettivi: FP8 + scale; FP4 + una scala FP8 ogni 16 valori


def estimate(params_b: float | None, active_b: float | None, arch: dict | None, hw: dict,
             pipeline: str | None = None) -> dict | None:
    """Verdetto per la GPU: GGUF (llama.cpp/Ollama/LM Studio) e formati con accelerazione hardware della GPU
    (es. NVFP4/FP8 su Blackwell, per vLLM/SGLang/TensorRT-LLM)."""
    if not params_b:
        return None
    text_model = pipeline in TEXT_PIPES
    usable = hw["vram_gb"] - hw["reserved_gb"]
    margin = float(hw.get("margin_gb", 1.5))  # spazio libero da lasciare (desktop, picchi, altri processi)
    ref = int(hw["ref_context"])
    max_pos = int(arch["max_ctx"]) if arch and arch.get("max_ctx") else None
    kv_known = bool(arch and (arch.get("mla_dim") or (arch.get("kv_heads") and arch.get("head_dim"))))
    is_moe = bool(active_b and active_b < params_b * 0.7)
    act = active_b or params_b
    eff = 0.7 if not is_moe else min(0.7, 0.15 + 0.03 * act)
    gpu = hw["gpu_name"]

    def total_gb(w, ctx):
        return w + kv_bytes(arch, ctx) / GiB + 0.75 + 0.02 * w

    def make_row(q, bpw, native=False):
        w = params_b * 1e9 * bpw / 8 / GiB
        rows = []
        for c in hw["contexts"]:
            if max_pos and c > max_pos:
                rows.append(None)
                continue
            g = total_gb(w, c)
            rows.append({"ctx": c, "gb": round(g, 1), "fits": g <= usable})
        max_ctx = None
        if kv_known and total_gb(w, 512) <= usable:
            lo, hi = 512, max_pos or 1_048_576
            if total_gb(w, hi) <= usable:
                lo = hi
            while hi - lo > 256:
                mid = (lo + hi) // 2
                lo, hi = (mid, hi) if total_gb(w, mid) <= usable else (lo, mid)
            max_ctx = lo
        bytes_tok = act * 1e9 * bpw / 8 + kv_bytes(arch, 4096)
        tok_s = round(eff * hw["bandwidth_gbs"] * 1e9 / bytes_tok) if bytes_tok and text_model else None
        return {"q": q, "bpw": bpw, "weights_gb": round(w, 1), "rows": rows, "max_ctx": max_ctx, "tok_s": tok_s,
                "native": native}

    table = [make_row(q, bpw) for q, bpw in QUANTS]
    natives = sorted((make_row(f, NATIVE[f], True) for f in hw.get("native_formats", []) if f in NATIVE),
                     key=lambda r: -r["bpw"])
    ref_ctx = min(ref, max_pos) if max_pos else ref
    ctx_ref = ref_ctx if kv_known else 0
    fits = lambda r, c=ctx_ref: total_gb(r["weights_gb"], c) <= usable            # noqa: E731
    comfy = lambda r: total_gb(r["weights_gb"], ctx_ref) <= usable - margin       # noqa: E731
    options = [{"q": r["q"], "native": r["native"], "gb": round(total_gb(r["weights_gb"], ctx_ref), 1), "fits": fits(r),
                "comfy": comfy(r), "max_ctx": r["max_ctx"], "tok_s": r["tok_s"]} for r in table + natives]
    bf16 = table[0]
    gguf_best = next((r for r in table if comfy(r)), None)
    gguf_fit = next((r for r in table if fits(r)), None)
    gguf_tight = gguf_fit if gguf_fit is not None and gguf_fit is not gguf_best else None
    nat_best = next((r for r in natives if comfy(r)), None)
    q4 = next(r for r in table if r["q"] == "Q4_K_M")

    def desc(r):
        if not text_model:
            return f"{r['weights_gb']} GB di pesi"
        if not kv_known:
            return f"{r['weights_gb']} GB di pesi, contesto non stimabile"
        return f"contesto fino a ~{_fmt_ctx(r['max_ctx'] or ref_ctx)}" + (f", {_tok(r['tok_s'])} tok/s" if r["tok_s"] else "")

    def short(r):
        return f"{r['q']} · {_fmt_ctx(r['max_ctx'])}" if text_model and r["max_ctx"] else r["q"]

    primary, tight = None, None
    if comfy(bf16):
        primary = bf16
        level = "ok"
        text = f"Gira su {gpu} senza quantizzare (BF16), {desc(bf16)}"
        label = "BF16 nativo" + (f" · {_fmt_ctx(bf16['max_ctx'])}" if text_model and bf16["max_ctx"] else "")
    elif nat_best or gguf_best:
        parts = []
        if nat_best:
            parts.append(f"{nat_best['q']} (formato nativo della {gpu}): {desc(nat_best)}")
        if gguf_best:
            parts.append(f"GGUF {gguf_best['q']}: {desc(gguf_best)}" + (f"; {gguf_tight['q']} al limite" if gguf_tight else ""))
        tight = gguf_tight
        primary = nat_best if (nat_best and hw.get("prefer_native", True)) else (gguf_best or nat_best)
        good = primary["native"] or primary["q"] in GOOD_QUANTS
        level = "ok" if good else "warn"
        lead = f"Serve quantizzazione (in BF16 i pesi sono ~{bf16['weights_gb']:.0f} GB). " if good else \
            "Solo con quantizzazione aggressiva. "
        text = lead + " · ".join(parts)
        label = short(primary)
    else:
        small_ctx = min(hw["contexts"]) if kv_known else 0
        squeeze = next((r for r in natives + table if fits(r, small_ctx)), None)
        if squeeze:
            primary = squeeze
            level = "warn"
            text = f"Serve quantizzazione e contesto ridotto: {squeeze['q']} fino a ~{_fmt_ctx(squeeze['max_ctx'] or small_ctx or ref_ctx)}"
            label = f"{squeeze['q']} · {_fmt_ctx(squeeze['max_ctx'] or small_ctx or ref_ctx)}"
        elif is_moe and q4["weights_gb"] <= usable + hw["system_ram_gb"] * 0.8:
            level = "warn"
            text = (f"MoE da {params_b:g}B ({act:g}B attivi): non entra in VRAM ma gira con offload degli esperti "
                    f"in RAM (~{q4['weights_gb']:.0f} GB in Q4_K_M, llama.cpp --n-cpu-moe). Velocità limitata dalla RAM")
            label = "offload MoE"
        else:
            level = "no"
            text = f"Troppo grande per una {gpu}: ~{q4['weights_gb']:.0f} GB solo di pesi anche in Q4_K_M"
            label = "non entra"
    return {
        "level": level,
        "label": label,
        "text": text,
        "best_quant": primary["q"] if primary else None,
        "quantized": bool(primary and primary["q"] != "BF16"),
        "native_best": nat_best["q"] if nat_best else None,
        "gguf_best": gguf_best["q"] if gguf_best else None,
        "tight": tight["q"] if tight else None,
        "native_gb": bf16["weights_gb"],
        "ref_ctx": ref_ctx,
        "tok_s": primary["tok_s"] if primary else None,
        "kv_known": kv_known,
        "is_moe": is_moe,
        "options": options,
        "table": table + natives,
    }
