from __future__ import annotations

import tomllib
from pathlib import Path


def load_config(path: Path) -> dict:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    cfg.setdefault("profile", {})
    cfg.setdefault("hardware", {})
    cfg.setdefault("run", {})
    cfg.setdefault("llm", {})
    cfg.setdefault("sources", {})
    hw = cfg["hardware"]
    hw.setdefault("gpu_name", "GPU")
    hw.setdefault("vram_gb", 24)
    hw.setdefault("reserved_gb", 1.0)
    hw.setdefault("bandwidth_gbs", 1000)
    hw.setdefault("system_ram_gb", 32)
    hw.setdefault("ref_context", 32768)
    hw.setdefault("contexts", [8192, 32768, 131072])
    run = cfg["run"]
    run.setdefault("timezone", "UTC")
    run.setdefault("lookback_hours", 30)
    run.setdefault("bootstrap_days", 7)
    run.setdefault("new_model_days", 10)
    run.setdefault("max_items_per_day", 150)
    run.setdefault("min_relevance", 3)
    llm = cfg["llm"]
    llm.setdefault("max_calls_per_run", 24)
    llm.setdefault("triage_items", 100)
    llm.setdefault("triage_batch", 20)
    llm.setdefault("max_deep_dives", 8)
    llm.setdefault("temperature", 0.2)
    llm.setdefault("providers", [])
    return cfg
