"""Uso:
  python3 -m radar run                 # esegue la rassegna di oggi (aggiorna docs/data)
  python3 -m radar run --no-llm        # senza LLM (solo euristiche)
  python3 -m radar run --days 7        # forza una finestra di N giorni
  python3 -m radar serve [--port 8000] # anteprima locale del sito
  python3 -m radar check <org/modello> # verdetto VRAM per un modello Hugging Face qualsiasi
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main():
    ap = argparse.ArgumentParser(prog="radar", description="AI Radar — rassegna quotidiana personale")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="esegue la rassegna")
    r.add_argument("--no-llm", action="store_true", help="non usare l'LLM")
    r.add_argument("--days", type=int, default=None, help="finestra temporale in giorni")
    r.add_argument("--refresh-deep", action="store_true", help="rifà schede ed editoriale dei modelli di oggi")
    r.add_argument("--config", default=str(ROOT / "config.toml"))
    s = sub.add_parser("serve", help="anteprima locale del sito")
    s.add_argument("--port", type=int, default=8000)
    c = sub.add_parser("check", help="verdetto VRAM per un repo HF")
    c.add_argument("repo")
    c.add_argument("--config", default=str(ROOT / "config.toml"))
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")

    if args.cmd == "run":
        from .config import load_config
        from .pipeline import run
        run(ROOT, load_config(Path(args.config)), use_llm=not args.no_llm, bootstrap_days=args.days,
            refresh_deep=args.refresh_deep)
    elif args.cmd == "serve":
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT / "docs"))
        print(f"AI Radar su http://localhost:{args.port}  (Ctrl+C per uscire)")
        http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()
    elif args.cmd == "check":
        from . import models as M
        from .config import load_config
        cfg = load_config(Path(args.config))
        api = M.hf_model(args.repo)
        if not api:
            raise SystemExit(f"Modello non trovato: {args.repo}")
        e = M.entry_from_hf(api, cfg["hardware"])
        v = e.get("vram") or {}
        print(f"{e['id']}  —  {e.get('params_b')}B parametri"
              + (f" ({e['active_b']}B attivi)" if e.get("active_b") else "")
              + f", contesto {e.get('context')}, licenza {e.get('license')}")
        print(f"Verdetto: {v.get('text', 'n/d')}\n")
        if v:
            ctxs = cfg["hardware"]["contexts"]
            print(f"{'quant':8} {'pesi':>7} " + " ".join(f"{c // 1024:>6}k" for c in ctxs) + "   ctx max   tok/s")
            for row in v["table"]:
                cells = []
                for cell in row["rows"]:
                    cells.append("     -" if cell is None else f"{cell['gb']:>5.1f}{'✓' if cell['fits'] else '✗'}")
                mc = f"{row['max_ctx'] // 1024}k" if row["max_ctx"] else "-"
                print(f"{row['q']:8} {row['weights_gb']:>6.1f}G " + "  ".join(cells) + f"   {mc:>7}   {row['tok_s'] or '-':>5}")
        if e.get("quants"):
            print("\nQuantizzazioni: " + ", ".join(f"{q['id']} [{q['format']}]" for q in e["quants"]))
        if e.get("note"):
            print(f"\nNota: {e['note']}")
        print("\n" + json.dumps(e.get("arch"), indent=None))


if __name__ == "__main__":
    main()
