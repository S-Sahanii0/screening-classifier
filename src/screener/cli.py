"""Command line: `screener classify` for one paper, `screener eval` for the full ablation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .classify import Classifier
from .data import fewshot_papers, load_records
from .evaluate import (disagreements_markdown, report_for, run_variant,
                       summary_markdown, write_predictions)
from .llm import LLMClient, LLMConfig, LLMError
from .prompts import VARIANTS

import os


def _client(args) -> LLMClient:
    cfg = LLMConfig.from_env(
        provider=args.provider,
        model=args.model,
        min_interval=args.min_interval,
        cache_dir=None if args.no_cache else ".cache/llm",
    )
    return LLMClient(cfg)


def cmd_classify(args) -> int:
    client = _client(args)
    fs = None
    if args.variant == "few_shot":
        fs = fewshot_papers(load_records(args.data))
    pred = Classifier(client, args.variant, fs).classify(args.title, args.abstract)
    print(json.dumps(pred.to_dict(), indent=2))
    return 0


def cmd_eval(args) -> int:
    records = load_records(args.data)
    client = _client(args)
    out = Path(args.out)
    reports, all_rows = {}, {}
    for variant in args.variants:
        print(f"Running {variant} ...")
        rows = run_variant(client, variant, records, limit=args.limit)
        write_predictions(rows, out / f"predictions_{variant}.csv")
        if rows:
            all_rows[variant] = rows
            reports[variant] = report_for(rows)
    if not reports:
        print("No predictions were produced.", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(summary_markdown(reports, client.name), encoding="utf-8")
    for variant, rows in all_rows.items():
        (out / f"disagreements_{variant}.md").write_text(
            disagreements_markdown(rows, variant), encoding="utf-8")
    print(f"\nWrote {out}/summary.md and per-variant files.")
    print((out / "summary.md").read_text(encoding="utf-8"))
    return 0

def load_env_file(path: str = ".env") -> None:
    """Read KEY=value lines from .env into the environment. Real env vars win."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

def main(argv: list[str] | None = None) -> int:
    load_env_file()
    ap = argparse.ArgumentParser(prog="screener", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--provider", choices=["gemini", "groq", "ollama"], default=None,
                       help="default: SCREENER_PROVIDER or gemini")
        p.add_argument("--model", default=None, help="override the provider's default model")
        p.add_argument("--data", default="data/records.csv")
        p.add_argument("--min-interval", type=float, default=None,
                       help="seconds between calls (use ~5 on free tiers)")
        p.add_argument("--no-cache", action="store_true")

    c = sub.add_parser("classify", help="classify one paper")
    common(c)
    c.add_argument("--title", required=True)
    c.add_argument("--abstract", required=True)
    c.add_argument("--variant", choices=VARIANTS, default="rules")
    c.set_defaults(func=cmd_classify)

    e = sub.add_parser("eval", help="score prompt variants against my labels")
    common(e)
    e.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    e.add_argument("--out", default="results")
    e.add_argument("--limit", type=int, default=None, help="only the first N papers (smoke test)")
    e.set_defaults(func=cmd_eval)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except (LLMError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
