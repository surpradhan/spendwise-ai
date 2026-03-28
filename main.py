#!/usr/bin/env python3
"""
SpendWise AI — CLI Entry Point
================================
Orchestrates the full pipeline: ingest → classify → output → (dashboard).

Usage examples
--------------
# Human review mode (interactive feedback + HTML dashboard)
python main.py --file data/raw/chase_jan2026.csv --dashboard

# Agent / pipe mode (JSON output, no interactive prompts)
python main.py --file data/raw/chase_jan2026.csv --json --no-feedback

# Save JSON summary to a file
python main.py --file data/raw/chase_jan2026.csv --output-json reports/summary.json

# Full help
python main.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add repo root to sys.path so `scripts.*` imports work when running from
# any working directory.
_REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.ingest import ingest
from scripts.classifier import (
    classify_all,
    load_keywords,
    prompt_uncategorized,
    save_keywords,
    write_processed_csv,
)
from scripts.terminal_output import build_summary, print_summary, print_recurring, to_json
from scripts.recurring import detect_recurring


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_KEYWORDS_PATH = _REPO_ROOT / "config" / "keywords.json"
_EXPORTS_DIR   = _REPO_ROOT / "exports"


# ---------------------------------------------------------------------------
# CLI definition
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spendwise",
        description="SpendWise AI — local personal finance analyser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--file", "-f",
        required=True,
        metavar="PATH",
        help="Path to the raw bank export (.csv or .xlsx)",
    )
    parser.add_argument(
        "--keywords",
        default=str(_KEYWORDS_PATH),
        metavar="PATH",
        help=f"Path to keywords.json (default: {_KEYWORDS_PATH})",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Generate a self-contained interactive HTML dashboard",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Generate a multi-page PDF dashboard (requires kaleido + reportlab)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_stdout",
        help="Write JSON summary to stdout instead of human-readable output",
    )
    parser.add_argument(
        "--output-json",
        metavar="PATH",
        help="Write JSON summary to a file",
    )
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="Skip the interactive uncategorized-transaction review step",
    )
    parser.add_argument(
        "--exports-dir",
        default=str(_EXPORTS_DIR),
        metavar="DIR",
        help=f"Directory for HTML dashboard output (default: {_EXPORTS_DIR})",
    )

    return parser


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    """Execute the full SpendWise pipeline."""

    input_path    = Path(args.file)
    keywords_path = Path(args.keywords)
    interactive   = not args.no_feedback

    # ── 1. Ingest ─────────────────────────────────────────────────────────
    df = ingest(input_path, interactive=interactive)

    # ── 2. Classify ───────────────────────────────────────────────────────
    print("[Classifier]")
    keywords = load_keywords(keywords_path)
    df = classify_all(df, keywords)

    # ── 3. Interactive feedback loop ──────────────────────────────────────
    if interactive:
        df, keywords = prompt_uncategorized(df, keywords)
        save_keywords(keywords, keywords_path)
        # Re-classify after corrections
        df = classify_all(df, keywords)

    # ── 4. Persist categorised CSV ────────────────────────────────────────
    write_processed_csv(df, input_path)

    # ── 5. Build summary + detect recurring ───────────────────────────────
    summary      = build_summary(df)
    recurring_df = detect_recurring(df)

    # ── 6. Output ─────────────────────────────────────────────────────────
    if args.json_stdout:
        payload = json.loads(to_json(summary))
        payload["recurring_transactions"] = recurring_df.to_dict(orient="records")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print_summary(summary)
        print_recurring(recurring_df)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(to_json(summary), encoding="utf-8")
        print(f"  ✓ JSON summary saved → '{out}'")

    # ── 7. Dashboard / PDF ────────────────────────────────────────────────
    if args.dashboard:
        from scripts.dashboard import export_dashboard
        export_dashboard(df, Path(args.exports_dir))

    if args.pdf:
        from scripts.dashboard import export_pdf
        export_pdf(df, Path(args.exports_dir))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()
    try:
        run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\n  ✗ Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n  ↩  Interrupted by user.", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
