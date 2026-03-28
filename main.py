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

# Retrain ML classifier from processed CSVs
python main.py --file data/raw/chase_jan2026.csv --retrain-ml

# Full help
python main.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add repo root to sys.path so `scripts.*` imports work when running from
# any working directory.
_REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.ingest import ingest
from scripts.classifier import (
    load_keywords,
    prompt_uncategorized,
    save_keywords,
    write_processed_csv,
)
from scripts.ml_classifier import (
    classify_all_v2,
    is_model_stale,
    load_model,
    load_training_data,
    save_model,
    train_model,
)
from scripts.terminal_output import build_summary, print_summary, print_recurring, to_json
from scripts.recurring import detect_recurring


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_KEYWORDS_PATH  = _REPO_ROOT / "config" / "keywords.json"
_EXPORTS_DIR    = _REPO_ROOT / "exports"
_ML_CONFIG_PATH = _REPO_ROOT / "config" / "ml_config.json"
_MODEL_DIR      = _REPO_ROOT / "models"
_PROCESSED_DIR  = _REPO_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_ml_config() -> dict:
    """Load ML configuration from ml_config.json, with safe defaults."""
    if _ML_CONFIG_PATH.exists():
        with _ML_CONFIG_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"confidence_threshold": 0.70, "min_training_samples": 20, "model_dir": "models"}


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
    parser.add_argument(
        "--retrain-ml",
        action="store_true",
        help=(
            "Retrain the ML classifier from all processed CSVs in data/processed/ "
            "after writing the categorised output for this run."
        ),
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
    ml_config     = _load_ml_config()
    threshold     = ml_config.get("confidence_threshold", 0.70)

    # ── 1. Ingest ─────────────────────────────────────────────────────────
    df = ingest(input_path, interactive=interactive)

    # ── 2. Classify (keywords first, ML for remainders) ───────────────────
    print("[Classifier]")
    keywords = load_keywords(keywords_path)

    # Advisory: warn if saved model is older than newest processed CSV
    if is_model_stale(_MODEL_DIR, _PROCESSED_DIR):
        print(
            "  [ML] Advisory: model may be stale — consider re-running with "
            "--retrain-ml to incorporate new training data."
        )

    # Cold-start info when no model exists
    if load_model(_MODEL_DIR) is None:
        print("  ML classifier not yet trained — using keyword matching only.")

    df = classify_all_v2(df, keywords, model_path=_MODEL_DIR, threshold=threshold)

    # ── 3. Interactive feedback loop ──────────────────────────────────────
    if interactive:
        # prompt_uncategorized works on keyword-classified data; pass df with
        # Category already populated by classify_all_v2.
        df, keywords = prompt_uncategorized(df, keywords)
        save_keywords(keywords, keywords_path)
        # Re-classify after corrections (re-runs both keyword + ML passes)
        df = classify_all_v2(df, keywords, model_path=_MODEL_DIR, threshold=threshold)

    # ── 4. Persist categorised CSV ────────────────────────────────────────
    write_processed_csv(df, input_path)

    # ── 5. (Optional) Retrain ML classifier ──────────────────────────────
    if args.retrain_ml:
        print("\n[ML Trainer]")
        X, y = load_training_data(_PROCESSED_DIR)
        try:
            model = train_model(X, y)
            meta = {
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "n_samples": len(X),
                "classes": sorted(y.unique().tolist()),
            }
            save_model(model, meta, _MODEL_DIR)
            print(
                f"  ✓ ML classifier trained on {len(X):,} samples "
                f"({y.nunique()} categories) → '{_MODEL_DIR}'"
            )
        except RuntimeError as exc:
            print(f"  ✗ ML training skipped: {exc}")

    # ── 6. Build summary + detect recurring ───────────────────────────────
    summary      = build_summary(df)
    recurring_df = detect_recurring(df)

    # ── 7. Output ─────────────────────────────────────────────────────────
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

    # ── 8. Dashboard / PDF ────────────────────────────────────────────────
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
