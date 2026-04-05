#!/usr/bin/env python3
"""
SpendWise AI — Local Web UI
============================
Wraps the existing pipeline in a FastAPI server so you can upload bank
exports and query your spending from the browser — all locally, no cloud.

Usage
-----
    pip install fastapi uvicorn python-multipart
    python app.py

Then open http://localhost:8000 in your browser.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

# ── Repo root on sys.path so scripts.* imports work ──────────────────────────
_REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.budget import evaluate_budgets, load_budgets
from scripts.classifier import load_keywords, write_processed_csv
from scripts.ingest import ingest
from scripts.ml_classifier import classify_all_v2, load_model
from scripts.nl_query_agent import execute_query_agent as execute_query
from scripts.recurring import detect_recurring
from scripts.terminal_output import build_summary, currency_label

# ── Default paths ─────────────────────────────────────────────────────────────
_KEYWORDS_PATH = _REPO_ROOT / "config" / "keywords.json"
_BUDGETS_PATH  = _REPO_ROOT / "config" / "budgets.json"
_ML_CONFIG     = _REPO_ROOT / "config" / "ml_config.json"
_MODEL_DIR     = _REPO_ROOT / "models"
_EXPORTS_DIR   = _REPO_ROOT / "exports"
_TEMPLATES_DIR = _REPO_ROOT / "templates"

# ── In-memory state (local single-user app — global is fine) ─────────────────
_state: dict = {"df": None, "summary": None, "currency_sym": "$"}

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="SpendWise AI", docs_url=None, redoc_url=None)


def _load_threshold() -> float:
    """Read confidence_threshold from ml_config.json, default 0.70."""
    if _ML_CONFIG.exists():
        return json.loads(_ML_CONFIG.read_text()).get("confidence_threshold", 0.70)
    return 0.70


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the single-page frontend."""
    html_path = _TEMPLATES_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(500, "templates/index.html not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> JSONResponse:
    """Accept a CSV or XLSX bank export, run the full pipeline, return results."""
    suffix = Path(file.filename or "upload.csv").suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(400, f"Unsupported file type: {suffix!r}. Use CSV or XLSX.")

    # Save upload to a temp file so ingest() can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        # ── Ingest ────────────────────────────────────────────────────────────
        df = ingest(tmp_path, interactive=False)

        # ── Classify ──────────────────────────────────────────────────────────
        keywords  = load_keywords(_KEYWORDS_PATH)
        threshold = _load_threshold()
        df = classify_all_v2(df, keywords, model_path=_MODEL_DIR, threshold=threshold)

        # Persist categorised CSV (feeds future ML retraining)
        write_processed_csv(df, tmp_path)

        # ── Summary & downstream ───────────────────────────────────────────────
        summary  = build_summary(df)
        budgets  = load_budgets(_BUDGETS_PATH)
        alerts   = evaluate_budgets(summary, budgets)
        recurring = detect_recurring(df)
        cur_sym  = currency_label(summary.get("currencies", ["USD"])[0])

        # Stash for /query calls
        _state["df"]           = df
        _state["summary"]      = summary
        _state["currency_sym"] = cur_sym

        return JSONResponse({
            "filename":      file.filename,
            "summary":       summary,
            "budget_alerts": alerts,
            "recurring":     recurring.to_dict(orient="records"),
            "currency_sym":  cur_sym,
        })

    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/query")
async def query(request: Request) -> JSONResponse:
    """Run a plain-English query against the last uploaded file."""
    if _state["df"] is None:
        raise HTTPException(400, "No file uploaded yet. Upload a bank export first.")

    body = await request.json()
    q    = (body.get("query") or "").strip()
    if not q:
        raise HTTPException(400, "Query must not be empty.")

    try:
        result = execute_query(q, _state["df"], currency_sym=_state["currency_sym"])
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    return JSONResponse({"result": result})


@app.get("/pdf")
async def pdf() -> FileResponse:
    """Generate and return the PDF report for the uploaded file as a download."""
    if _state["df"] is None:
        raise HTTPException(400, "No file uploaded yet.")

    from scripts.dashboard import export_pdf
    _EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    budgets = load_budgets(_BUDGETS_PATH)

    try:
        pdf_path = export_pdf(_state["df"], _EXPORTS_DIR, budgets=budgets)
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=pdf_path.name,
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    """Generate and return the Plotly HTML dashboard for the uploaded file."""
    if _state["df"] is None:
        raise HTTPException(400, "No file uploaded yet.")

    from scripts.dashboard import export_dashboard
    _EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    budgets = load_budgets(_BUDGETS_PATH)
    export_dashboard(_state["df"], _EXPORTS_DIR, budgets=budgets)

    # Return the most-recently-written dashboard file
    candidates = sorted(_EXPORTS_DIR.glob("dashboard_*.html"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise HTTPException(500, "Dashboard generation failed.")

    return HTMLResponse(content=candidates[-1].read_text(encoding="utf-8"))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("SpendWise AI — Web UI")
    print("Open http://localhost:8000 in your browser\n")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
