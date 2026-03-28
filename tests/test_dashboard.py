"""Tests for scripts/dashboard.py — PDF export and budget chart."""
from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.dashboard import build_budget_chart, build_dashboard, export_pdf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def classified_df() -> pd.DataFrame:
    """Minimal classified DataFrame that satisfies the canonical schema."""
    return pd.DataFrame({
        "Date":        ["2026-01-05", "2026-01-10", "2026-01-15",
                        "2026-02-05", "2026-02-10", "2026-02-15"],
        "Description": ["NETFLIX", "WHOLE FOODS", "SALARY DEPOSIT",
                        "NETFLIX", "WHOLE FOODS", "AMAZON PRIME"],
        "Amount":      [-15.49, -87.32, 3500.00,
                        -15.49, -91.14, -14.99],
        "Category":    ["Entertainment", "Groceries", "Income",
                        "Entertainment", "Groceries", "Shopping"],
    })


@pytest.fixture(scope="session")
def tiny_png() -> bytes:
    """Generate a valid 4×4 white PNG once per test session (lazy, not at import time)."""
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color=(255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# export_pdf — happy path
# ---------------------------------------------------------------------------

def test_export_pdf_creates_file(classified_df, tmp_path, tiny_png):
    """export_pdf() writes a .pdf file to the output directory."""
    with patch("plotly.io.to_image", return_value=tiny_png):
        out = export_pdf(classified_df, tmp_path)

    assert out.suffix == ".pdf"
    assert out.exists()
    assert out.stat().st_size > 0


def test_export_pdf_filename_contains_date_range(classified_df, tmp_path, tiny_png):
    """PDF filename encodes the transaction date range."""
    with patch("plotly.io.to_image", return_value=tiny_png):
        out = export_pdf(classified_df, tmp_path)

    assert "2026-01-05" in out.name
    assert "2026-02-15" in out.name


def test_export_pdf_creates_output_dir(classified_df, tmp_path, tiny_png):
    """export_pdf() creates the output directory if it does not exist."""
    new_dir = tmp_path / "nested" / "exports"
    assert not new_dir.exists()

    with patch("plotly.io.to_image", return_value=tiny_png):
        export_pdf(classified_df, new_dir)

    assert new_dir.exists()


def test_export_pdf_renders_four_charts(classified_df, tmp_path, tiny_png):
    """kaleido is called once per chart (4 charts total)."""
    with patch("plotly.io.to_image", return_value=tiny_png) as mock_to_image:
        export_pdf(classified_df, tmp_path)

    assert mock_to_image.call_count == 4


def test_export_pdf_with_uncategorized_produces_larger_file(tmp_path, tiny_png):
    """PDF is larger when uncategorized rows are present (extra page rendered)."""
    df_clean = pd.DataFrame({
        "Date":        ["2026-01-05", "2026-01-10"],
        "Description": ["KNOWN VENDOR A", "KNOWN VENDOR B"],
        "Amount":      [-25.00, -10.00],
        "Category":    ["Shopping", "Shopping"],
    })
    df_uncat = pd.DataFrame({
        "Date":        ["2026-01-05", "2026-01-10"],
        "Description": ["MYSTERY CHARGE", "KNOWN VENDOR"],
        "Amount":      [-25.00, -10.00],
        "Category":    ["Uncategorized", "Shopping"],
    })
    with patch("plotly.io.to_image", return_value=tiny_png):
        out_clean = export_pdf(df_clean, tmp_path / "clean")
        out_uncat = export_pdf(df_uncat, tmp_path / "uncat")

    # The extra uncategorized page must add measurable bytes to the PDF.
    assert out_uncat.stat().st_size > out_clean.stat().st_size


# ---------------------------------------------------------------------------
# export_pdf — error handling
# ---------------------------------------------------------------------------

def test_export_pdf_raises_when_reportlab_missing(classified_df, tmp_path):
    """RuntimeError with clear message when reportlab is not installed."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "reportlab.lib.pagesizes":
            raise ImportError("No module named 'reportlab'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        with pytest.raises(RuntimeError, match="reportlab"):
            export_pdf(classified_df, tmp_path)


def test_export_pdf_raises_when_kaleido_missing(classified_df, tmp_path):
    """RuntimeError with clear message when kaleido is not installed."""
    kaleido_err = ValueError(
        "You must install the kaleido package in order to use `figure.to_image()`"
    )
    with patch("plotly.io.to_image", side_effect=kaleido_err):
        with pytest.raises(RuntimeError, match="kaleido"):
            export_pdf(classified_df, tmp_path)


# ---------------------------------------------------------------------------
# build_budget_chart
# ---------------------------------------------------------------------------

def test_build_budget_chart_returns_figure(classified_df):
    """build_budget_chart returns a go.Figure with budgets provided."""
    budgets = {"Groceries": 200.0, "Entertainment": 50.0}
    fig = build_budget_chart(classified_df, budgets)
    assert isinstance(fig, go.Figure)


def test_build_budget_chart_empty_budgets_returns_figure(classified_df):
    """build_budget_chart returns a go.Figure with annotation when budgets empty."""
    fig = build_budget_chart(classified_df, {})
    assert isinstance(fig, go.Figure)
    # Should contain the "No budgets configured" annotation
    annotations = fig.layout.annotations
    assert any("No budgets configured" in (a.text or "") for a in annotations)


# ---------------------------------------------------------------------------
# build_dashboard — budget card integration
# ---------------------------------------------------------------------------

def test_build_dashboard_accepts_budgets_param(classified_df):
    """build_dashboard runs without error when budgets is provided."""
    budgets = {"Groceries": 200.0, "Entertainment": 50.0}
    html = build_dashboard(classified_df, budgets=budgets)
    assert isinstance(html, str)
    assert "<!DOCTYPE html>" in html


def test_build_dashboard_no_budget_card_when_budgets_empty(classified_df):
    """build_dashboard without budgets omits the budget chart card."""
    html_no_budget = build_dashboard(classified_df, budgets=None)
    html_with_budget = build_dashboard(classified_df, budgets={"Groceries": 200.0})
    # Dashboard with budgets should be larger (extra card HTML)
    assert len(html_with_budget) > len(html_no_budget)
