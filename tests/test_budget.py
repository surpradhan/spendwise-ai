"""Tests for scripts/budget.py"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.budget import evaluate_budgets, load_budgets, save_budgets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_budgets(tmp_path: Path, data: dict) -> Path:
    """Write *data* as JSON to a temp file and return the path."""
    p = tmp_path / "budgets.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _make_summary(
    period_start: str,
    period_end: str,
    category_totals: dict[str, float],
) -> dict:
    """Build a minimal summary dict compatible with build_summary output."""
    return {
        "period_start": period_start,
        "period_end":   period_end,
        "total_income": 0.0,
        "total_expenses": sum(category_totals.values()),
        "net": 0.0,
        "category_totals": category_totals,
        "top_merchants": [],
        "uncategorized_count": 0,
        "uncategorized_sample": [],
    }


# ---------------------------------------------------------------------------
# load_budgets
# ---------------------------------------------------------------------------

def test_load_budgets_returns_empty_dict_when_file_missing(tmp_path):
    result = load_budgets(tmp_path / "nonexistent.json")
    assert result == {}


def test_load_budgets_returns_correct_mapping(tmp_path):
    p = _write_budgets(tmp_path, {"Groceries": 400.0, "Transport": 100.0})
    result = load_budgets(p)
    assert result == {"Groceries": 400.0, "Transport": 100.0}


def test_load_budgets_raises_value_error_on_negative_amount(tmp_path):
    p = _write_budgets(tmp_path, {"Groceries": -50.0})
    with pytest.raises(ValueError, match="Groceries"):
        load_budgets(p)


def test_load_budgets_raises_value_error_on_zero_amount(tmp_path):
    p = _write_budgets(tmp_path, {"Entertainment": 0})
    with pytest.raises(ValueError, match="Entertainment"):
        load_budgets(p)


def test_load_budgets_raises_value_error_on_non_numeric_value(tmp_path):
    p = _write_budgets(tmp_path, {"Groceries": "four hundred"})
    with pytest.raises(ValueError, match="Groceries"):
        load_budgets(p)


# ---------------------------------------------------------------------------
# save_budgets
# ---------------------------------------------------------------------------

def test_save_budgets_writes_sorted_keys(tmp_path):
    p = tmp_path / "out.json"
    save_budgets({"Transport": 100.0, "Groceries": 400.0, "Entertainment": 50.0}, p)
    data = json.loads(p.read_text())
    assert list(data.keys()) == sorted(data.keys())


def test_save_budgets_round_trips_correctly(tmp_path):
    p = tmp_path / "out.json"
    original = {"Groceries": 400.0, "Transport": 100.0}
    save_budgets(original, p)
    result = load_budgets(p)
    assert result == original


# ---------------------------------------------------------------------------
# evaluate_budgets
# ---------------------------------------------------------------------------

def test_evaluate_budgets_exceeded_when_over_100pct():
    summary = _make_summary("2026-01-01", "2026-01-31", {"Groceries": 500.0})
    budgets = {"Groceries": 400.0}
    alerts = evaluate_budgets(summary, budgets)
    assert len(alerts) == 1
    assert alerts[0]["status"] == "EXCEEDED"


def test_evaluate_budgets_approaching_when_between_threshold_and_100pct():
    # 85% usage with default 80% threshold
    summary = _make_summary("2026-01-01", "2026-01-31", {"Groceries": 340.0})
    budgets = {"Groceries": 400.0}
    alerts = evaluate_budgets(summary, budgets)
    assert alerts[0]["status"] == "APPROACHING"


def test_evaluate_budgets_ok_when_under_threshold():
    # 50% usage
    summary = _make_summary("2026-01-01", "2026-01-31", {"Groceries": 200.0})
    budgets = {"Groceries": 400.0}
    alerts = evaluate_budgets(summary, budgets)
    assert alerts[0]["status"] == "OK"


def test_evaluate_budgets_normalises_to_monthly_avg_for_multi_month_data():
    # 2 months of data, total spend = 800
    summary = _make_summary("2026-01-01", "2026-02-28", {"Groceries": 800.0})
    budgets = {"Groceries": 400.0}
    alerts = evaluate_budgets(summary, budgets)
    assert alerts[0]["num_months"] == 2
    assert alerts[0]["monthly_avg"] == pytest.approx(400.0)
    assert alerts[0]["status"] == "EXCEEDED"  # 400/400 = 100%


def test_evaluate_budgets_single_month_uses_raw_total():
    summary = _make_summary("2026-03-01", "2026-03-31", {"Transport": 75.0})
    budgets = {"Transport": 100.0}
    alerts = evaluate_budgets(summary, budgets)
    assert alerts[0]["num_months"] == 1
    assert alerts[0]["monthly_avg"] == pytest.approx(75.0)


def test_evaluate_budgets_skips_unbudgeted_categories():
    summary = _make_summary("2026-01-01", "2026-01-31", {
        "Groceries": 300.0,
        "Entertainment": 60.0,
    })
    budgets = {"Groceries": 400.0}  # Entertainment not budgeted
    alerts = evaluate_budgets(summary, budgets)
    categories = [a["category"] for a in alerts]
    assert "Entertainment" not in categories
    assert "Groceries" in categories


def test_evaluate_budgets_returns_sorted_by_pct_used_descending():
    summary = _make_summary("2026-01-01", "2026-01-31", {
        "Transport": 50.0,
        "Groceries": 380.0,
        "Entertainment": 45.0,
    })
    budgets = {"Transport": 100.0, "Groceries": 400.0, "Entertainment": 50.0}
    alerts = evaluate_budgets(summary, budgets)
    pcts = [a["pct_used"] for a in alerts]
    assert pcts == sorted(pcts, reverse=True)


def test_evaluate_budgets_empty_budgets_returns_empty_list():
    summary = _make_summary("2026-01-01", "2026-01-31", {"Groceries": 300.0})
    assert evaluate_budgets(summary, {}) == []


def test_evaluate_budgets_custom_warn_threshold():
    # 70% usage, default threshold is 80% → OK; custom threshold 60% → APPROACHING
    summary = _make_summary("2026-01-01", "2026-01-31", {"Transport": 70.0})
    budgets = {"Transport": 100.0}
    default_alert = evaluate_budgets(summary, budgets)
    assert default_alert[0]["status"] == "OK"

    custom_alert = evaluate_budgets(summary, budgets, warn_threshold=0.60)
    assert custom_alert[0]["status"] == "APPROACHING"
