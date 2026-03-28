"""Tests for scripts/terminal_output.py"""
import json

import pandas as pd
import pytest

from scripts.terminal_output import build_summary, print_recurring, to_json


def _categorized_df():
    return pd.DataFrame({
        "Date": [
            "2024-01-05", "2024-01-10", "2024-01-15",
            "2024-01-20", "2024-01-25",
        ],
        "Description": [
            "Starbucks", "Lyft", "Salary", "Netflix", "Whole Foods",
        ],
        "Amount": [-5.50, -12.00, 2000.00, -15.99, -80.00],
        "Category": [
            "Food & Drink", "Transport", "Income", "Entertainment", "Groceries",
        ],
    })


# ---------------------------------------------------------------------------
# build_summary
# ---------------------------------------------------------------------------

def test_build_summary_period_dates():
    summary = build_summary(_categorized_df())
    assert summary["period_start"] == "2024-01-05"
    assert summary["period_end"] == "2024-01-25"


def test_build_summary_total_income():
    summary = build_summary(_categorized_df())
    assert summary["total_income"] == pytest.approx(2000.00)


def test_build_summary_total_expenses():
    summary = build_summary(_categorized_df())
    expected = 5.50 + 12.00 + 15.99 + 80.00
    assert summary["total_expenses"] == pytest.approx(expected)


def test_build_summary_net():
    summary = build_summary(_categorized_df())
    assert summary["net"] == pytest.approx(2000.00 - (5.50 + 12.00 + 15.99 + 80.00))


def test_build_summary_category_totals_keys():
    summary = build_summary(_categorized_df())
    cats = set(summary["category_totals"].keys())
    # Income rows are positive → excluded from expense totals
    assert "Food & Drink" in cats
    assert "Transport" in cats
    assert "Entertainment" in cats
    assert "Groceries" in cats
    assert "Income" not in cats  # positive amounts excluded


def test_build_summary_category_totals_sorted_desc():
    summary = build_summary(_categorized_df())
    values = list(summary["category_totals"].values())
    assert values == sorted(values, reverse=True)


def test_build_summary_top_merchants_list():
    summary = build_summary(_categorized_df())
    assert isinstance(summary["top_merchants"], list)
    assert all("merchant" in m and "total" in m for m in summary["top_merchants"])


def test_build_summary_top_merchants_max_10():
    # Build a df with 15 distinct merchants
    rows = {
        "Date": ["2024-01-01"] * 15,
        "Description": [f"Merchant{i}" for i in range(15)],
        "Amount": [-float(i + 1) for i in range(15)],
        "Category": ["Shopping"] * 15,
    }
    df = pd.DataFrame(rows)
    summary = build_summary(df)
    assert len(summary["top_merchants"]) <= 10


def test_build_summary_uncategorized_count():
    df = _categorized_df().copy()
    df.loc[0, "Category"] = "Uncategorized"
    summary = build_summary(df)
    assert summary["uncategorized_count"] == 1


def test_build_summary_no_uncategorized():
    summary = build_summary(_categorized_df())
    assert summary["uncategorized_count"] == 0


def test_build_summary_required_keys():
    summary = build_summary(_categorized_df())
    required = {
        "period_start", "period_end", "total_income", "total_expenses",
        "net", "category_totals", "top_merchants",
        "uncategorized_count", "uncategorized_sample",
    }
    assert required.issubset(summary.keys())


# ---------------------------------------------------------------------------
# to_json
# ---------------------------------------------------------------------------

def test_to_json_returns_valid_json():
    summary = build_summary(_categorized_df())
    result = to_json(summary)
    parsed = json.loads(result)  # raises if invalid
    assert parsed["total_income"] == pytest.approx(2000.00)


def test_to_json_returns_string():
    summary = build_summary(_categorized_df())
    assert isinstance(to_json(summary), str)


# ---------------------------------------------------------------------------
# print_recurring
# ---------------------------------------------------------------------------

def _recurring_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Description": ["NETFLIX", "SPOTIFY"],
        "Category":    ["Entertainment", "Entertainment"],
        "Avg_Amount":  [-15.49, -9.99],
        "Median_Days": [30.0, 30.0],
        "Frequency":   ["Monthly", "Monthly"],
        "Occurrences": [3, 3],
        "First_Date":  ["2026-01-01", "2026-01-01"],
        "Last_Date":   ["2026-03-01", "2026-03-01"],
    })


def test_print_recurring_empty_produces_no_output(capsys):
    """print_recurring with an empty DataFrame prints nothing."""
    print_recurring(pd.DataFrame(columns=[
        "Description", "Category", "Avg_Amount", "Median_Days",
        "Frequency", "Occurrences", "First_Date", "Last_Date",
    ]))
    captured = capsys.readouterr()
    assert captured.out == ""


def test_print_recurring_shows_description(capsys):
    """Each description appears in the output."""
    print_recurring(_recurring_df())
    captured = capsys.readouterr()
    assert "NETFLIX" in captured.out
    assert "SPOTIFY" in captured.out


def test_print_recurring_shows_frequency(capsys):
    """Frequency label appears in the output."""
    print_recurring(_recurring_df())
    captured = capsys.readouterr()
    assert "Monthly" in captured.out


def test_print_recurring_shows_amount(capsys):
    """Formatted per-cycle amount appears in the output."""
    print_recurring(_recurring_df())
    captured = capsys.readouterr()
    assert "15.49" in captured.out


def test_print_recurring_masks_card_numbers(capsys):
    """Descriptions containing card-like digit sequences are masked before printing."""
    df = pd.DataFrame({
        "Description": ["CHARGE 4111111111111111"],
        "Category":    ["Shopping"],
        "Avg_Amount":  [-50.00],
        "Median_Days": [30.0],
        "Frequency":   ["Monthly"],
        "Occurrences": [2],
        "First_Date":  ["2026-01-01"],
        "Last_Date":   ["2026-02-01"],
    })
    print_recurring(df)
    captured = capsys.readouterr()
    assert "4111111111111111" not in captured.out
    assert "****1111" in captured.out
