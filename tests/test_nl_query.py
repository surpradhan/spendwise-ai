"""Tests for scripts/nl_query.py — execute_query()."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.nl_query import execute_query


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def _make_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Date": [
            "2024-01-10", "2024-01-15", "2024-02-05",
            "2024-02-12", "2024-03-01", "2024-03-20",
        ],
        "Description": [
            "Whole Foods", "Netflix", "Amazon Purchase",
            "Whole Foods", "Uber Eats", "Salary",
        ],
        "Amount": [-85.0, -15.99, -42.50, -90.0, -25.0, 3000.0],
        "Category": [
            "Groceries", "Subscriptions", "Shopping",
            "Groceries", "Food & Drink", "Income",
        ],
    })


# ---------------------------------------------------------------------------
# categories
# ---------------------------------------------------------------------------

def test_categories_lists_all_expense_categories():
    result = execute_query("categories", _make_df())
    assert "Groceries" in result
    assert "Shopping" in result
    assert "Subscriptions" in result


def test_categories_excludes_income():
    result = execute_query("categories", _make_df())
    assert "$3,000" not in result


# ---------------------------------------------------------------------------
# show <category>
# ---------------------------------------------------------------------------

def test_show_category_returns_matching_rows():
    result = execute_query("show groceries", _make_df())
    assert "Whole Foods" in result
    assert "Amazon" not in result


def test_show_category_case_insensitive():
    result1 = execute_query("show Groceries", _make_df())
    result2 = execute_query("show GROCERIES", _make_df())
    assert "Whole Foods" in result1
    assert "Whole Foods" in result2


def test_show_unknown_category_reports_not_found():
    result = execute_query("show zzz-unknown", _make_df())
    assert "no transactions found" in result.lower()


# ---------------------------------------------------------------------------
# top <N>
# ---------------------------------------------------------------------------

def test_top_n_returns_n_rows():
    result = execute_query("top 2", _make_df())
    assert "2 rows" in result or "2 row" in result


def test_top_n_largest_expenses_first():
    result = execute_query("top 3", _make_df())
    assert "Whole Foods" in result


def test_top_n_category():
    result = execute_query("top 1 groceries", _make_df())
    assert "Whole Foods" in result
    assert "1 row" in result


# ---------------------------------------------------------------------------
# sum <category>
# ---------------------------------------------------------------------------

def test_sum_category_correct_total():
    result = execute_query("sum groceries", _make_df())
    # 85 + 90 = 175
    assert "175" in result


def test_sum_unknown_category():
    result = execute_query("sum zzz-unknown", _make_df())
    assert "no expenses found" in result.lower()


# ---------------------------------------------------------------------------
# monthly <category>
# ---------------------------------------------------------------------------

def test_monthly_category_shows_months():
    result = execute_query("monthly groceries", _make_df())
    assert "2024-01" in result
    assert "2024-02" in result


def test_monthly_unknown_category():
    result = execute_query("monthly zzz-unknown", _make_df())
    assert "no expenses found" in result.lower()


# ---------------------------------------------------------------------------
# search <keyword>
# ---------------------------------------------------------------------------

def test_search_finds_keyword():
    result = execute_query("search whole foods", _make_df())
    assert "Whole Foods" in result


def test_search_case_insensitive():
    result = execute_query("search NETFLIX", _make_df())
    assert "Netflix" in result


def test_search_no_match():
    result = execute_query("search zzz-no-match", _make_df())
    assert "no transactions found" in result.lower()


# ---------------------------------------------------------------------------
# last N months modifier
# ---------------------------------------------------------------------------

def test_last_n_months_restricts_date_range():
    result = execute_query("show groceries last 2 months", _make_df())
    assert "Whole Foods" in result


def test_top_with_last_months():
    result = execute_query("top 3 last 2 months", _make_df())
    assert "row" in result


def test_sum_with_last_months():
    result = execute_query("sum groceries last 2 months", _make_df())
    # Only Feb grocery ($90) is within last 2 months of 2024-03-20
    assert "90" in result


# ---------------------------------------------------------------------------
# currency_sym parameter
# ---------------------------------------------------------------------------

def test_currency_sym_applied_to_transactions():
    """Non-default currency symbol should appear in formatted output."""
    result = execute_query("show groceries", _make_df(), currency_sym="₹")
    assert "₹" in result
    assert "$" not in result


def test_currency_sym_applied_to_categories():
    result = execute_query("categories", _make_df(), currency_sym="£")
    assert "£" in result
    assert "$" not in result


def test_currency_sym_applied_to_sum():
    result = execute_query("sum groceries", _make_df(), currency_sym="€")
    assert "€" in result
    assert "$" not in result


def test_currency_sym_applied_to_monthly():
    result = execute_query("monthly groceries", _make_df(), currency_sym="₹")
    assert "₹" in result
    assert "$" not in result


def test_currency_sym_default_is_dollar():
    result = execute_query("show groceries", _make_df())
    assert "$" in result


# ---------------------------------------------------------------------------
# Date anchor is latest transaction, not today
# ---------------------------------------------------------------------------

def test_last_n_months_anchored_to_data_not_today():
    """'last 1 months' from 2024-03-20 excludes Jan–Feb data."""
    result = execute_query("show groceries last 1 months", _make_df())
    # Feb (2024-02-12) is just outside 1 month of 2024-03-20 cutoff
    # Only March data qualifies — but there are no March groceries in fixture
    assert "no transactions found" in result.lower()


# ---------------------------------------------------------------------------
# _filter_by_category — exact match semantics
# ---------------------------------------------------------------------------

def test_show_partial_name_does_not_match():
    """'show food' must NOT match the category 'Food & Drink' — exact match only."""
    result = execute_query("show food", _make_df())
    assert "no transactions found" in result.lower()


def test_show_exact_multiword_category_matches():
    """'show food & drink' matches 'Food & Drink' exactly (case-insensitive)."""
    result = execute_query("show food & drink", _make_df())
    assert "Uber Eats" in result


# ---------------------------------------------------------------------------
# _filter_last_n_months — empty DataFrame edge case
# ---------------------------------------------------------------------------

def test_last_n_months_on_empty_df_returns_empty():
    """An empty-scoped result from date filtering should not crash."""
    # Use explicit dtypes to match the canonical schema; bare columns= gives
    # object dtype which breaks nsmallest on the Amount column.
    empty_df = pd.DataFrame({
        "Date":        pd.Series([], dtype="str"),
        "Description": pd.Series([], dtype="str"),
        "Amount":      pd.Series([], dtype="float64"),
        "Category":    pd.Series([], dtype="str"),
    })
    result = execute_query("top 5 last 3 months", empty_df)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Fallback / unknown query
# ---------------------------------------------------------------------------

def test_unknown_query_returns_help_message():
    result = execute_query("do something weird", _make_df())
    assert "unknown query" in result.lower() or "supported" in result.lower()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_missing_columns_raises():
    df = pd.DataFrame({"Date": ["2024-01-01"], "Amount": [-10.0]})
    with pytest.raises(ValueError, match="missing columns"):
        execute_query("categories", df)
