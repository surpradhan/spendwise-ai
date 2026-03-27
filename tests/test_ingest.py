"""Tests for scripts/ingest.py"""
import io
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from scripts.ingest import (
    CANONICAL_COLUMNS,
    fuzzy_match_columns,
    ingest,
    mask_card_numbers,
    normalize_amounts,
    normalize_dates,
    remove_duplicates,
    validate_columns,
)


# ---------------------------------------------------------------------------
# validate_columns
# ---------------------------------------------------------------------------

def test_validate_columns_all_present():
    df = pd.DataFrame(columns=["Date", "Description", "Amount"])
    assert validate_columns(df) == []


def test_validate_columns_missing_one():
    df = pd.DataFrame(columns=["Date", "Amount"])
    assert validate_columns(df) == ["Description"]


def test_validate_columns_missing_all():
    df = pd.DataFrame(columns=["foo", "bar"])
    missing = validate_columns(df)
    assert set(missing) == set(CANONICAL_COLUMNS)


# ---------------------------------------------------------------------------
# fuzzy_match_columns
# ---------------------------------------------------------------------------

def test_fuzzy_match_alias_lowercase():
    result = fuzzy_match_columns(
        ["Date", "Description", "Amount"],
        ["post date", "memo", "debit/credit"],
    )
    assert result == {"post date": "Date", "memo": "Description", "debit/credit": "Amount"}


def test_fuzzy_match_alias_mixed_case():
    result = fuzzy_match_columns(
        ["Date", "Description", "Amount"],
        ["Post Date", "Memo", "Debit/Credit"],
    )
    assert result == {"Post Date": "Date", "Memo": "Description", "Debit/Credit": "Amount"}


def test_fuzzy_match_alias_partial_missing():
    result = fuzzy_match_columns(["Description"], ["Date", "Amount", "Narration"])
    assert result == {"Narration": "Description"}


def test_fuzzy_match_difflib_fallback():
    # Pluralised names not in aliases — should match via difflib (score > 0.8)
    result = fuzzy_match_columns(
        ["Date", "Description", "Amount"],
        ["Dates", "Descriptions", "Amounts"],
    )
    assert result == {"Dates": "Date", "Descriptions": "Description", "Amounts": "Amount"}


def test_fuzzy_match_no_confident_match_returns_empty():
    result = fuzzy_match_columns(
        ["Date", "Description", "Amount"],
        ["col_a", "col_b", "col_c"],
    )
    assert result == {}


def test_fuzzy_match_does_not_double_map():
    # Only one available column — should map to Date only, not also Description
    result = fuzzy_match_columns(["Date", "Description"], ["Transaction Date"])
    assert len(result) == 1
    assert result == {"Transaction Date": "Date"}


def test_fuzzy_match_returns_rename_compatible_dict():
    df = pd.DataFrame(columns=["Post Date", "Memo", "Amount"])
    rename_map = fuzzy_match_columns(["Date", "Description"], list(df.columns))
    df2 = df.rename(columns=rename_map)
    assert "Date" in df2.columns
    assert "Description" in df2.columns


def test_fuzzy_match_empty_available():
    assert fuzzy_match_columns(["Date", "Description", "Amount"], []) == {}


def test_fuzzy_match_empty_missing():
    assert fuzzy_match_columns([], ["Post Date", "Memo", "Amount"]) == {}


def test_ingest_auto_maps_aliases(tmp_path):
    csv = tmp_path / "bank.csv"
    csv.write_text("Post Date,Memo,Debit/Credit\n2026-01-15,STARBUCKS,-4.50\n2026-01-16,AMAZON,-32.99\n")
    df = ingest(csv, interactive=False)
    assert list(df.columns) == ["Date", "Description", "Amount"]
    assert len(df) == 2


def test_ingest_raises_on_unresolvable_no_feedback(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("foo,bar,baz\n1,2,3\n")
    with pytest.raises(ValueError, match="Missing required columns"):
        ingest(csv, interactive=False)


def test_fuzzy_match_currency_amount_aliases():
    # Currency-prefixed amount columns (Barclays, HSBC, HDFC style)
    assert fuzzy_match_columns(["Amount"], ["GBP Amount"]) == {"GBP Amount": "Amount"}
    assert fuzzy_match_columns(["Amount"], ["USD Amount"]) == {"USD Amount": "Amount"}
    assert fuzzy_match_columns(["Amount"], ["EUR Amount"]) == {"EUR Amount": "Amount"}
    assert fuzzy_match_columns(["Amount"], ["INR Amount"]) == {"INR Amount": "Amount"}


def test_fuzzy_match_difflib_case_insensitive():
    # Stage 2 should match regardless of case in source column
    result = fuzzy_match_columns(["Date"], ["DATES"])
    assert result == {"DATES": "Date"}


# ---------------------------------------------------------------------------
# remove_duplicates
# ---------------------------------------------------------------------------

def _make_df(rows):
    return pd.DataFrame(rows, columns=["Date", "Description", "Amount"])


def test_remove_duplicates_no_dupes():
    df = _make_df([
        ("2024-01-01", "Coffee", -5.0),
        ("2024-01-02", "Salary", 1000.0),
    ])
    result = remove_duplicates(df)
    assert len(result) == 2


def test_remove_duplicates_removes_exact_dupe():
    df = _make_df([
        ("2024-01-01", "Coffee", -5.0),
        ("2024-01-01", "Coffee", -5.0),
        ("2024-01-02", "Salary", 1000.0),
    ])
    result = remove_duplicates(df)
    assert len(result) == 2


def test_remove_duplicates_different_amount_not_removed():
    df = _make_df([
        ("2024-01-01", "Coffee", -5.0),
        ("2024-01-01", "Coffee", -6.0),
    ])
    result = remove_duplicates(df)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# normalize_dates
# ---------------------------------------------------------------------------

def test_normalize_dates_iso():
    df = _make_df([("2024-03-15", "X", -1.0)])
    result = normalize_dates(df)
    assert result.iloc[0]["Date"] == "2024-03-15"


def test_normalize_dates_slash_format():
    df = _make_df([("03/15/2024", "X", -1.0)])
    result = normalize_dates(df)
    assert result.iloc[0]["Date"] == "2024-03-15"


def test_normalize_dates_invalid_raises():
    df = _make_df([("not-a-date", "X", -1.0)])
    with pytest.raises(ValueError, match="Cannot parse"):
        normalize_dates(df)


def test_normalize_dates_does_not_mutate():
    df = _make_df([("2024-01-01", "X", -1.0)])
    original_val = df.iloc[0]["Date"]
    normalize_dates(df)
    assert df.iloc[0]["Date"] == original_val


# ---------------------------------------------------------------------------
# normalize_amounts
# ---------------------------------------------------------------------------

def test_normalize_amounts_plain_float():
    df = _make_df([("2024-01-01", "X", -12.50)])
    result = normalize_amounts(df)
    assert result.iloc[0]["Amount"] == pytest.approx(-12.50)


def test_normalize_amounts_string_with_currency_symbol():
    df = _make_df([("2024-01-01", "X", "$45.00")])
    result = normalize_amounts(df)
    assert result.iloc[0]["Amount"] == pytest.approx(45.00)


def test_normalize_amounts_cr_suffix():
    df = _make_df([("2024-01-01", "X", "500.00 CR")])
    result = normalize_amounts(df)
    assert result.iloc[0]["Amount"] == pytest.approx(500.00)


def test_normalize_amounts_dr_suffix():
    df = _make_df([("2024-01-01", "X", "75.00 DR")])
    result = normalize_amounts(df)
    assert result.iloc[0]["Amount"] == pytest.approx(-75.00)


def test_normalize_amounts_invalid_raises():
    df = _make_df([("2024-01-01", "X", "NOT_A_NUMBER")])
    with pytest.raises(ValueError, match="Cannot parse amount"):
        normalize_amounts(df)


def test_normalize_amounts_does_not_mutate():
    df = _make_df([("2024-01-01", "X", "$10.00")])
    original = df.iloc[0]["Amount"]
    normalize_amounts(df)
    assert df.iloc[0]["Amount"] == original


# ---------------------------------------------------------------------------
# mask_card_numbers
# ---------------------------------------------------------------------------

def test_mask_card_numbers_16_digit():
    df = pd.DataFrame({
        "Date": ["2024-01-01"],
        "Description": ["Purchase 1234567890123456 end"],
        "Amount": [-10.0],
    })
    result = mask_card_numbers(df)
    assert "****3456" in result.iloc[0]["Description"]
    assert "1234567890123456" not in result.iloc[0]["Description"]


def test_mask_card_numbers_no_card_unchanged():
    df = pd.DataFrame({
        "Date": ["2024-01-01"],
        "Description": ["Starbucks Coffee"],
        "Amount": [-5.0],
    })
    result = mask_card_numbers(df)
    assert result.iloc[0]["Description"] == "Starbucks Coffee"


def test_mask_card_numbers_does_not_mutate():
    df = pd.DataFrame({
        "Date": ["2024-01-01"],
        "Description": ["Purchase 1234567890123456"],
        "Amount": [-10.0],
    })
    original = df.iloc[0]["Description"]
    mask_card_numbers(df)
    assert df.iloc[0]["Description"] == original
