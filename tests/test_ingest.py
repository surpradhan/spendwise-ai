"""Tests for scripts/ingest.py"""
import io
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from scripts.ingest import (
    CANONICAL_COLUMNS,
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
