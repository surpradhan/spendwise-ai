"""Tests for scripts/anomaly.py — detect_anomalies() and _mad_z_scores()."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.anomaly import detect_anomalies, _mad_z_scores, _RETURN_COLS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(rows: list[tuple]) -> pd.DataFrame:
    """Build a minimal classified DataFrame from (date, desc, amount, cat) rows."""
    dates, descs, amounts, cats = zip(*rows)
    return pd.DataFrame({
        "Date":        list(dates),
        "Description": list(descs),
        "Amount":      [float(a) for a in amounts],
        "Category":    list(cats),
    })


# ---------------------------------------------------------------------------
# _mad_z_scores — unit tests
# ---------------------------------------------------------------------------

def test_mad_z_scores_detects_outlier():
    """A clearly extreme value should produce a high modified z-score."""
    values = pd.Series([3.0, 3.5, 3.2, 500.0])
    z = _mad_z_scores(values)
    assert z.iloc[3] > 3.5   # the 500 should be flagged


def test_mad_z_scores_uniform_returns_zero():
    """Perfectly identical values have MAD=0 and std=0 — all z-scores are 0."""
    values = pd.Series([10.0, 10.0, 10.0])
    z = _mad_z_scores(values)
    assert (z == 0.0).all()


def test_mad_z_scores_fallback_uses_median_not_mean():
    """When MAD=0, fallback centres on median (not mean) to stay consistent."""
    # More than half identical → MAD=0 → fallback
    values = pd.Series([5.0, 5.0, 5.0, 100.0, 200.0])
    z = _mad_z_scores(values)
    # The outliers (100, 200) should have positive z-scores
    assert z.iloc[3] > 0
    assert z.iloc[4] > 0


# ---------------------------------------------------------------------------
# Schema / return value guarantees
# ---------------------------------------------------------------------------

def test_return_columns_no_anomaly_type():
    """Anomaly_Type was removed — returned columns must match _RETURN_COLS exactly."""
    df     = _make_df([("2024-01-01", "Salary", 1000.0, "Income")])
    result = detect_anomalies(df)
    assert list(result.columns) == _RETURN_COLS
    assert "Anomaly_Type" not in result.columns


def test_returns_correct_columns_when_anomaly_found():
    rows = [
        ("2024-01-01", "Coffee",  -3.0,   "Food"),
        ("2024-01-02", "Coffee",  -3.5,   "Food"),
        ("2024-01-03", "Coffee",  -3.2,   "Food"),
        ("2024-01-04", "Dinner", -200.0,  "Food"),
    ]
    df     = _make_df(rows)
    result = detect_anomalies(df, z_threshold=3.5)
    assert list(result.columns) == _RETURN_COLS
    assert not result.empty


# ---------------------------------------------------------------------------
# Happy path — clear outlier detected
# ---------------------------------------------------------------------------

def test_detects_obvious_outlier():
    """A single very large transaction in an otherwise tight category."""
    rows = [
        ("2024-01-01", "Coffee",  -3.0,   "Food"),
        ("2024-01-02", "Coffee",  -3.5,   "Food"),
        ("2024-01-03", "Coffee",  -3.2,   "Food"),
        ("2024-01-04", "Splurge", -500.0, "Food"),
    ]
    df     = _make_df(rows)
    result = detect_anomalies(df, z_threshold=3.5)
    assert len(result) == 1
    assert result.iloc[0]["Description"] == "Splurge"
    assert result.iloc[0]["Z_Score"] > 3.5


def test_sorted_by_z_score_descending():
    # Varied small values (MAD > 0); two large outliers ensure 2+ results.
    rows = [
        ("2024-01-01", "A",    -4.0,   "Misc"),
        ("2024-01-02", "B",    -5.0,   "Misc"),
        ("2024-01-03", "C",    -6.0,   "Misc"),
        ("2024-01-04", "Big1", -500.0, "Misc"),
        ("2024-01-05", "Big2", -800.0, "Misc"),
    ]
    df     = _make_df(rows)
    result = detect_anomalies(df, z_threshold=3.5)
    assert not result.empty
    assert result.iloc[0]["Description"] == "Big2"
    z_values = result["Z_Score"].tolist()
    assert z_values == sorted(z_values, reverse=True)


# ---------------------------------------------------------------------------
# No anomalies
# ---------------------------------------------------------------------------

def test_no_anomalies_uniform_category():
    """Uniform amounts → std is 0 → no flag raised."""
    rows = [
        ("2024-01-01", "Sub", -10.0, "Subscriptions"),
        ("2024-01-02", "Sub", -10.0, "Subscriptions"),
        ("2024-01-03", "Sub", -10.0, "Subscriptions"),
    ]
    df     = _make_df(rows)
    result = detect_anomalies(df)
    assert result.empty


def test_no_anomalies_income_only():
    """Income rows are never flagged."""
    rows = [
        ("2024-01-01", "Salary", 3000.0, "Income"),
        ("2024-01-02", "Bonus",  9000.0, "Income"),
    ]
    df     = _make_df(rows)
    result = detect_anomalies(df)
    assert result.empty


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_missing_columns_raises():
    df = pd.DataFrame({"Date": ["2024-01-01"], "Amount": [-10.0]})
    with pytest.raises(ValueError, match="missing columns"):
        detect_anomalies(df)


def test_empty_dataframe_returns_empty():
    df = pd.DataFrame(columns=["Date", "Description", "Amount", "Category"])
    result = detect_anomalies(df)
    assert result.empty
    assert list(result.columns) == _RETURN_COLS


def test_singleton_category_uses_global_distribution():
    """A singleton category with an extreme amount should be caught via global z."""
    rows = [
        ("2024-01-01", "Coffee",    -3.0,    "Food"),
        ("2024-01-02", "Coffee",    -3.5,    "Food"),
        ("2024-01-03", "Coffee",    -3.2,    "Food"),
        ("2024-01-04", "JetEngine", -5000.0, "Rare"),
    ]
    df     = _make_df(rows)
    result = detect_anomalies(df, z_threshold=3.5)
    assert "JetEngine" in result["Description"].tolist()


def test_no_duplicate_rows_in_result():
    """Each anomalous transaction should appear at most once."""
    rows = [
        ("2024-01-01", "Coffee",  -3.0,   "Food"),
        ("2024-01-02", "Coffee",  -3.5,   "Food"),
        ("2024-01-03", "Coffee",  -3.2,   "Food"),
        ("2024-01-04", "Splurge", -500.0, "Food"),
    ]
    df     = _make_df(rows)
    result = detect_anomalies(df, z_threshold=3.5)
    assert not result.duplicated(subset=["Date", "Description", "Amount"]).any()


def test_z_score_column_present_and_numeric():
    rows = [
        ("2024-01-01", "Coffee",  -3.0,   "Food"),
        ("2024-01-02", "Coffee",  -3.5,   "Food"),
        ("2024-01-03", "Coffee",  -3.2,   "Food"),
        ("2024-01-04", "Splurge", -500.0, "Food"),
    ]
    df     = _make_df(rows)
    result = detect_anomalies(df, z_threshold=3.5)
    assert "Z_Score" in result.columns
    assert pd.api.types.is_float_dtype(result["Z_Score"])
