"""Tests for scripts/recurring.py — detect_recurring()."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.recurring import detect_recurring, _classify_frequency


# ---------------------------------------------------------------------------
# _classify_frequency — unit tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("days,expected", [
    (7.0,   "Weekly"),
    (4.0,   "Weekly"),
    (10.0,  "Weekly"),
    (14.0,  "Biweekly"),
    (11.0,  "Biweekly"),
    (18.0,  "Biweekly"),
    (30.0,  "Monthly"),
    (25.0,  "Monthly"),
    (35.0,  "Monthly"),
    (90.0,  "Quarterly"),
    (85.0,  "Quarterly"),
    (95.0,  "Quarterly"),
    (365.0, "Annual"),
    (350.0, "Annual"),
    (380.0, "Annual"),
    (20.0,  "Irregular"),
    (60.0,  "Irregular"),
    (200.0, "Irregular"),
    (0.0,   "Irregular"),
])
def test_classify_frequency(days, expected):
    assert _classify_frequency(days) == expected


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(rows: list[tuple]) -> pd.DataFrame:
    """Build a minimal classified DataFrame from (date, desc, amount, cat) rows."""
    dates, descs, amounts, cats = zip(*rows)
    return pd.DataFrame({
        "Date":        list(dates),
        "Description": list(descs),
        "Amount":      list(amounts),
        "Category":    list(cats),
    })


# ---------------------------------------------------------------------------
# detect_recurring — happy path
# ---------------------------------------------------------------------------

def test_monthly_subscription_detected():
    """Netflix appearing monthly at the same amount is flagged as Monthly."""
    df = _make_df([
        ("2026-01-01", "NETFLIX", -15.49, "Entertainment"),
        ("2026-02-01", "NETFLIX", -15.49, "Entertainment"),
        ("2026-03-01", "NETFLIX", -15.49, "Entertainment"),
    ])
    result = detect_recurring(df)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["Description"] == "NETFLIX"
    assert row["Frequency"] == "Monthly"
    assert row["Occurrences"] == 3
    assert abs(row["Avg_Amount"] - (-15.49)) < 0.01


def test_weekly_charge_detected():
    """A charge every 7 days is classified as Weekly."""
    df = _make_df([
        ("2026-01-07", "GYM WEEKLY", -12.00, "Health"),
        ("2026-01-14", "GYM WEEKLY", -12.00, "Health"),
        ("2026-01-21", "GYM WEEKLY", -12.00, "Health"),
    ])
    result = detect_recurring(df)

    assert len(result) == 1
    assert result.iloc[0]["Frequency"] == "Weekly"


def test_biweekly_charge_detected():
    """A charge every 14 days is classified as Biweekly."""
    df = _make_df([
        ("2026-01-01", "PARKING PERMIT", -40.00, "Transport"),
        ("2026-01-15", "PARKING PERMIT", -40.00, "Transport"),
        ("2026-01-29", "PARKING PERMIT", -40.00, "Transport"),
    ])
    result = detect_recurring(df)

    assert result.iloc[0]["Frequency"] == "Biweekly"


def test_return_columns():
    """Result DataFrame has all required columns in the correct order."""
    df = _make_df([
        ("2026-01-01", "SPOTIFY", -9.99, "Entertainment"),
        ("2026-02-01", "SPOTIFY", -9.99, "Entertainment"),
    ])
    result = detect_recurring(df)

    expected_cols = [
        "Description", "Category", "Avg_Amount", "Median_Days",
        "Frequency", "Occurrences", "First_Date", "Last_Date",
    ]
    assert list(result.columns) == expected_cols


def test_first_and_last_date():
    """First_Date and Last_Date reflect chronological extremes."""
    df = _make_df([
        ("2026-03-01", "RENT", -1500.00, "Housing"),
        ("2026-01-01", "RENT", -1500.00, "Housing"),
        ("2026-02-01", "RENT", -1500.00, "Housing"),
    ])
    result = detect_recurring(df)

    row = result.iloc[0]
    assert row["First_Date"] == "2026-01-01"
    assert row["Last_Date"]  == "2026-03-01"


def test_sorted_by_abs_amount_descending():
    """Rows are sorted largest absolute amount first."""
    df = _make_df([
        ("2026-01-01", "RENT",    -1500.00, "Housing"),
        ("2026-02-01", "RENT",    -1500.00, "Housing"),
        ("2026-01-01", "SPOTIFY",   -9.99,  "Entertainment"),
        ("2026-02-01", "SPOTIFY",   -9.99,  "Entertainment"),
    ])
    result = detect_recurring(df)

    assert result.iloc[0]["Description"] == "RENT"
    assert result.iloc[1]["Description"] == "SPOTIFY"


def test_amount_tolerance_allows_small_variation():
    """Amounts within 10% of the median still qualify as recurring."""
    df = _make_df([
        ("2026-01-01", "ELECTRIC BILL", -98.00,  "Utilities"),
        ("2026-02-01", "ELECTRIC BILL", -105.00, "Utilities"),
        ("2026-03-01", "ELECTRIC BILL", -102.00, "Utilities"),
    ])
    # Median ≈ 102, max deviation ≈ 4%, within 10% tolerance
    result = detect_recurring(df)

    assert len(result) == 1
    assert result.iloc[0]["Description"] == "ELECTRIC BILL"


def test_amount_tolerance_excludes_high_variance():
    """Amounts varying more than tolerance are not flagged as recurring."""
    df = _make_df([
        ("2026-01-01", "AMAZON", -12.99,  "Shopping"),
        ("2026-02-01", "AMAZON", -254.00, "Shopping"),
        ("2026-03-01", "AMAZON", -89.50,  "Shopping"),
    ])
    result = detect_recurring(df)

    assert result.empty


def test_minimum_occurrences_default():
    """A description appearing only once is not recurring."""
    df = _make_df([
        ("2026-01-01", "ONE-OFF", -50.00, "Shopping"),
    ])
    result = detect_recurring(df)

    assert result.empty


def test_minimum_occurrences_custom():
    """min_occurrences=3 requires at least three appearances."""
    df = _make_df([
        ("2026-01-01", "SPOTIFY", -9.99, "Entertainment"),
        ("2026-02-01", "SPOTIFY", -9.99, "Entertainment"),
    ])
    result = detect_recurring(df, min_occurrences=3)

    assert result.empty


def test_multiple_descriptions():
    """Multiple recurring descriptions are all returned."""
    df = _make_df([
        ("2026-01-01", "NETFLIX",  -15.49, "Entertainment"),
        ("2026-02-01", "NETFLIX",  -15.49, "Entertainment"),
        ("2026-01-01", "RENT",    -1500.0, "Housing"),
        ("2026-02-01", "RENT",    -1500.0, "Housing"),
        ("2026-01-15", "ONE-OFF",  -99.99, "Shopping"),
    ])
    result = detect_recurring(df)

    assert set(result["Description"]) == {"NETFLIX", "RENT"}


def test_empty_dataframe_returns_empty():
    """Empty input produces an empty DataFrame with the correct columns."""
    df = pd.DataFrame(columns=["Date", "Description", "Amount", "Category"])
    result = detect_recurring(df)

    assert result.empty
    assert "Description" in result.columns
    assert "Frequency"   in result.columns


def test_no_recurring_returns_empty():
    """DataFrame with only non-recurring transactions returns empty result."""
    df = _make_df([
        ("2026-01-01", "MERCHANT A", -20.00, "Shopping"),
        ("2026-02-01", "MERCHANT B", -30.00, "Shopping"),
        ("2026-03-01", "MERCHANT C", -40.00, "Shopping"),
    ])
    result = detect_recurring(df)

    assert result.empty


def test_duplicate_dates_excluded():
    """Two transactions on the same date with the same description don't
    count as two distinct recurring events."""
    df = _make_df([
        ("2026-01-01", "CHARGE", -10.00, "Other"),
        ("2026-01-01", "CHARGE", -10.00, "Other"),
    ])
    # Only one unique date → does not qualify as recurring (min 2 distinct dates)
    result = detect_recurring(df)

    assert result.empty


def test_category_assigned_from_mode():
    """Category is the most frequent category for that description."""
    df = _make_df([
        ("2026-01-01", "AMAZON PRIME", -14.99, "Shopping"),
        ("2026-02-01", "AMAZON PRIME", -14.99, "Shopping"),
        ("2026-03-01", "AMAZON PRIME", -14.99, "Entertainment"),
    ])
    result = detect_recurring(df)

    assert result.iloc[0]["Category"] == "Shopping"


# ---------------------------------------------------------------------------
# detect_recurring — error handling
# ---------------------------------------------------------------------------

def test_missing_column_raises_value_error():
    """ValueError is raised when required columns are absent."""
    df = pd.DataFrame({
        "Date":   ["2026-01-01", "2026-02-01"],
        "Amount": [-10.0, -10.0],
        # Missing Description and Category
    })
    with pytest.raises(ValueError, match="missing columns"):
        detect_recurring(df)


def test_custom_tolerance_strict():
    """amount_tolerance=0 requires identical amounts."""
    df = _make_df([
        ("2026-01-01", "CHARGE", -10.00, "Other"),
        ("2026-02-01", "CHARGE", -10.01, "Other"),  # 0.1% deviation
    ])
    result = detect_recurring(df, amount_tolerance=0.0)

    assert result.empty


def test_custom_tolerance_strict_exact_match():
    """amount_tolerance=0 accepts truly identical amounts."""
    df = _make_df([
        ("2026-01-01", "CHARGE", -10.00, "Other"),
        ("2026-02-01", "CHARGE", -10.00, "Other"),
    ])
    result = detect_recurring(df, amount_tolerance=0.0)

    assert len(result) == 1
