"""
Module 5 — Recurring Transaction Detector
==========================================
Identifies likely recurring charges from a classified transaction DataFrame.

A transaction group (keyed on Description) is considered recurring when:
- The description appears at least *min_occurrences* times.
- All individual amounts lie within *amount_tolerance* of the group's
  median amount (catches minor price changes, e.g. subscription tax).
- Transactions span at least two distinct dates (deduplication guard).

Frequency classification is based on the median gap between consecutive
transaction dates:

  4–10 days   → Weekly
  11–18 days  → Biweekly
  25–35 days  → Monthly
  85–95 days  → Quarterly
  350–380 days→ Annual
  otherwise   → Irregular
"""

from __future__ import annotations

import pandas as pd


# (inclusive_low, inclusive_high), label
_FREQ_BANDS: list[tuple[tuple[float, float], str]] = [
    ((4.0,   10.0),  "Weekly"),
    ((11.0,  18.0),  "Biweekly"),
    ((25.0,  35.0),  "Monthly"),
    ((85.0,  95.0),  "Quarterly"),
    ((350.0, 380.0), "Annual"),
]

_RETURN_COLS: list[str] = [
    "Description", "Category", "Avg_Amount", "Median_Days",
    "Frequency", "Occurrences", "First_Date", "Last_Date",
]


def _classify_frequency(median_days: float) -> str:
    """Map a median inter-transaction gap (days) to a frequency label.

    Parameters
    ----------
    median_days : float

    Returns
    -------
    str
        One of: Weekly, Biweekly, Monthly, Quarterly, Annual, Irregular.
    """
    for (lo, hi), label in _FREQ_BANDS:
        if lo <= median_days <= hi:
            return label
    return "Irregular"


def detect_recurring(
    df: pd.DataFrame,
    min_occurrences: int = 2,
    amount_tolerance: float = 0.10,
) -> pd.DataFrame:
    """Detect likely recurring transactions from a classified DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: Date, Description, Amount, Category.
        Typically the output of ``classify_all``.
    min_occurrences : int
        Minimum number of appearances of the same description to qualify
        as recurring.  Default: 2.
    amount_tolerance : float
        Maximum allowed deviation from the median amount, expressed as a
        fraction of the median (e.g. 0.10 = ±10%).  Groups where any
        transaction exceeds this threshold are excluded — they likely
        represent irregular purchases rather than a fixed recurring charge.
        Default: 0.10.

    Returns
    -------
    pd.DataFrame
        Columns: Description, Category, Avg_Amount, Median_Days, Frequency,
        Occurrences, First_Date, Last_Date.
        Sorted by absolute Avg_Amount descending (largest charges first).
        Returns an empty DataFrame with the same columns when nothing qualifies.

    Raises
    ------
    ValueError
        If any of the required columns are missing from *df*.
    """
    required = {"Date", "Description", "Amount", "Category"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"detect_recurring: DataFrame is missing columns: {sorted(missing)}"
        )

    work = df.copy()
    work["_date_parsed"] = pd.to_datetime(work["Date"])

    rows: list[dict] = []

    for description, group in work.groupby("Description", sort=False):
        if len(group) < min_occurrences:
            continue

        amounts     = group["Amount"]
        median_amt  = float(amounts.median())

        # Amount-consistency check — skip groups with high variance.
        # Guard against zero median (e.g. refunds that net to zero).
        if median_amt != 0.0:
            deviations = (amounts - median_amt).abs() / abs(median_amt)
            if (deviations > amount_tolerance).any():
                continue

        sorted_dates = (
            group["_date_parsed"].sort_values().reset_index(drop=True)
        )
        # Must span at least two distinct calendar days (deduplication guard).
        if sorted_dates.nunique() < 2:
            continue

        gaps = sorted_dates.diff().dropna().dt.days
        median_gap = float(gaps.median()) if len(gaps) > 0 else 0.0

        rows.append({
            "Description": description,
            "Category":    group["Category"].mode().iloc[0],
            "Avg_Amount":  round(float(amounts.mean()), 2),
            "Median_Days": round(median_gap, 1),
            "Frequency":   _classify_frequency(median_gap),
            "Occurrences": len(group),
            "First_Date":  sorted_dates.iloc[0].strftime("%Y-%m-%d"),
            "Last_Date":   sorted_dates.iloc[-1].strftime("%Y-%m-%d"),
        })

    if not rows:
        return pd.DataFrame(columns=_RETURN_COLS)

    result = pd.DataFrame(rows, columns=_RETURN_COLS)
    # Sort: largest absolute amount first (most impactful recurring charges).
    result = (
        result
        .assign(_sort_key=result["Avg_Amount"].abs())
        .sort_values("_sort_key", ascending=False)
        .drop(columns="_sort_key")
        .reset_index(drop=True)
    )
    return result
