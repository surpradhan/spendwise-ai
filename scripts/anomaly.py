"""
Module 9 — Anomaly Detector
============================
Identifies statistically unusual transactions using per-category
modified z-score analysis (median ± MAD).

A transaction is flagged as anomalous when its modified z-score exceeds
*z_threshold*.  Using the median and Median Absolute Deviation (MAD)
instead of mean and std makes the detector robust to small samples where
a single outlier would otherwise inflate the mean and shrink its own
ordinary z-score.

Modified z-score formula (Iglewicz & Hoaglin, 1993):
    z_i = 0.6745 * (x_i - median) / MAD

Categories with fewer than three transactions fall back to the global
expense distribution so singleton merchants are still evaluated.

Degenerate MAD (more than half the values in a group are identical):
    Falls back to (x_i - median) / std, preserving the median anchor
    so the sensitivity stays consistent with the MAD path.

Only expense rows (Amount < 0) are analysed — income spikes are not
flagged.
"""

from __future__ import annotations

import pandas as pd


# Consistency constant: under a normal distribution, MAD ≈ 0.6745 * σ,
# so this factor aligns the modified z-score scale with ordinary z-scores.
_SCALE: float = 0.6745

# Columns present in every returned DataFrame
_RETURN_COLS: list[str] = [
    "Date", "Description", "Amount", "Category", "Z_Score",
]


def _mad_z_scores(values: pd.Series) -> pd.Series:
    """Return modified z-scores for *values* using median + MAD.

    Falls back to ``(x - median) / std`` when MAD is zero, which occurs
    when more than half the values are identical.  The median anchor is
    preserved in the fallback so sensitivity stays consistent with the
    MAD path.

    Parameters
    ----------
    values : pd.Series
        Numeric series (absolute expense amounts).

    Returns
    -------
    pd.Series
        Modified z-scores, same index as *values*.
    """
    med = values.median()
    mad = (values - med).abs().median()
    if mad != 0 and not pd.isna(mad):
        return _SCALE * (values - med) / mad
    # Degenerate MAD — fall back to std, keeping median as the centre
    std = values.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=values.index)
    return (values - med) / std


def detect_anomalies(
    df: pd.DataFrame,
    z_threshold: float = 3.5,
) -> pd.DataFrame:
    """Identify statistically unusual expense transactions.

    For each spending category with at least three transactions, computes
    the per-category modified z-score of each transaction amount.
    Categories with fewer than three transactions are evaluated against
    the global expense distribution.  Transactions whose z-score exceeds
    *z_threshold* are returned.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: Date, Description, Amount, Category.
        Typically the output of ``classify_all_v2``.
    z_threshold : float
        Modified z-score threshold above which a transaction is flagged.
        Iglewicz & Hoaglin recommend 3.5 for general use; lower values
        are more aggressive.  Default: 3.5.

    Returns
    -------
    pd.DataFrame
        Columns: Date, Description, Amount, Category, Z_Score.
        Sorted by Z_Score descending (most extreme first).
        Returns an empty DataFrame with the same columns when no anomalies
        are detected.

    Raises
    ------
    ValueError
        If any required column is missing from *df*.
    """
    required = {"Date", "Description", "Amount", "Category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"detect_anomalies: DataFrame is missing columns: {sorted(missing)}"
        )

    expenses = df[df["Amount"] < 0].copy()
    expenses["_abs"] = expenses["Amount"].abs()

    if expenses.empty:
        return pd.DataFrame(columns=_RETURN_COLS)

    global_z = _mad_z_scores(expenses["_abs"])

    anomaly_rows: list[dict] = []

    for category, group in expenses.groupby("Category"):
        if len(group) >= 3:
            z_scores = _mad_z_scores(group["_abs"])
        else:
            # Singleton / doubleton — use global distribution
            z_scores = global_z.loc[group.index]

        flagged = group[z_scores > z_threshold]

        for idx in flagged.index:
            row = expenses.loc[idx]
            anomaly_rows.append({
                "Date":        row["Date"],
                "Description": row["Description"],
                "Amount":      row["Amount"],
                "Category":    row["Category"],
                "Z_Score":     round(float(z_scores.loc[idx]), 2),
            })

    if not anomaly_rows:
        return pd.DataFrame(columns=_RETURN_COLS)

    result = (
        pd.DataFrame(anomaly_rows, columns=_RETURN_COLS)
        .sort_values("Z_Score", ascending=False)
        .reset_index(drop=True)
    )
    return result
