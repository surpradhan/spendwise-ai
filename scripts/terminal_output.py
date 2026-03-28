"""
Module 3 — Terminal Output
===========================
Produces a human-readable structured summary in the terminal and optionally
a machine-readable JSON payload for piping into other scripts or AI agents.
"""

from __future__ import annotations

import json
import re

import pandas as pd


# ---------------------------------------------------------------------------
# 1. Build summary dict
# ---------------------------------------------------------------------------

def build_summary(df: pd.DataFrame) -> dict:
    """Compute the full spending summary from a categorised DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: Date, Description, Amount, Category.

    Returns
    -------
    dict with keys:
        period_start        : str  (YYYY-MM-DD)
        period_end          : str  (YYYY-MM-DD)
        total_income        : float
        total_expenses      : float
        net                 : float
        category_totals     : dict[str, float]   (expenses only, sorted desc)
        top_merchants       : list[dict]          (top 10 by cumulative spend)
        uncategorized_count : int
        uncategorized_sample: list[dict]
    """
    dates = pd.to_datetime(df["Date"])
    period_start = dates.min().strftime("%Y-%m-%d")
    period_end   = dates.max().strftime("%Y-%m-%d")

    income_mask   = df["Amount"] > 0
    expense_mask  = df["Amount"] < 0

    total_income   = float(df.loc[income_mask,  "Amount"].sum())
    total_expenses = float(df.loc[expense_mask, "Amount"].sum())  # negative
    net            = total_income + total_expenses

    # Category totals (expenses only, positive magnitudes)
    expense_df = df[expense_mask].copy()
    expense_df["AbsAmount"] = expense_df["Amount"].abs()
    cat_totals_series = (
        expense_df.groupby("Category")["AbsAmount"]
        .sum()
        .sort_values(ascending=False)
    )
    category_totals: dict[str, float] = {
        cat: round(float(val), 2)
        for cat, val in cat_totals_series.items()
    }

    # Top 10 merchants by cumulative spend
    merchant_totals = (
        expense_df.groupby("Description")["AbsAmount"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    top_merchants = [
        {"merchant": desc, "total": round(float(amt), 2)}
        for desc, amt in merchant_totals.items()
    ]

    # Uncategorized
    uncat_df = df[df["Category"] == "Uncategorized"]
    uncategorized_count = len(uncat_df)
    uncategorized_sample = uncat_df.head(5)[
        ["Date", "Description", "Amount"]
    ].to_dict(orient="records")

    return {
        "period_start":         period_start,
        "period_end":           period_end,
        "total_income":         round(total_income, 2),
        "total_expenses":       round(abs(total_expenses), 2),
        "net":                  round(net, 2),
        "category_totals":      category_totals,
        "top_merchants":        top_merchants,
        "uncategorized_count":  uncategorized_count,
        "uncategorized_sample": uncategorized_sample,
    }


# ---------------------------------------------------------------------------
# 2. Terminal printer
# ---------------------------------------------------------------------------

def _bar(fraction: float, width: int = 20) -> str:
    """Return a unicode block bar proportional to *fraction* (0..1)."""
    filled = round(fraction * width)
    return "█" * filled + "░" * (width - filled)


def _mask_residual(text: str) -> str:
    """Mask any residual card-number sequences before printing."""
    pattern = re.compile(r"\b(\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{0,4}|\d{12,16})\b")
    def replacer(m: re.Match) -> str:
        digits = re.sub(r"[\s\-]", "", m.group(0))
        return f"****{digits[-4:]}" if 12 <= len(digits) <= 16 else m.group(0)
    return pattern.sub(replacer, text)


def print_summary(summary: dict) -> None:
    """Format and print the spending summary to stdout.

    Parameters
    ----------
    summary : dict
        As returned by :func:`build_summary`.
    """
    W = 54
    sep  = "═" * W
    thin = "─" * W

    print(f"\n{sep}")
    print(" SpendWise AI — Spending Summary")
    print(f" Period: {summary['period_start']} to {summary['period_end']}")
    print(sep)

    income   = summary["total_income"]
    expenses = summary["total_expenses"]
    net      = summary["net"]

    print(f" {'Total Income':<20}: ${income:>10,.2f}")
    print(f" {'Total Expenses':<20}: ${expenses:>10,.2f}")
    print(f" {'Net':<20}: ${net:>10,.2f}")

    print(f"\n {'Spending by Category'}")
    print(f" {thin}")

    cat_totals = summary["category_totals"]
    total_expense = expenses or 1  # avoid division by zero
    col_w = max((len(c) for c in cat_totals), default=12)

    for cat, amount in cat_totals.items():
        pct = (amount / total_expense) * 100
        bar = _bar(amount / total_expense, width=16)
        print(f"  {cat:<{col_w}}  ${amount:>9,.2f}   {bar}  {pct:5.1f}%")

    print(f"\n {'Top 10 Merchants'}")
    print(f" {thin}")
    for entry in summary["top_merchants"]:
        merchant = _mask_residual(entry["merchant"])[:40]
        print(f"  {merchant:<40}  ${entry['total']:>9,.2f}")

    uncat = summary["uncategorized_count"]
    if uncat:
        print(f"\n  ⚠  Uncategorized: {uncat} transaction(s) flagged.")
        print("  Run again after reviewing to re-classify them.")

    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# 3. Recurring transaction printer
# ---------------------------------------------------------------------------

def print_recurring(recurring_df: pd.DataFrame) -> None:
    """Print a recurring-transaction summary to stdout.

    Parameters
    ----------
    recurring_df : pd.DataFrame
        As returned by :func:`scripts.recurring.detect_recurring`.
        Expected columns: Description, Category, Avg_Amount, Frequency,
        Occurrences, Last_Date.
    """
    if recurring_df.empty:
        return

    W    = 54
    thin = "─" * W

    print(f"\n {'Recurring Transactions Detected'}")
    print(f" {thin}")

    desc_w = min(
        max((len(str(r)) for r in recurring_df["Description"]), default=12),
        36,
    )

    for _, row in recurring_df.iterrows():
        desc   = _mask_residual(str(row["Description"]))[:desc_w]
        amount = abs(float(row["Avg_Amount"]))
        freq   = row["Frequency"]
        occ    = int(row["Occurrences"])
        last   = row["Last_Date"]
        print(
            f"  {desc:<{desc_w}}  ${amount:>8,.2f}/cycle"
            f"  {freq:<10}  {occ}× (last: {last})"
        )

    print()


# ---------------------------------------------------------------------------
# 4. JSON serialisation
# ---------------------------------------------------------------------------

def to_json(summary: dict) -> str:
    """Serialise the summary dict to a pretty-printed JSON string.

    Parameters
    ----------
    summary : dict

    Returns
    -------
    str
        Indented JSON.
    """
    return json.dumps(summary, indent=2, ensure_ascii=False)
