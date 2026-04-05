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
        Optionally contains a ``Currency`` column (ISO-4217 code).

    Returns
    -------
    dict with keys:
        period_start        : str  (YYYY-MM-DD)
        period_end          : str  (YYYY-MM-DD)
        total_income        : float  (aggregate across all currencies)
        total_expenses      : float  (aggregate across all currencies)
        net                 : float
        category_totals     : dict[str, float]   (expenses only, sorted desc)
        top_merchants       : list[dict]          (top 10 by cumulative spend)
        uncategorized_count : int
        uncategorized_sample: list[dict]
        currencies          : list[str]           (sorted distinct ISO codes)
        currency_totals     : dict[str, dict]     (only when >1 currency;
                              keys: income, expenses, net per currency)
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

    # Currency tracking — always produce at least one entry so consumers can
    # safely index currencies[0] without guarding against an empty list.
    _raw_currencies = (
        sorted(df["Currency"].dropna().unique().tolist())
        if "Currency" in df.columns
        else []
    )
    currencies = _raw_currencies if _raw_currencies else ["USD"]

    summary: dict = {
        "period_start":         period_start,
        "period_end":           period_end,
        "total_income":         round(total_income, 2),
        "total_expenses":       round(abs(total_expenses), 2),
        "net":                  round(net, 2),
        "category_totals":      category_totals,
        "top_merchants":        top_merchants,
        "uncategorized_count":  uncategorized_count,
        "uncategorized_sample": uncategorized_sample,
        "currencies":           currencies,
    }

    # Per-currency breakdown — only when multiple currencies present
    if len(currencies) > 1:
        per_currency: dict[str, dict] = {}
        for cur in currencies:
            cur_df = df[df["Currency"] == cur]
            cur_income   = float(cur_df.loc[cur_df["Amount"] > 0, "Amount"].sum())
            cur_expenses = float(cur_df.loc[cur_df["Amount"] < 0, "Amount"].sum())
            per_currency[cur] = {
                "income":   round(cur_income, 2),
                "expenses": round(abs(cur_expenses), 2),
                "net":      round(cur_income + cur_expenses, 2),
            }
        summary["currency_totals"] = per_currency

    return summary


# ---------------------------------------------------------------------------
# 2. Terminal printer
# ---------------------------------------------------------------------------

_CURRENCY_SYMBOLS: dict[str, str] = {
    "USD": "$",
    "GBP": "£",
    "EUR": "€",
    "INR": "₹",
    "CAD": "CA$",
    "AUD": "A$",
}


def currency_label(code: str) -> str:
    """Return a display symbol for a currency code, or the code itself.

    Parameters
    ----------
    code : str
        ISO-4217 currency code (e.g. ``"USD"``, ``"INR"``).

    Returns
    -------
    str
        Symbol (e.g. ``"$"``, ``"₹"``) or ``"CODE "`` for unknowns.
    """
    return _CURRENCY_SYMBOLS.get(code.upper(), f"{code.upper()} ")


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

    currencies = summary.get("currencies", ["USD"])
    sym = currency_label(currencies[0])

    print(f"\n{sep}")
    print(" SpendWise AI — Spending Summary")
    print(f" Period: {summary['period_start']} to {summary['period_end']}")
    print(sep)

    income   = summary["total_income"]
    expenses = summary["total_expenses"]
    net      = summary["net"]

    # Multi-currency notice
    if len(currencies) > 1:
        print(f"  ⚠  Multi-currency statement: {', '.join(currencies)}")
        print("  Totals below are aggregated across all currencies.")
        print(f" {thin}")

    print(f" {'Total Income':<20}: {sym}{income:>10,.2f}")
    print(f" {'Total Expenses':<20}: {sym}{expenses:>10,.2f}")
    print(f" {'Net':<20}: {sym}{net:>10,.2f}")

    # Per-currency breakdown (multi-currency only)
    if "currency_totals" in summary:
        print(f"\n {'Per-Currency Breakdown'}")
        print(f" {thin}")
        for cur, totals in summary["currency_totals"].items():
            csym = currency_label(cur)
            print(f"  {cur}  Income: {csym}{totals['income']:,.2f}  "
                  f"Expenses: {csym}{totals['expenses']:,.2f}  "
                  f"Net: {csym}{totals['net']:,.2f}")

    cat_label = "(all currencies combined)" if len(currencies) > 1 else ""
    print(f"\n {'Spending by Category'} {cat_label}")
    print(f" {thin}")

    cat_totals = summary["category_totals"]
    total_expense = expenses or 1  # avoid division by zero
    col_w = max((len(c) for c in cat_totals), default=12)

    for cat, amount in cat_totals.items():
        pct = (amount / total_expense) * 100
        bar = _bar(amount / total_expense, width=16)
        print(f"  {cat:<{col_w}}  {sym}{amount:>9,.2f}   {bar}  {pct:5.1f}%")

    print(f"\n {'Top 10 Merchants'}")
    print(f" {thin}")
    for entry in summary["top_merchants"]:
        merchant = _mask_residual(entry["merchant"])[:40]
        print(f"  {merchant:<40}  {sym}{entry['total']:>9,.2f}")

    uncat = summary["uncategorized_count"]
    if uncat:
        print(f"\n  ⚠  Uncategorized: {uncat} transaction(s) flagged.")
        print("  Run again after reviewing to re-classify them.")

    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# 3. Recurring transaction printer
# ---------------------------------------------------------------------------

def print_recurring(recurring_df: pd.DataFrame, currency_sym: str = "$") -> None:
    """Print a recurring-transaction summary to stdout.

    Parameters
    ----------
    recurring_df : pd.DataFrame
        As returned by :func:`scripts.recurring.detect_recurring`.
        Expected columns: Description, Category, Avg_Amount, Frequency,
        Occurrences, Last_Date.
    currency_sym : str
        Currency symbol to prefix amounts (e.g. ``"$"``, ``"₹"``).
        Defaults to ``"$"`` for backward compatibility.
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
            f"  {desc:<{desc_w}}  {currency_sym}{amount:>8,.2f}/cycle"
            f"  {freq:<10}  {occ}× (last: {last})"
        )

    print()


# ---------------------------------------------------------------------------
# 4. Budget alerts printer
# ---------------------------------------------------------------------------

def print_budget_alerts(alerts: list[dict], currency_sym: str = "$") -> None:
    """Print monthly budget alert summary to stdout.

    No-op when *alerts* is empty.

    Parameters
    ----------
    alerts : list[dict]
        As returned by :func:`scripts.budget.evaluate_budgets`.
        Each dict must have keys: category, budget, monthly_avg, pct_used,
        status, num_months.
    currency_sym : str
        Currency symbol to prefix amounts (e.g. ``"$"``, ``"₹"``).
        Defaults to ``"$"`` for backward compatibility.
    """
    if not alerts:
        return

    W    = 54
    sep  = "═" * W
    thin = "─" * W

    print(f"\n{sep}")
    print(" SpendWise AI — Budget Alerts")
    print(sep)

    col_w = max((len(a["category"]) for a in alerts), default=12)

    for alert in alerts:
        cat        = alert["category"]
        monthly    = alert["monthly_avg"]
        budget     = alert["budget"]
        pct        = alert["pct_used"]
        status     = alert["status"]
        bar_frac   = min(pct / 100, 1.0)
        bar        = _bar(bar_frac, width=16)

        match status:
            case "EXCEEDED":
                symbol = "  ✗"
            case "APPROACHING":
                symbol = "  ⚠"
            case _:
                symbol = ""

        print(
            f"  {cat:<{col_w}}  {currency_sym}{monthly:>8,.2f} / {currency_sym}{budget:>8,.2f}"
            f"   {bar}  {pct:5.1f}%{symbol}"
        )

    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# 5. Anomaly report printer
# ---------------------------------------------------------------------------

def print_anomaly_report(anomaly_df: pd.DataFrame, currency_sym: str = "$") -> None:
    """Print a flagged-anomaly summary to stdout.

    No-op when *anomaly_df* is empty.

    Parameters
    ----------
    anomaly_df : pd.DataFrame
        As returned by :func:`scripts.anomaly.detect_anomalies`.
        Expected columns: Date, Description, Amount, Category,
        Anomaly_Type, Z_Score.
    currency_sym : str
        Currency symbol to prefix amounts.  Defaults to ``"$"``.
    """
    if anomaly_df.empty:
        return

    W   = 54
    sep = "═" * W

    print(f"\n{sep}")
    print(" SpendWise AI — Anomaly Report")
    print(sep)
    print(f"  {len(anomaly_df)} unusual transaction(s) detected (z-score > threshold)")
    print(f" {'─' * W}")

    desc_w = min(
        max((len(str(r)) for r in anomaly_df["Description"]), default=12),
        36,
    )

    for _, row in anomaly_df.iterrows():
        desc   = _mask_residual(str(row["Description"]))[:desc_w]
        amount = abs(float(row["Amount"]))
        z      = float(row["Z_Score"])
        cat    = row["Category"]
        print(
            f"  {desc:<{desc_w}}  {currency_sym}{amount:>9,.2f}"
            f"  z={z:.2f}  [{cat}]"
        )

    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# 6. Month-over-month comparison
# ---------------------------------------------------------------------------

def build_mom_comparison(df: pd.DataFrame) -> dict:
    """Compute month-over-month spend change per category.

    Compares the two most recent calendar months present in *df*.  Only
    expense rows (Amount < 0) are included.  Categories that appear in
    only one of the two months are included with the missing month as 0.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: Date, Amount, Category.

    Returns
    -------
    dict with keys:
        current_month  : str          (YYYY-MM, the more recent month)
        previous_month : str          (YYYY-MM, the month before)
        changes        : list[dict]   sorted by abs pct_change descending
            Each entry has: category, current, previous, diff, pct_change.
            pct_change is None when previous spend is 0 (new category).
    """
    expenses = df[df["Amount"] < 0].copy()
    expenses["_month"] = pd.to_datetime(expenses["Date"]).dt.to_period("M").astype(str)

    months = sorted(expenses["_month"].unique())
    if len(months) < 2:
        return {"current_month": months[-1] if months else "", "previous_month": "", "changes": []}

    current_month  = months[-1]
    previous_month = months[-2]

    cur_totals  = (
        expenses[expenses["_month"] == current_month]
        .groupby("Category")["Amount"].sum().abs()
    )
    prev_totals = (
        expenses[expenses["_month"] == previous_month]
        .groupby("Category")["Amount"].sum().abs()
    )

    all_cats = sorted(set(cur_totals.index) | set(prev_totals.index))
    changes = []
    for cat in all_cats:
        cur  = float(cur_totals.get(cat, 0.0))
        prev = float(prev_totals.get(cat, 0.0))
        diff = cur - prev
        pct  = ((diff / prev) * 100) if prev != 0 else None
        changes.append({
            "category":   cat,
            "current":    round(cur, 2),
            "previous":   round(prev, 2),
            "diff":       round(diff, 2),
            "pct_change": round(pct, 1) if pct is not None else None,
        })

    # Sort by absolute pct change descending; new-category rows (pct=None) go last
    changes.sort(key=lambda x: abs(x["pct_change"]) if x["pct_change"] is not None else -1, reverse=True)

    return {
        "current_month":  current_month,
        "previous_month": previous_month,
        "changes":        changes,
    }


def print_mom_comparison(mom: dict, currency_sym: str = "$") -> None:
    """Print a month-over-month category comparison to stdout.

    No-op when fewer than two months of data are available.

    Parameters
    ----------
    mom : dict
        As returned by :func:`build_mom_comparison`.
    currency_sym : str
        Currency symbol to prefix amounts.  Defaults to ``"$"``.
    """
    if not mom.get("previous_month") or not mom.get("changes"):
        return

    W    = 54
    sep  = "═" * W
    thin = "─" * W

    cur  = mom["current_month"]
    prev = mom["previous_month"]

    print(f"\n{sep}")
    print(f" Month-over-Month: {prev}  →  {cur}")
    print(sep)

    col_w = max((len(c["category"]) for c in mom["changes"]), default=12)

    for entry in mom["changes"]:
        cat  = entry["category"]
        cur_amt  = entry["current"]
        prev_amt = entry["previous"]
        diff     = entry["diff"]
        pct      = entry["pct_change"]

        if pct is None:
            arrow = "  NEW"
            pct_str = "   n/a"
        elif diff > 0:
            arrow = "  ↑"
            pct_str = f"+{pct:5.1f}%"
        elif diff < 0:
            arrow = "  ↓"
            pct_str = f"{pct:5.1f}%"
        else:
            arrow = "   ="
            pct_str = "  0.0%"

        print(
            f"  {cat:<{col_w}}  {currency_sym}{prev_amt:>8,.2f} → "
            f"{currency_sym}{cur_amt:>8,.2f}  {pct_str}{arrow}"
        )

    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# 7. JSON serialisation
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
