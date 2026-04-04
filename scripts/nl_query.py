"""
Module 10 — Natural Language Query Engine
==========================================
Parses plain-English queries against a classified transaction DataFrame.
All processing is local; no external services or API calls are made.

Supported query patterns
------------------------
  categories                   List all categories with total spend
  show <category>              All transactions in a category
  top <N>                      Top N expenses by absolute amount
  top <N> <category>           Top N expenses within a category
  sum <category>               Total spend for a category
  monthly <category>           Month-by-month breakdown for a category
  search <keyword>             Transactions with keyword in Description
  last <N> months              Restrict any of the above to the most
                               recent N calendar months (composable)

Patterns compose — for example:
  "top 5 groceries last 3 months"
  "show transport last 2 months"

Date anchor
-----------
``last <N> months`` is relative to the **latest transaction date in the
DataFrame**, not today's calendar date.  This means "last 3 months" on a
six-month-old statement refers to the last 3 months of *data in that
statement*, not the most recent 3 calendar months from now.
"""

from __future__ import annotations

import re

import pandas as pd


# ---------------------------------------------------------------------------
# Internal filter helpers
# ---------------------------------------------------------------------------

def _filter_by_category(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Return rows whose Category matches *category* (case-insensitive, exact)."""
    return df[df["Category"].str.lower() == category.lower()]


def _filter_last_n_months(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return rows from the most recent *n* calendar months.

    The cutoff is computed relative to the latest transaction date present
    in *df* — not today's date — so results are reproducible regardless of
    when the query is run.
    """
    dates  = pd.to_datetime(df["Date"])
    cutoff = dates.max() - pd.DateOffset(months=n)
    return df[dates > cutoff]


def _top_n_expenses(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return the *n* largest expense rows (most negative amounts)."""
    expenses = df[df["Amount"] < 0].copy()
    return expenses.nsmallest(n, "Amount")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute_query(query: str, df: pd.DataFrame, currency_sym: str = "$") -> str:
    """Parse and execute a natural-language query against *df*.

    Patterns are matched in priority order.  The ``last N months`` modifier
    is applied before any other filter so it composes correctly with ``show``,
    ``top``, ``sum``, and ``monthly``.

    Parameters
    ----------
    query : str
        A plain-English query string (see module docstring for patterns).
    df : pd.DataFrame
        Must have columns: Date, Description, Amount, Category.
    currency_sym : str
        Currency symbol used when formatting amounts (e.g. ``"$"``, ``"₹"``).
        Defaults to ``"$"``.

    Returns
    -------
    str
        Formatted, human-readable result.

    Raises
    ------
    ValueError
        If required columns are missing from *df*.
    """
    required = {"Date", "Description", "Amount", "Category"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"execute_query: DataFrame is missing columns: {sorted(missing)}"
        )

    q = query.strip().lower()

    # ── Intent: categories ────────────────────────────────────────────────
    if re.match(r"^categories?\s*$", q):
        return _fmt_categories(df, currency_sym)

    # ── Intent: sum <category> [last N months] ────────────────────────────
    sum_m = re.match(r"^sum\s+(.+?)(?:\s+last\s+(\d+)\s+months?)?$", q)
    if sum_m:
        cat    = sum_m.group(1).strip()
        months = int(sum_m.group(2)) if sum_m.group(2) else None
        scope  = _filter_last_n_months(df, months) if months else df
        return _fmt_sum(scope, cat, currency_sym)

    # ── Intent: monthly <category> [last N months] ────────────────────────
    mon_m = re.match(r"^monthly\s+(.+?)(?:\s+last\s+(\d+)\s+months?)?$", q)
    if mon_m:
        cat    = mon_m.group(1).strip()
        months = int(mon_m.group(2)) if mon_m.group(2) else None
        scope  = _filter_last_n_months(df, months) if months else df
        return _fmt_monthly(scope, cat, currency_sym)

    # ── Intent: search <keyword> [last N months] ─────────────────────────
    srch_m = re.match(r"^search\s+(.+?)(?:\s+last\s+(\d+)\s+months?)?$", q)
    if srch_m:
        keyword = srch_m.group(1).strip()
        months  = int(srch_m.group(2)) if srch_m.group(2) else None
        scope   = _filter_last_n_months(df, months) if months else df
        return _fmt_search(scope, keyword, currency_sym)

    # ── Composable: apply optional date filter first ─────────────────────
    months_m = re.search(r"last\s+(\d+)\s+months?", q)
    scope    = _filter_last_n_months(df, int(months_m.group(1))) if months_m else df

    # Strip "last N months" fragment so remaining patterns parse cleanly
    q_stripped = re.sub(r"\s*last\s+\d+\s+months?", "", q).strip()

    # ── Intent: top <N> <category> ───────────────────────────────────────
    top_cat_m = re.match(r"^top\s+(\d+)\s+(.+)$", q_stripped)
    if top_cat_m:
        n         = int(top_cat_m.group(1))
        cat       = top_cat_m.group(2).strip()
        cat_scope = _filter_by_category(scope, cat)
        title     = f"Top {n} expenses — {cat.title()}"
        if months_m:
            title += f" (last {months_m.group(1)} months)"
        return _fmt_transactions(_top_n_expenses(cat_scope, n), title, currency_sym)

    # ── Intent: top <N> ───────────────────────────────────────────────────
    top_m = re.match(r"^top\s+(\d+)$", q_stripped)
    if top_m:
        n     = int(top_m.group(1))
        title = f"Top {n} expenses"
        if months_m:
            title += f" (last {months_m.group(1)} months)"
        return _fmt_transactions(_top_n_expenses(scope, n), title, currency_sym)

    # ── Intent: show <category> ───────────────────────────────────────────
    show_m = re.match(r"^show\s+(.+)$", q_stripped)
    if show_m:
        cat       = show_m.group(1).strip()
        cat_scope = _filter_by_category(scope, cat)
        title     = f"Transactions — {cat.title()}"
        if months_m:
            title += f" (last {months_m.group(1)} months)"
        return _fmt_transactions(cat_scope, title, currency_sym)

    # ── Fallback ──────────────────────────────────────────────────────────
    return (
        "Unknown query. Supported: categories | show <cat> | top <N> [<cat>] | "
        "sum <cat> | monthly <cat> | search <kw> [last <N> months]"
    )


# ---------------------------------------------------------------------------
# Formatters  (pure — no side effects)
# ---------------------------------------------------------------------------

def _fmt_transactions(df: pd.DataFrame, title: str, currency_sym: str) -> str:
    """Return a formatted table of transactions."""
    if df.empty:
        return f"{title}\n  (no transactions found)"

    lines = [f"\n{title}  ({len(df)} row{'s' if len(df) != 1 else ''})"]
    lines.append("  " + "─" * 72)

    for _, row in df.iterrows():
        amount = float(row["Amount"])
        sign   = "-" if amount < 0 else "+"
        desc   = str(row["Description"])[:38]
        lines.append(
            f"  {row['Date']}  {desc:<38}  "
            f"{sign}{currency_sym}{abs(amount):>9,.2f}  [{row['Category']}]"
        )
    return "\n".join(lines)


def _fmt_categories(df: pd.DataFrame, currency_sym: str) -> str:
    """Return a summary table of all categories and their total spend."""
    expenses = df[df["Amount"] < 0].copy()
    if expenses.empty:
        return "No expenses found."

    cat_totals = (
        expenses.assign(_abs=expenses["Amount"].abs())
        .groupby("Category")["_abs"]
        .sum()
        .sort_values(ascending=False)
    )

    lines = ["\nCategories — Total Spend"]
    lines.append("  " + "─" * 42)
    for cat, total in cat_totals.items():
        lines.append(f"  {cat:<28}  {currency_sym}{total:>9,.2f}")
    return "\n".join(lines)


def _fmt_sum(df: pd.DataFrame, category: str, currency_sym: str) -> str:
    """Return a one-line total spend for a category."""
    cat_df   = _filter_by_category(df, category)
    expenses = cat_df[cat_df["Amount"] < 0]

    if expenses.empty:
        return f"No expenses found for '{category}'."

    total       = expenses["Amount"].abs().sum()
    count       = len(expenses)
    matched_cat = expenses["Category"].iloc[0]
    return f"\n{matched_cat} — Total: {currency_sym}{total:,.2f} across {count} transaction(s)"


def _fmt_monthly(df: pd.DataFrame, category: str, currency_sym: str) -> str:
    """Return a month-by-month breakdown for a category."""
    cat_df   = _filter_by_category(df, category)
    expenses = cat_df[cat_df["Amount"] < 0].copy()

    if expenses.empty:
        return f"No expenses found for '{category}'."

    expenses["_month"] = pd.to_datetime(expenses["Date"]).dt.to_period("M").astype(str)
    monthly = (
        expenses.groupby("_month")["Amount"]
        .sum()
        .abs()
        .sort_index()
    )

    matched_cat = expenses["Category"].iloc[0]
    lines       = [f"\n{matched_cat} — Monthly Breakdown"]
    lines.append("  " + "─" * 32)
    for month, total in monthly.items():
        lines.append(f"  {month}    {currency_sym}{total:>9,.2f}")
    return "\n".join(lines)


def _fmt_search(df: pd.DataFrame, keyword: str, currency_sym: str) -> str:
    """Return transactions containing *keyword* in their Description."""
    mask    = df["Description"].str.lower().str.contains(re.escape(keyword.lower()), na=False)
    matches = df[mask]
    return _fmt_transactions(matches, f"Search: '{keyword}'", currency_sym)
