"""
Module 6 — Budget Targets & Alerts
====================================
Loads, saves, and evaluates monthly budget targets against actual spending
derived from the :func:`scripts.terminal_output.build_summary` dict.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------

def load_budgets(path: str | Path) -> dict[str, float]:
    """Load budget targets from a JSON file.

    Returns an empty dict (not an error) if the file does not exist.

    Parameters
    ----------
    path : str | Path
        Path to the budgets JSON file (e.g. ``config/budgets.json``).

    Returns
    -------
    dict[str, float]
        Mapping of category name → monthly budget amount (positive float).

    Raises
    ------
    ValueError
        If any budget value is not a positive number (zero or negative are
        also invalid).
    """
    path = Path(path)
    if not path.exists():
        return {}

    with path.open(encoding="utf-8") as fh:
        raw: dict = json.load(fh)

    budgets: dict[str, float] = {}
    for category, value in raw.items():
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Budget value for '{category}' must be a positive number, "
                f"got {value!r}. Edit '{path}' to fix."
            )
        fvalue = float(value)
        if fvalue <= 0:
            raise ValueError(
                f"Budget value for '{category}' must be a positive number "
                f"greater than zero, got {fvalue}. Edit '{path}' to fix."
            )
        budgets[category] = fvalue

    return budgets


def save_budgets(budgets: dict[str, float], path: str | Path) -> None:
    """Persist budget targets to a JSON file with sorted keys.

    Mirrors the :func:`scripts.classifier.save_keywords` pattern: keys are
    sorted alphabetically before writing so diffs are deterministic.

    Parameters
    ----------
    budgets : dict[str, float]
        Mapping of category name → monthly budget amount.
    path : str | Path
        Destination path for the JSON file.  Parent directories are created
        if they do not exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_budgets = {k: budgets[k] for k in sorted(budgets)}
    with path.open("w", encoding="utf-8") as fh:
        json.dump(sorted_budgets, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate_budgets(
    summary: dict,
    budgets: dict[str, float],
    warn_threshold: float = 0.80,
) -> list[dict]:
    """Compare actual monthly average spending against budget targets.

    The number of months covered is derived from ``summary['period_start']``
    and ``summary['period_end']`` by counting distinct ``YYYY-MM`` periods in
    that inclusive date range.

    Parameters
    ----------
    summary : dict
        As returned by :func:`scripts.terminal_output.build_summary`.
        Must contain ``period_start``, ``period_end``, and
        ``category_totals``.
    budgets : dict[str, float]
        Mapping of category name → monthly budget (positive float).
        Categories absent from this mapping are silently skipped.
    warn_threshold : float, optional
        Fraction of the budget (0–1) at which status becomes
        ``"APPROACHING"`` (default: ``0.80``).

    Returns
    -------
    list[dict]
        One entry per budgeted category, sorted by ``pct_used`` descending.
        Each dict contains:

        * ``category``    – str
        * ``budget``      – float  (monthly target)
        * ``monthly_avg`` – float  (actual monthly average spend)
        * ``pct_used``    – float  (monthly_avg / budget * 100)
        * ``status``      – str    ("EXCEEDED" | "APPROACHING" | "OK")
        * ``num_months``  – int
    """
    if not budgets:
        return []

    # Compute the number of distinct YYYY-MM periods covered
    start = pd.Period(summary["period_start"][:7], freq="M")
    end   = pd.Period(summary["period_end"][:7],   freq="M")
    num_months = max(1, (end - start).n + 1)

    category_totals: dict[str, float] = summary.get("category_totals", {})

    results: list[dict] = []
    for category, budget in budgets.items():
        total_spend = category_totals.get(category, 0.0)
        # category_totals stores absolute (positive) expense magnitudes
        monthly_avg = abs(total_spend) / num_months
        pct_used = (monthly_avg / budget) * 100

        match True:
            case _ if pct_used >= 100:
                status = "EXCEEDED"
            case _ if pct_used >= warn_threshold * 100:
                status = "APPROACHING"
            case _:
                status = "OK"

        results.append({
            "category":    category,
            "budget":      budget,
            "monthly_avg": round(monthly_avg, 2),
            "pct_used":    round(pct_used, 2),
            "status":      status,
            "num_months":  num_months,
        })

    results.sort(key=lambda d: d["pct_used"], reverse=True)
    return results
