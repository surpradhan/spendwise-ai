"""
Generic Bank Adapter (fallback)
================================
Wraps the existing SpendWise AI ingest pipeline as a thin adapter.  When no
bank-specific adapter matches, the Generic adapter is used.

Responsibilities:
* Infer the currency from the ``--currency`` flag, from a currency-prefixed
  column name (e.g. ``"GBP Amount"`` → ``"GBP"``), or default to ``"USD"``.
* Append a ``Currency`` column to the raw DataFrame and return it unchanged.
* Leave all column renaming, validation, and normalization to the downstream
  ingest pipeline steps (``fuzzy_match_columns``, ``prompt_column_mapping``,
  ``normalize_dates``, etc.).
"""

from __future__ import annotations

import re

import pandas as pd

from scripts.adapters.base import BankAdapter

# Matches amount column names that embed a currency code, e.g. "GBP Amount"
_CURRENCY_COL_RE = re.compile(
    r"^(gbp|usd|eur|inr|cad|aud)\s+amount$", re.IGNORECASE
)


def _infer_currency(df_raw: pd.DataFrame, override: str | None) -> str:
    """Determine the ISO-4217 currency code for a generic import.

    Inference order:

    1. ``override`` (from ``--currency`` CLI flag)
    2. Currency prefix in an amount column name
       (e.g. ``"GBP Amount"`` → ``"GBP"``)
    3. Default: ``"USD"``

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw DataFrame before column renaming.
    override : str | None
        Explicit currency code from the ``--currency`` flag.

    Returns
    -------
    str
        Three-letter ISO-4217 currency code, uppercased.
    """
    if override:
        return override.upper()

    for col in df_raw.columns:
        match = _CURRENCY_COL_RE.match(col.strip())
        if match:
            return match.group(1).upper()

    return "USD"


class GenericAdapter(BankAdapter):
    """Fallback adapter wrapping the existing fuzzy-match ingest logic.

    ``detect()`` always returns ``True`` — this adapter is the last entry in
    the registry and acts as a catch-all.

    ``normalize()`` only appends a ``Currency`` column and returns a copy of
    the raw DataFrame.  All column renaming, validation, deduplication, date
    normalisation, amount normalisation, and card masking are performed by the
    downstream ingest pipeline.

    Currency inference order:
    1. ``currency_override`` argument (from ``--currency`` CLI flag)
    2. Currency prefix detected in an amount column name
       (e.g. ``"EUR Amount"`` → ``"EUR"``)
    3. Default: ``"USD"``
    """

    name = "Generic"

    @classmethod
    def detect(cls, df_raw: pd.DataFrame) -> bool:
        """Always returns ``True`` — Generic is the fallback of last resort.

        Parameters
        ----------
        df_raw : pd.DataFrame

        Returns
        -------
        bool
        """
        return True

    def normalize(
        self,
        df_raw: pd.DataFrame,
        currency_override: str | None = None,
    ) -> pd.DataFrame:
        """Append a ``Currency`` column and return a copy of *df_raw*.

        Does **not** rename, validate, or transform any other columns —
        that is left to the downstream ingest pipeline steps.

        Parameters
        ----------
        df_raw : pd.DataFrame
        currency_override : str | None
            Explicit currency code from the ``--currency`` flag.

        Returns
        -------
        pd.DataFrame
            Copy of *df_raw* with a ``Currency`` column added.
        """
        currency = _infer_currency(df_raw, currency_override)
        result = df_raw.copy()
        result["Currency"] = currency
        return result
