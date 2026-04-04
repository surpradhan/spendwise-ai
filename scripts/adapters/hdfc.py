"""
HDFC Bank Adapter
=================
Normalises HDFC Bank statement CSV exports into the SpendWise AI canonical
schema.

HDFC export columns (typical):
    Date | Narration | Chq./Ref.No. | Value Dt | Withdrawal Amt. |
    Deposit Amt. | Closing Balance

Detection fingerprint: presence of all three columns
    "Narration", "Withdrawal Amt.", "Deposit Amt."
"""

from __future__ import annotations

import pandas as pd

from scripts.adapters.base import BankAdapter

_HDFC_REQUIRED: frozenset[str] = frozenset(
    {"Narration", "Withdrawal Amt.", "Deposit Amt."}
)


class HDFCAdapter(BankAdapter):
    """Adapter for HDFC Bank statement exports (CSV format).

    Detection fingerprint: all three columns ``"Narration"``,
    ``"Withdrawal Amt."``, and ``"Deposit Amt."`` present in the raw
    DataFrame (after stripping leading/trailing whitespace from column names).

    Normalization pipeline (no in-place mutation):

    1. Strip whitespace from column names.
    2. Build ``Description`` from ``Narration``; append
       ``" | Ref: {ref}"`` when ``Chq./Ref.No.`` is non-empty.
    3. Build signed ``Amount``: ``Deposit Amt. - Withdrawal Amt.``
       (deposits positive, withdrawals negative).  Empty strings → 0.0.
    4. Parse ``Date`` with ``dayfirst=True`` (HDFC uses DD/MM/YY format).
       Raises ``ValueError`` on unparseable dates.
    5. Set ``Currency`` to ``currency_override`` or ``"INR"``.
    6. Return only ``{Date, Description, Amount, Currency}``.
    """

    name = "HDFC"
    #: Lowercase key used with the ``--bank`` CLI flag (e.g. ``--bank hdfc``).
    hint_key = "hdfc"

    @staticmethod
    def _to_float(series: pd.Series) -> pd.Series:
        """Parse an HDFC amount column: strip commas, coerce empty/nan → 0.0."""
        return (
            series.astype(str)
            .str.strip()
            .str.replace(",", "", regex=False)
            .replace({"": "0", "nan": "0"})
            .astype(float)
        )

    @classmethod
    def detect(cls, df_raw: pd.DataFrame) -> bool:
        """Return True when HDFC-specific columns are all present.

        Comparison is performed after stripping whitespace from column names.

        Parameters
        ----------
        df_raw : pd.DataFrame

        Returns
        -------
        bool
        """
        cols = {c.strip() for c in df_raw.columns}
        return _HDFC_REQUIRED.issubset(cols)

    def normalize(
        self,
        df_raw: pd.DataFrame,
        currency_override: str | None = None,
    ) -> pd.DataFrame:
        """Normalise an HDFC Bank raw export to the canonical schema.

        Parameters
        ----------
        df_raw : pd.DataFrame
            Raw HDFC statement DataFrame as returned by
            :func:`scripts.ingest.load_file`.
        currency_override : str | None
            When provided, overrides the default currency ``"INR"``.

        Returns
        -------
        pd.DataFrame
            Columns: ``Date``, ``Description``, ``Amount``, ``Currency``.

        Raises
        ------
        ValueError
            If any ``Date`` value cannot be parsed.
        """
        # 1. Strip whitespace from column names (work on a copy)
        df = df_raw.rename(columns=lambda c: c.strip())

        # 2. Build Description
        narration = df["Narration"].astype(str).str.strip()
        if "Chq./Ref.No." in df.columns:
            ref = df["Chq./Ref.No."].astype(str).str.strip()
            # Append ref only when it's non-empty and not a bare "nan"
            has_ref = ref.str.len() > 0
            has_ref &= ref.str.lower() != "nan"
            description = narration.where(~has_ref, narration + " | Ref: " + ref)
        else:
            description = narration

        # 3. Build signed Amount (deposit = +, withdrawal = -)
        deposit    = self._to_float(df["Deposit Amt."])
        withdrawal = self._to_float(df["Withdrawal Amt."])
        amount     = deposit - withdrawal

        # 4. Parse Date (DD/MM/YY or DD/MM/YYYY → YYYY-MM-DD)
        # Use format="mixed" (pandas ≥ 2.0) to handle both YY and YYYY variants
        # in the same column without pandas inferring a single fixed format.
        parsed_dates = pd.to_datetime(
            df["Date"], dayfirst=True, errors="coerce", format="mixed"
        )
        bad = df.loc[parsed_dates.isna(), "Date"]
        if not bad.empty:
            samples = bad.head(5).tolist()
            raise ValueError(
                f"HDFC adapter: cannot parse {len(bad)} date value(s). "
                f"Examples: {samples}\n"
                "Please fix these rows in the source file before re-running."
            )
        date_str = parsed_dates.dt.strftime("%Y-%m-%d")

        # 5. Currency
        currency = (currency_override.upper() if currency_override else "INR")

        # 6. Return canonical columns only
        return pd.DataFrame(
            {
                "Date":        date_str.values,
                "Description": description.values,
                "Amount":      amount.values,
                "Currency":    currency,
            }
        )
