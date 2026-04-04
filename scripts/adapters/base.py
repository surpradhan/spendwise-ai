"""
Bank Adapter Base Class
========================
Defines the abstract interface that all bank-format adapters must implement.
"""

from __future__ import annotations

import abc

import pandas as pd


class BankAdapter(abc.ABC):
    """Abstract base class for all bank-format adapters.

    Subclasses implement :meth:`detect` (class-level heuristic) and
    :meth:`normalize` (raw-to-canonical transformation).

    The ``detect`` classmethod allows the registry to probe adapters without
    instantiation.  ``normalize`` is an instance method so subclasses can be
    extended with configuration if needed in the future.
    """

    #: Human-readable adapter name, e.g. ``"HDFC"`` or ``"Generic"``.
    name: str

    @classmethod
    @abc.abstractmethod
    def detect(cls, df_raw: pd.DataFrame) -> bool:
        """Return True if *df_raw* matches this bank's column fingerprint.

        Parameters
        ----------
        df_raw : pd.DataFrame
            Raw, unprocessed DataFrame as returned by
            :func:`scripts.ingest.load_file`.

        Returns
        -------
        bool
        """

    @abc.abstractmethod
    def normalize(
        self,
        df_raw: pd.DataFrame,
        currency_override: str | None = None,
    ) -> pd.DataFrame:
        """Transform *df_raw* into (or toward) the canonical schema.

        Must return a **new** DataFrame — no in-place mutation.  The result
        must contain at minimum:

        * ``Date``        — str, ``YYYY-MM-DD``
        * ``Description`` — str, card numbers masked
        * ``Amount``      — float64, expenses negative / income positive
        * ``Currency``    — str, ISO-4217 code (e.g. ``"INR"``, ``"USD"``)

        Concrete adapters (HDFC, etc.) produce the full canonical set.
        The :class:`GenericAdapter` only adds ``Currency`` and leaves
        column renaming/validation to the downstream ingest pipeline.

        Parameters
        ----------
        df_raw : pd.DataFrame
        currency_override : str | None
            When provided, overrides the adapter's default currency
            detection/default.

        Returns
        -------
        pd.DataFrame
        """
