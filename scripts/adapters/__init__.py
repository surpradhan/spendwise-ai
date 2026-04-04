"""
Bank Adapter Registry
======================
Provides :func:`detect_adapter` — the single entry point for selecting the
appropriate bank adapter for a raw DataFrame.

Adapter resolution order:

1. If ``bank_hint`` is provided, look up ``_BANK_HINTS`` (derived
   automatically from adapters that declare a ``hint_key`` class attribute)
   and return that adapter directly (raises ``ValueError`` for unknown hints).
2. Otherwise, iterate ``_REGISTRY`` in order and return the first adapter
   whose ``detect()`` classmethod returns ``True``.

:class:`~scripts.adapters.generic.GenericAdapter` is always the last entry
in ``_REGISTRY`` because its ``detect()`` always returns ``True``.

**To add a new bank adapter (single place to edit):**
1. Create ``scripts/adapters/<bank>.py`` with a class that inherits from
   :class:`~scripts.adapters.base.BankAdapter` and declares a lowercase
   ``hint_key: str`` class attribute (e.g. ``hint_key = "chase"``).
2. Import it here and insert it into ``_REGISTRY`` *before* ``GenericAdapter``.
   The hint key is picked up automatically — no other changes needed.
"""

from __future__ import annotations

import pandas as pd

from scripts.adapters.base import BankAdapter
from scripts.adapters.hdfc import HDFCAdapter
from scripts.adapters.generic import GenericAdapter

# Ordered registry: specific adapters first, GenericAdapter (catch-all) last.
# GenericAdapter MUST remain the final entry — its detect() always returns True.
_REGISTRY: list[type[BankAdapter]] = [
    HDFCAdapter,
    GenericAdapter,
]

# Derived automatically from _REGISTRY: maps hint_key → adapter class.
# Adapters without a hint_key (i.e. GenericAdapter) are excluded.
_BANK_HINTS: dict[str, type[BankAdapter]] = {
    cls.hint_key: cls
    for cls in _REGISTRY
    if hasattr(cls, "hint_key")
}


def detect_adapter(
    df_raw: pd.DataFrame,
    bank_hint: str | None = None,
) -> BankAdapter:
    """Select and instantiate the appropriate adapter for *df_raw*.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw DataFrame as returned by :func:`scripts.ingest.load_file`.
    bank_hint : str | None
        Optional explicit bank name from the ``--bank`` CLI flag
        (e.g. ``"hdfc"``).  When provided, skips auto-detection and uses
        the named adapter directly.  Case-insensitive.

    Returns
    -------
    BankAdapter
        Instantiated adapter ready to call ``.normalize()``.

    Raises
    ------
    ValueError
        If *bank_hint* is provided but not recognised.
    """
    if bank_hint is not None:
        key = bank_hint.strip().lower()
        adapter_cls = _BANK_HINTS.get(key)
        if adapter_cls is None:
            known = ", ".join(sorted(_BANK_HINTS))
            raise ValueError(
                f"Unknown bank hint '{bank_hint}'. Known values: {known}. "
                "Omit --bank to use auto-detection (GenericAdapter is the fallback)."
            )
        print(f"  ↳ Bank adapter: {adapter_cls.name} (forced via --bank)")
        return adapter_cls()

    for adapter_cls in _REGISTRY:
        if adapter_cls.detect(df_raw):
            if adapter_cls is not GenericAdapter:
                print(f"  ↳ Bank adapter: {adapter_cls.name} (auto-detected)")
            return adapter_cls()

    # Should never reach here — GenericAdapter.detect always returns True.
    return GenericAdapter()
