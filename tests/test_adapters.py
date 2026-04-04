"""Tests for scripts/adapters/ — detection, normalisation, and registry."""
import pandas as pd
import pytest

from scripts.adapters import detect_adapter
from scripts.adapters.hdfc import HDFCAdapter
from scripts.adapters.generic import GenericAdapter, _infer_currency


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hdfc_raw() -> pd.DataFrame:
    """Minimal raw HDFC DataFrame matching the real export format."""
    return pd.DataFrame({
        "Date":             ["01/01/24", "15/06/24"],
        "Narration":        ["ATM WITHDRAWAL", "SALARY CREDIT"],
        "Chq./Ref.No.":     ["123456", ""],
        "Value Dt":         ["01/01/24", "15/06/24"],
        "Withdrawal Amt.":  [500.0, 0.0],
        "Deposit Amt.":     [0.0, 50000.0],
        "Closing Balance":  [9500.0, 59500.0],
    })


def _make_generic_raw() -> pd.DataFrame:
    """Minimal generic (non-HDFC) DataFrame."""
    return pd.DataFrame({
        "Date":        ["2024-01-15"],
        "Description": ["STARBUCKS"],
        "Amount":      [-4.50],
    })


# ---------------------------------------------------------------------------
# HDFCAdapter.detect
# ---------------------------------------------------------------------------

def test_hdfc_detect_true_with_correct_columns():
    assert HDFCAdapter.detect(_make_hdfc_raw()) is True


def test_hdfc_detect_false_without_narration():
    df = _make_hdfc_raw().drop(columns=["Narration"])
    assert HDFCAdapter.detect(df) is False


def test_hdfc_detect_false_without_withdrawal():
    df = _make_hdfc_raw().drop(columns=["Withdrawal Amt."])
    assert HDFCAdapter.detect(df) is False


def test_hdfc_detect_false_without_deposit():
    df = _make_hdfc_raw().drop(columns=["Deposit Amt."])
    assert HDFCAdapter.detect(df) is False


def test_hdfc_detect_handles_whitespace_in_column_names():
    df = _make_hdfc_raw().rename(columns={
        "Narration":        "  Narration  ",
        "Withdrawal Amt.":  " Withdrawal Amt. ",
        "Deposit Amt.":     " Deposit Amt. ",
    })
    assert HDFCAdapter.detect(df) is True


# ---------------------------------------------------------------------------
# HDFCAdapter.normalize — canonical columns
# ---------------------------------------------------------------------------

def test_hdfc_normalize_produces_canonical_columns():
    result = HDFCAdapter().normalize(_make_hdfc_raw())
    assert set(result.columns) == {"Date", "Description", "Amount", "Currency"}


def test_hdfc_normalize_row_count_preserved():
    result = HDFCAdapter().normalize(_make_hdfc_raw())
    assert len(result) == 2


# ---------------------------------------------------------------------------
# HDFCAdapter.normalize — Date
# ---------------------------------------------------------------------------

def test_hdfc_normalize_date_format_ddmmyy():
    result = HDFCAdapter().normalize(_make_hdfc_raw())
    assert result.iloc[0]["Date"] == "2024-01-01"


def test_hdfc_normalize_date_format_ddmmyyyy():
    df = _make_hdfc_raw()
    df.loc[0, "Date"] = "01/01/2024"
    result = HDFCAdapter().normalize(df)
    assert result.iloc[0]["Date"] == "2024-01-01"


def test_hdfc_normalize_invalid_date_raises():
    df = _make_hdfc_raw()
    df.loc[0, "Date"] = "not-a-date"
    with pytest.raises(ValueError, match="cannot parse"):
        HDFCAdapter().normalize(df)


# ---------------------------------------------------------------------------
# HDFCAdapter.normalize — Amount
# ---------------------------------------------------------------------------

def test_hdfc_normalize_withdrawal_is_negative():
    result = HDFCAdapter().normalize(_make_hdfc_raw())
    assert result.iloc[0]["Amount"] == pytest.approx(-500.0)


def test_hdfc_normalize_deposit_is_positive():
    result = HDFCAdapter().normalize(_make_hdfc_raw())
    assert result.iloc[1]["Amount"] == pytest.approx(50000.0)


def test_hdfc_normalize_amount_with_comma_separator():
    df = _make_hdfc_raw()
    # Cast to object before assigning a string to avoid pandas FutureWarning
    # about setting incompatible dtype on a float64 column.
    df["Withdrawal Amt."] = df["Withdrawal Amt."].astype(object)
    df.loc[0, "Withdrawal Amt."] = "1,500.00"
    df.loc[0, "Deposit Amt."]    = 0.0
    result = HDFCAdapter().normalize(df)
    assert result.iloc[0]["Amount"] == pytest.approx(-1500.0)


def test_hdfc_normalize_empty_withdrawal_treated_as_zero():
    df = _make_hdfc_raw()
    # Cast to object before assigning an empty string to avoid pandas FutureWarning.
    df["Withdrawal Amt."] = df["Withdrawal Amt."].astype(object)
    df.loc[0, "Withdrawal Amt."] = ""
    df.loc[0, "Deposit Amt."]    = 200.0
    result = HDFCAdapter().normalize(df)
    assert result.iloc[0]["Amount"] == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# HDFCAdapter.normalize — Description
# ---------------------------------------------------------------------------

def test_hdfc_normalize_description_uses_narration():
    result = HDFCAdapter().normalize(_make_hdfc_raw())
    assert "ATM WITHDRAWAL" in result.iloc[0]["Description"]


def test_hdfc_normalize_description_appends_ref_when_present():
    result = HDFCAdapter().normalize(_make_hdfc_raw())
    # Row 0 has Chq./Ref.No. = "123456"
    assert "Ref: 123456" in result.iloc[0]["Description"]


def test_hdfc_normalize_description_no_ref_when_empty():
    result = HDFCAdapter().normalize(_make_hdfc_raw())
    # Row 1 has empty ref
    assert "Ref:" not in result.iloc[1]["Description"]


# ---------------------------------------------------------------------------
# HDFCAdapter.normalize — Currency
# ---------------------------------------------------------------------------

def test_hdfc_normalize_currency_is_inr_by_default():
    result = HDFCAdapter().normalize(_make_hdfc_raw())
    assert (result["Currency"] == "INR").all()


def test_hdfc_normalize_currency_override():
    result = HDFCAdapter().normalize(_make_hdfc_raw(), currency_override="USD")
    assert (result["Currency"] == "USD").all()


def test_hdfc_normalize_currency_override_uppercased():
    result = HDFCAdapter().normalize(_make_hdfc_raw(), currency_override="usd")
    assert (result["Currency"] == "USD").all()


# ---------------------------------------------------------------------------
# HDFCAdapter.normalize — immutability
# ---------------------------------------------------------------------------

def test_hdfc_normalize_does_not_mutate_input():
    df = _make_hdfc_raw()
    original_cols = list(df.columns)
    HDFCAdapter().normalize(df)
    assert list(df.columns) == original_cols


# ---------------------------------------------------------------------------
# GenericAdapter.detect
# ---------------------------------------------------------------------------

def test_generic_detect_always_true():
    assert GenericAdapter.detect(pd.DataFrame({"whatever": [1]})) is True


def test_generic_detect_true_for_empty_df():
    assert GenericAdapter.detect(pd.DataFrame()) is True


# ---------------------------------------------------------------------------
# _infer_currency
# ---------------------------------------------------------------------------

def test_infer_currency_uses_override():
    df = pd.DataFrame({"Amount": [1.0]})
    assert _infer_currency(df, "GBP") == "GBP"


def test_infer_currency_override_uppercased():
    df = pd.DataFrame({"Amount": [1.0]})
    assert _infer_currency(df, "gbp") == "GBP"


def test_infer_currency_from_gbp_amount_column():
    df = pd.DataFrame({"GBP Amount": [1.0]})
    assert _infer_currency(df, None) == "GBP"


def test_infer_currency_from_eur_amount_column():
    df = pd.DataFrame({"EUR Amount": [1.0]})
    assert _infer_currency(df, None) == "EUR"


def test_infer_currency_from_inr_amount_column():
    df = pd.DataFrame({"INR Amount": [1.0]})
    assert _infer_currency(df, None) == "INR"


def test_infer_currency_from_aud_amount_column_case_insensitive():
    df = pd.DataFrame({"aud amount": [1.0]})
    assert _infer_currency(df, None) == "AUD"


def test_infer_currency_default_usd_when_no_hint():
    df = pd.DataFrame({"Amount": [1.0]})
    assert _infer_currency(df, None) == "USD"


def test_infer_currency_override_takes_precedence_over_column():
    df = pd.DataFrame({"GBP Amount": [1.0]})
    assert _infer_currency(df, "EUR") == "EUR"


# ---------------------------------------------------------------------------
# GenericAdapter.normalize
# ---------------------------------------------------------------------------

def test_generic_normalize_adds_currency_column():
    result = GenericAdapter().normalize(_make_generic_raw())
    assert "Currency" in result.columns


def test_generic_normalize_default_currency_usd():
    result = GenericAdapter().normalize(_make_generic_raw())
    assert result.iloc[0]["Currency"] == "USD"


def test_generic_normalize_currency_override():
    result = GenericAdapter().normalize(_make_generic_raw(), currency_override="GBP")
    assert result.iloc[0]["Currency"] == "GBP"


def test_generic_normalize_preserves_other_columns():
    result = GenericAdapter().normalize(_make_generic_raw())
    assert "Date" in result.columns
    assert "Description" in result.columns
    assert "Amount" in result.columns


def test_generic_normalize_does_not_mutate_input():
    df = _make_generic_raw()
    original_cols = list(df.columns)
    GenericAdapter().normalize(df)
    assert list(df.columns) == original_cols


# ---------------------------------------------------------------------------
# detect_adapter registry
# ---------------------------------------------------------------------------

def test_detect_adapter_picks_hdfc_for_hdfc_columns():
    adapter = detect_adapter(_make_hdfc_raw())
    assert isinstance(adapter, HDFCAdapter)


def test_detect_adapter_picks_generic_for_generic_columns():
    adapter = detect_adapter(_make_generic_raw())
    assert isinstance(adapter, GenericAdapter)


def test_detect_adapter_bank_hint_forces_hdfc():
    # Even a generic-looking DataFrame is routed to HDFC when hint is given
    adapter = detect_adapter(_make_generic_raw(), bank_hint="hdfc")
    assert isinstance(adapter, HDFCAdapter)


def test_detect_adapter_bank_hint_case_insensitive():
    adapter = detect_adapter(_make_generic_raw(), bank_hint="HDFC")
    assert isinstance(adapter, HDFCAdapter)


def test_detect_adapter_unknown_hint_raises():
    with pytest.raises(ValueError, match="Unknown bank hint"):
        detect_adapter(_make_generic_raw(), bank_hint="foobank")


def test_detect_adapter_unknown_hint_lists_known_values():
    with pytest.raises(ValueError, match="hdfc"):
        detect_adapter(_make_generic_raw(), bank_hint="barclays")
