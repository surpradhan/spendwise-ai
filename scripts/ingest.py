"""
Module 1 — Input Ingestion & Normalisation
==========================================
Responsible for loading raw bank exports, normalising them into a consistent
schema, and flagging / removing data-quality issues before downstream processing.

Canonical schema:
    Date        : str  (YYYY-MM-DD)
    Description : str
    Amount      : float  (expenses negative, income positive)
"""

from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

import chardet
import pandas as pd


# ---------------------------------------------------------------------------
# 1. File loading
# ---------------------------------------------------------------------------

def load_file(path: str | Path) -> pd.DataFrame:
    """Load a CSV or XLSX file and return a raw DataFrame.

    Parameters
    ----------
    path : str | Path
        Path to the input file.

    Returns
    -------
    pd.DataFrame
        Raw, unprocessed rows from the file.

    Raises
    ------
    ValueError
        If the file extension is not .csv or .xlsx/.xls.
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        encoding = _detect_encoding(path)
        try:
            df = pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, engine="openpyxl")
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Expected .csv, .xlsx, or .xls."
        )

    print(f"  ✓ Loaded {len(df):,} rows from '{path.name}'")
    return df


# ---------------------------------------------------------------------------
# 2. Column validation & mapping
# ---------------------------------------------------------------------------

CANONICAL_COLUMNS = ("Date", "Description", "Amount")

_COLUMN_ALIASES: dict[str, list[str]] = {
    "Date": [
        "date",
        "post date",
        "posted date",
        "posting date",
        "transaction date",
        "trans date",
        "trans. date",
        "value date",
        "settlement date",
        "effective date",
        "booking date",
        "trade date",
    ],
    "Description": [
        "description",
        "memo",
        "narrative",
        "details",
        "transaction description",
        "transaction details",
        "merchant",
        "merchant name",
        "payee",
        "payee name",
        "reference",
        "remarks",
        "particulars",
        "narration",
        "note",
        "notes",
    ],
    "Amount": [
        "amount",
        "debit/credit",
        "debit / credit",
        "credit/debit",
        "transaction amount",
        "net amount",
        "value",
        "payment",
        "withdrawal/deposit",
        "deposit/withdrawal",
        "dr/cr",
        "cr/dr",
        "gbp amount",
        "usd amount",
        "eur amount",
        "inr amount",
        "cad amount",
        "aud amount",
    ],
}


def validate_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of canonical columns missing from *df*.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    list[str]
        Missing canonical column names.  Empty list → all present.
    """
    return [col for col in CANONICAL_COLUMNS if col not in df.columns]


def fuzzy_match_columns(
    missing: list[str],
    available: list[str],
) -> dict[str, str]:
    """Auto-map missing canonical column names to available DataFrame columns.

    Uses a two-stage strategy:

    1. **Alias lookup** — checks *available* against a curated list of known
       bank-export synonyms (``_COLUMN_ALIASES``).  Comparison is
       case-insensitive and whitespace-stripped.
    2. **difflib fallback** — if no alias matches, uses
       ``difflib.get_close_matches`` with a cutoff of 0.8 to find a
       high-confidence fuzzy match against the remaining unmapped columns.

    Only unambiguous mappings are returned.  If a canonical column cannot be
    confidently resolved, it is omitted so the caller can escalate to
    interactive prompting.

    Parameters
    ----------
    missing : list[str]
        Canonical column names absent from the DataFrame
        (e.g. ``["Date", "Description"]``).
    available : list[str]
        Column names actually present in the DataFrame.

    Returns
    -------
    dict[str, str]
        ``{available_col: canonical_col}`` for every confident auto-mapping.
        Suitable for passing directly to ``DataFrame.rename(columns=...)``.
    """
    rename_map: dict[str, str] = {}
    claimed: set[str] = set()
    available_lower = {col.strip().lower(): col for col in available}

    for canonical in missing:
        # Stage 1: alias lookup
        matched: str | None = None
        for alias in _COLUMN_ALIASES.get(canonical, []):
            original = available_lower.get(alias)
            if original is not None and original not in claimed:
                matched = original
                break

        if matched is None:
            # Stage 2: difflib fallback (case-insensitive, consistent with Stage 1)
            unclaimed = [c for c in available if c not in claimed]
            unclaimed_lower = [c.lower() for c in unclaimed]
            candidates = difflib.get_close_matches(canonical.lower(), unclaimed_lower, n=1, cutoff=0.8)
            if candidates:
                # Map back to the original-case column name
                matched = unclaimed[unclaimed_lower.index(candidates[0])]

        if matched is not None:
            rename_map[matched] = canonical
            claimed.add(matched)

    return rename_map


def prompt_column_mapping(df: pd.DataFrame, missing: list[str]) -> pd.DataFrame:
    """Interactively map user-supplied column names to canonical names.

    Displays available columns, asks the user to pick one for each missing
    canonical column, and renames the DataFrame accordingly.

    Parameters
    ----------
    df : pd.DataFrame
    missing : list[str]
        Canonical columns absent from *df*.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns renamed to canonical names.

    Raises
    ------
    RuntimeError
        If the user provides an invalid column name.
    """
    if not missing:
        return df

    available = list(df.columns)
    print("\n  ⚠  Some expected columns were not found.")
    print(f"  Available columns: {available}\n")

    rename_map: dict[str, str] = {}
    for canonical in missing:
        while True:
            user_input = input(
                f"  Map '{canonical}' → enter the matching column name from the list above: "
            ).strip()
            if user_input in available:
                rename_map[user_input] = canonical
                break
            print(f"  ✗ '{user_input}' not found. Please enter an exact column name.")

    return df.rename(columns=rename_map)


# ---------------------------------------------------------------------------
# 3. Deduplication
# ---------------------------------------------------------------------------

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where (Date, Description, Amount) are identical.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame.
    """
    before = len(df)
    deduped = df.drop_duplicates(subset=["Date", "Description", "Amount"]).reset_index(
        drop=True
    )
    removed = before - len(deduped)
    if removed:
        print(f"  ✓ Removed {removed:,} duplicate row(s)")
    return deduped


# ---------------------------------------------------------------------------
# 4. Encoding detection
# ---------------------------------------------------------------------------

def _detect_encoding(path: Path) -> str:
    """Detect the character encoding of a file using chardet.

    Parameters
    ----------
    path : Path

    Returns
    -------
    str
        Detected encoding name (e.g. 'utf-8', 'latin-1').
    """
    raw = path.read_bytes()
    result = chardet.detect(raw)
    encoding = result.get("encoding") or "utf-8"
    return encoding


def normalize_encoding(path: str | Path) -> pd.DataFrame:
    """Load a CSV with automatic encoding detection and Latin-1 fallback.

    Parameters
    ----------
    path : str | Path

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    encoding = _detect_encoding(path)
    try:
        return pd.read_csv(path, encoding=encoding)
    except (UnicodeDecodeError, LookupError):
        return pd.read_csv(path, encoding="latin-1")


# ---------------------------------------------------------------------------
# 5. Date normalisation
# ---------------------------------------------------------------------------

def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse heterogeneous date strings in the Date column to YYYY-MM-DD.

    Parameters
    ----------
    df : pd.DataFrame  (must have 'Date' column)

    Returns
    -------
    pd.DataFrame  with Date as str in YYYY-MM-DD format.

    Raises
    ------
    ValueError
        If any date value cannot be parsed.
    """
    parsed = pd.to_datetime(df["Date"], errors="coerce")
    bad_rows = df.loc[parsed.isna(), "Date"]
    if not bad_rows.empty:
        samples = bad_rows.head(5).tolist()
        raise ValueError(
            f"Cannot parse {len(bad_rows)} date value(s). Examples: {samples}\n"
            "Please fix these rows in the source file before re-running."
        )
    result = df.copy()
    result["Date"] = parsed.dt.strftime("%Y-%m-%d")
    return result


# ---------------------------------------------------------------------------
# 6. Amount normalisation
# ---------------------------------------------------------------------------

_CR_DR_RE = re.compile(r"^([0-9,]+\.?[0-9]*)\s*(CR|DR)$", re.IGNORECASE)


def _parse_amount(value) -> float:
    """Convert a single amount value to a signed float.

    Handles:
    - Plain numeric strings / floats / ints
    - Strings with currency symbols (£, $, €, ₹)
    - CR / DR suffix notation  (CR → positive, DR → negative)
    - Already-signed floats
    """
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()
    # Remove common currency symbols
    s = re.sub(r"[£$€₹]", "", s).replace(",", "").strip()

    m = _CR_DR_RE.match(s)
    if m:
        number = float(m.group(1))
        return number if m.group(2).upper() == "CR" else -number

    try:
        return float(s)
    except ValueError as exc:
        raise ValueError(f"Cannot parse amount value: '{value}'") from exc


def normalize_amounts(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the Amount column to signed floats (expenses negative).

    Parameters
    ----------
    df : pd.DataFrame  (must have 'Amount' column)

    Returns
    -------
    pd.DataFrame  with Amount as float64.
    """
    result = df.copy()
    result["Amount"] = result["Amount"].map(_parse_amount)
    return result


# ---------------------------------------------------------------------------
# 7. Card number masking
# ---------------------------------------------------------------------------

_CARD_RE = re.compile(r"\b(\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{0,4}|\d{12,16})\b")


def _mask_card(text: str) -> str:
    """Replace 12-16 digit card number sequences with ****NNNN."""
    def replacer(m: re.Match) -> str:
        digits = re.sub(r"[\s\-]", "", m.group(0))
        if 12 <= len(digits) <= 16:
            return f"****{digits[-4:]}"
        return m.group(0)

    return _CARD_RE.sub(replacer, str(text))


def mask_card_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """Regex-scan the Description column and mask 12–16 digit card numbers.

    Parameters
    ----------
    df : pd.DataFrame  (must have 'Description' column)

    Returns
    -------
    pd.DataFrame  with masked descriptions.
    """
    result = df.copy()
    result["Description"] = result["Description"].map(_mask_card)
    return result


# ---------------------------------------------------------------------------
# 8. Full ingestion pipeline
# ---------------------------------------------------------------------------

def ingest(path: str | Path, interactive: bool = True) -> pd.DataFrame:
    """Run the full ingestion pipeline on a raw bank export.

    Steps:
        load_file → validate_columns → (prompt_column_mapping) →
        remove_duplicates → normalize_dates → normalize_amounts →
        mask_card_numbers

    Parameters
    ----------
    path : str | Path
        Path to the raw export (.csv or .xlsx).
    interactive : bool
        If True, prompt user for column mappings when columns are missing.
        If False, raise ValueError instead of prompting.

    Returns
    -------
    pd.DataFrame
        Clean, normalised DataFrame with canonical schema.
    """
    path = Path(path)
    print(f"\n[Ingestion] {path.name}")

    df = load_file(path)
    missing = validate_columns(df)

    if missing:
        # Auto-map via alias / fuzzy matching
        auto_map = fuzzy_match_columns(missing, list(df.columns))
        if auto_map:
            for src, canonical in auto_map.items():
                print(f"  ↳ Auto-mapped '{src}' → '{canonical}'")
            df = df.rename(columns=auto_map)
            missing = validate_columns(df)

        if missing:
            if interactive:
                df = prompt_column_mapping(df, missing)
            else:
                raise ValueError(
                    f"Missing required columns: {missing}. "
                    f"Could not auto-map from available columns: {list(df.columns)}. "
                    "Re-run without --no-feedback to map interactively."
                )

    df = remove_duplicates(df)
    df = normalize_dates(df)
    df = normalize_amounts(df)
    df = mask_card_numbers(df)

    print(f"  ✓ Ingestion complete — {len(df):,} clean rows\n")
    return df
