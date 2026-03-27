"""
Module 2 — Transaction Classifier
==================================
Applies keyword-based category matching to each transaction and manages the
feedback loop that persists user corrections back to the keyword dictionary.

Category → keyword mapping is stored in config/keywords.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# 1. Keyword I/O
# ---------------------------------------------------------------------------

def load_keywords(path: str | Path) -> dict[str, list[str]]:
    """Load the category → keyword mapping from a JSON file.

    Parameters
    ----------
    path : str | Path
        Path to keywords.json.

    Returns
    -------
    dict[str, list[str]]
        Mapping of category name → list of keyword strings.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Keywords file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_keywords(keywords: dict[str, list[str]], path: str | Path) -> None:
    """Write the keyword dictionary back to JSON with sorted keys.

    Parameters
    ----------
    keywords : dict[str, list[str]]
    path : str | Path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sort top-level keys and keyword lists for clean diffs
    sorted_kw = {
        cat: sorted(set(kws))
        for cat, kws in sorted(keywords.items())
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(sorted_kw, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 2. Classification
# ---------------------------------------------------------------------------

def classify_transaction(description: str, keywords: dict[str, list[str]]) -> str:
    """Apply case-insensitive substring matching to classify one transaction.

    Returns the first matching category, or 'Uncategorized' if no keyword
    matches.

    Parameters
    ----------
    description : str
    keywords : dict[str, list[str]]

    Returns
    -------
    str
        Category name or 'Uncategorized'.
    """
    desc_lower = str(description).lower()
    for category, kw_list in keywords.items():
        for kw in kw_list:
            if kw.lower() in desc_lower:
                return category
    return "Uncategorized"


def classify_all(df: pd.DataFrame, keywords: dict[str, list[str]]) -> pd.DataFrame:
    """Classify every transaction and append a 'Category' column.

    Pure function — never mutates the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame  (must have 'Description' column)
    keywords : dict[str, list[str]]

    Returns
    -------
    pd.DataFrame  with new 'Category' column appended.
    """
    result = df.copy()
    result["Category"] = result["Description"].map(
        lambda desc: classify_transaction(desc, keywords)
    )
    n_uncategorized = (result["Category"] == "Uncategorized").sum()
    total = len(result)
    print(
        f"  ✓ Classified {total - n_uncategorized:,}/{total:,} transactions "
        f"({n_uncategorized:,} uncategorized)"
    )
    return result


# ---------------------------------------------------------------------------
# 3. Interactive feedback loop
# ---------------------------------------------------------------------------

def prompt_uncategorized(
    df: pd.DataFrame,
    keywords: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Interactively review uncategorized transactions and collect corrections.

    For each uncategorized transaction the user is shown the date, description,
    and amount and asked to type a category.  Their input is:
    - Used to update the Category column in the returned DataFrame.
    - Added as a keyword to the keywords dict (if the user provides a
      non-empty keyword to add).

    Typing 's' or pressing Enter with no input skips the transaction.
    Typing 'q' exits the review loop early.

    Parameters
    ----------
    df : pd.DataFrame  (must have Date, Description, Amount, Category columns)
    keywords : dict[str, list[str]]

    Returns
    -------
    tuple[pd.DataFrame, dict[str, list[str]]]
        Updated DataFrame and keywords dict.
    """
    uncategorized_idx = df.index[df["Category"] == "Uncategorized"].tolist()
    if not uncategorized_idx:
        print("  ✓ No uncategorized transactions — nothing to review.")
        return df, keywords

    total = len(uncategorized_idx)
    print(f"\n  {total} uncategorized transaction(s) need review.")
    print("  Commands: type a category name | 's' = skip | 'q' = quit review\n")

    existing_categories = sorted(keywords.keys())
    print(f"  Known categories: {', '.join(existing_categories)}\n")

    result = df.copy()
    updated_kw = {cat: list(kws) for cat, kws in keywords.items()}

    for i, idx in enumerate(uncategorized_idx, 1):
        row = result.loc[idx]
        print(
            f"  [{i}/{total}]  {row['Date']}  |  {row['Description']}  |  "
            f"${abs(row['Amount']):,.2f}"
        )
        category_input = input("  Category: ").strip()

        if category_input.lower() == "q":
            print("  ↩  Review exited early.")
            break
        if not category_input or category_input.lower() == "s":
            continue

        # Update category on this row
        result.at[idx, "Category"] = category_input

        # Optionally persist a keyword
        kw_input = input(
            f"  Add a keyword for '{category_input}' "
            "(or Enter to skip): "
        ).strip().lower()
        if kw_input:
            if category_input not in updated_kw:
                updated_kw[category_input] = []
            if kw_input not in updated_kw[category_input]:
                updated_kw[category_input].append(kw_input)
                print(f"  ✓ Added keyword '{kw_input}' → '{category_input}'")

    return result, updated_kw


# ---------------------------------------------------------------------------
# 4. Output
# ---------------------------------------------------------------------------

def write_processed_csv(df: pd.DataFrame, source_path: str | Path) -> Path:
    """Write the categorised DataFrame to data/processed/.

    Output path: data/processed/{stem}_categorized.csv

    Parameters
    ----------
    df : pd.DataFrame
    source_path : str | Path
        Original input file path (used to derive output filename).

    Returns
    -------
    Path
        Path to the written CSV.
    """
    source_path = Path(source_path)
    # Resolve relative to repo root (two levels up from scripts/)
    repo_root = Path(__file__).parent.parent
    out_dir = repo_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source_path.stem}_categorized.csv"
    df.to_csv(out_path, index=False)
    print(f"  ✓ Saved categorised data → '{out_path}'")
    return out_path
