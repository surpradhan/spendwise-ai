"""
Module 6 — ML Classifier (v2.0)
================================
Augments the keyword-based classifier with a TF-IDF + Logistic Regression
model that fills in transactions the keyword rules leave as "Uncategorized".

The model is trained from previously categorised CSVs in data/processed/ and
is persisted to models/classifier.pkl alongside a JSON metadata file.

Public API
----------
load_training_data(processed_dir)  -> (X, y)
train_model(X, y)                  -> Pipeline
save_model(model, meta, model_dir) -> None
load_model(model_dir)              -> Pipeline | None
predict_with_confidence(model, descriptions, threshold) -> pd.Series
is_model_stale(model_dir, processed_dir) -> bool
classify_all_v2(df, keywords, model_path, threshold)    -> pd.DataFrame
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.classifier import classify_all


# ---------------------------------------------------------------------------
# 1. Training data
# ---------------------------------------------------------------------------

def load_training_data(processed_dir: Path) -> tuple[pd.Series, pd.Series]:
    """Load labelled transactions from all categorised CSVs in processed_dir.

    Reads every file matching ``*_categorized.csv``, drops rows where
    Category is "Uncategorized", and deduplicates on (Description, Category).

    Parameters
    ----------
    processed_dir : Path
        Directory containing ``*_categorized.csv`` files.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        ``(X_descriptions, y_categories)`` — both as ``pd.Series``.
        Returns a pair of empty Series if the directory doesn't exist or
        contains no matching files.
    """
    processed_dir = Path(processed_dir)

    if not processed_dir.exists():
        return pd.Series(dtype=str), pd.Series(dtype=str)

    csv_files = list(processed_dir.glob("*_categorized.csv"))
    if not csv_files:
        return pd.Series(dtype=str), pd.Series(dtype=str)

    frames = [pd.read_csv(f) for f in csv_files]
    combined = pd.concat(frames, ignore_index=True)

    # Keep only categorised rows
    labelled = combined[combined["Category"] != "Uncategorized"].copy()

    # Deduplicate on (Description, Category)
    labelled = labelled.drop_duplicates(subset=["Description", "Category"])

    X = labelled["Description"].reset_index(drop=True)
    y = labelled["Category"].reset_index(drop=True)
    return X, y


# ---------------------------------------------------------------------------
# 2. Training
# ---------------------------------------------------------------------------

def train_model(X: pd.Series, y: pd.Series) -> Pipeline:
    """Fit a TF-IDF + Logistic Regression pipeline on the supplied data.

    Parameters
    ----------
    X : pd.Series
        Transaction descriptions.
    y : pd.Series
        Corresponding category labels.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted pipeline ready for inference.

    Raises
    ------
    RuntimeError
        If fewer than 2 distinct classes are present, or if fewer than
        20 total training samples are provided.
    """
    n_samples = len(X)
    n_classes = y.nunique()

    if n_samples < 20:
        raise RuntimeError(
            f"Insufficient training data: {n_samples} sample(s) found, "
            "but at least 20 are required. Categorise more transactions "
            "to enable ML classification."
        )

    if n_classes < 2:
        raise RuntimeError(
            f"Only {n_classes} distinct categories found in training data. "
            "At least 2 distinct categories are required to train the model."
        )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)),
        ("clf", LogisticRegression(max_iter=500, C=1.0)),
    ])
    pipeline.fit(X, y)
    return pipeline


# ---------------------------------------------------------------------------
# 3. Persistence
# ---------------------------------------------------------------------------

def save_model(model: Pipeline, meta: dict, model_dir: Path) -> None:
    """Persist the fitted pipeline and metadata to model_dir.

    Creates ``classifier.pkl`` (pickle) and ``classifier_meta.json``.
    Creates model_dir if it doesn't exist.

    Parameters
    ----------
    model : Pipeline
        Fitted sklearn pipeline.
    meta : dict
        Arbitrary metadata to persist alongside the model (e.g. trained_at,
        n_samples, classes).
    model_dir : Path
        Directory where model artefacts will be written.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = model_dir / "classifier.pkl"
    with pkl_path.open("wb") as fh:
        pickle.dump(model, fh)

    meta_path = model_dir / "classifier_meta.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)


def load_model(model_dir: Path) -> Pipeline | None:
    """Load a previously saved pipeline from model_dir.

    Parameters
    ----------
    model_dir : Path
        Directory containing ``classifier.pkl``.

    Returns
    -------
    Pipeline | None
        The fitted pipeline, or ``None`` if the file doesn't exist (cold start).
        Never raises.
    """
    pkl_path = Path(model_dir) / "classifier.pkl"
    if not pkl_path.exists():
        return None
    try:
        with pkl_path.open("rb") as fh:
            return pickle.load(fh)
    except Exception:  # noqa: BLE001 — cold-start silence is intentional
        return None


# ---------------------------------------------------------------------------
# 4. Inference
# ---------------------------------------------------------------------------

def predict_with_confidence(
    model: Pipeline,
    descriptions: pd.Series,
    threshold: float,
) -> pd.Series:
    """Predict categories, returning "Uncategorized" where confidence is low.

    Pure function — never mutates the input Series.

    Parameters
    ----------
    model : Pipeline
        Fitted sklearn pipeline with ``predict_proba`` support.
    descriptions : pd.Series
        Transaction descriptions to classify.
    threshold : float
        Minimum predicted probability required to accept a label.
        Predictions below this value are replaced with "Uncategorized".

    Returns
    -------
    pd.Series
        Predicted category labels aligned to the input index.
    """
    if descriptions.empty:
        return pd.Series(dtype=str)

    proba = model.predict_proba(descriptions)          # shape (n, n_classes)
    max_proba = proba.max(axis=1)                      # confidence per row
    best_idx = proba.argmax(axis=1)                    # index of top class
    predicted = model.classes_[best_idx]               # derive labels (single pass)

    labels = pd.Series(
        [label if conf >= threshold else "Uncategorized"
         for label, conf in zip(predicted, max_proba)],
        index=descriptions.index,
        dtype=str,
    )
    return labels


# ---------------------------------------------------------------------------
# 5. Staleness check
# ---------------------------------------------------------------------------

def is_model_stale(model_dir: Path, processed_dir: Path) -> bool:
    """Check whether the saved model pre-dates the newest processed CSV.

    Returns ``True`` (stale) when:
    - ``classifier_meta.json`` is absent, OR
    - the newest ``*_categorized.csv`` mtime is newer than ``trained_at``.

    Returns ``False`` (fresh) when:
    - model_dir or processed_dir don't exist, OR
    - no CSVs are found (nothing to retrain on), OR
    - model was trained after the newest CSV.

    Parameters
    ----------
    model_dir : Path
        Directory containing ``classifier_meta.json``.
    processed_dir : Path
        Directory containing ``*_categorized.csv`` files.

    Returns
    -------
    bool
    """
    model_dir = Path(model_dir)
    processed_dir = Path(processed_dir)

    if not model_dir.exists() or not processed_dir.exists():
        return False

    meta_path = model_dir / "classifier_meta.json"
    if not meta_path.exists():
        return True

    csv_files = list(processed_dir.glob("*_categorized.csv"))
    if not csv_files:
        return False

    newest_mtime = max(f.stat().st_mtime for f in csv_files)

    try:
        with meta_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        trained_at_str = meta["trained_at"]
        trained_at = datetime.fromisoformat(trained_at_str).timestamp()
    except (KeyError, ValueError, OSError):
        return True

    return newest_mtime > trained_at


# ---------------------------------------------------------------------------
# 6. Combined v2 classifier
# ---------------------------------------------------------------------------

def classify_all_v2(
    df: pd.DataFrame,
    keywords: dict,
    model_path: Path,
    threshold: float = 0.70,
) -> pd.DataFrame:
    """Classify transactions using keywords first, then ML for remainders.

    Pipeline:
    1. Run ``classify_all(df, keywords)`` from ``scripts.classifier``.
    2. For rows still "Uncategorized", run ML model (if one is loaded).
    3. Accept ML predictions whose confidence >= threshold; mark with
       ``ML_Classified=True``.
    4. Rows that remain "Uncategorized" after ML (or when no model exists)
       get ``ML_Classified=False``.

    Pure function — never mutates the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Ingested transaction DataFrame (Date, Description, Amount columns).
    keywords : dict[str, list[str]]
        Keyword → category mapping for the rule-based first pass.
    model_path : Path
        Directory containing ``classifier.pkl``; passed to ``load_model``.
    threshold : float
        Minimum confidence for ML predictions to be accepted (default 0.70).

    Returns
    -------
    pd.DataFrame
        Copy of df with ``Category`` and ``ML_Classified`` columns appended.
    """
    # Step 1 — keyword classification (returns a copy)
    result = classify_all(df, keywords)

    # Initialise ML_Classified column — False for everything by default
    result = result.assign(ML_Classified=False)

    # Step 2 — load model (returns None on cold start)
    model = load_model(Path(model_path))

    if model is None:
        return result

    # Step 3 — apply ML to rows still Uncategorized
    uncategorized_mask = result["Category"] == "Uncategorized"
    if not uncategorized_mask.any():
        return result

    ml_descriptions = result.loc[uncategorized_mask, "Description"]
    ml_predictions = predict_with_confidence(model, ml_descriptions, threshold)

    # Build new columns via assignment (no in-place mutation)
    new_category = result["Category"].copy()
    new_ml_flag = result["ML_Classified"].copy()

    accepted_mask = ml_predictions != "Uncategorized"
    accepted_idx = ml_predictions[accepted_mask].index

    new_category.loc[accepted_idx] = ml_predictions.loc[accepted_idx]
    new_ml_flag.loc[accepted_idx] = True

    result = result.assign(Category=new_category, ML_Classified=new_ml_flag)
    return result
