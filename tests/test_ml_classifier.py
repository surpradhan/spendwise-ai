"""Tests for scripts/ml_classifier.py"""
from __future__ import annotations

import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from scripts.ml_classifier import (
    classify_all_v2,
    is_model_stale,
    load_model,
    load_training_data,
    predict_with_confidence,
    save_model,
    train_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KEYWORDS = {
    "Food & Drink": ["starbucks", "mcdonald"],
    "Transport": ["uber", "lyft"],
}


def _make_categorized_csv(path: Path, n: int = 25) -> None:
    """Write a small categorised CSV with n rows (alternating two categories)."""
    rows = []
    for i in range(n):
        if i % 2 == 0:
            rows.append({"Date": "2024-01-01", "Description": f"starbucks visit {i}",
                         "Amount": -5.0, "Category": "Food & Drink"})
        else:
            rows.append({"Date": "2024-01-02", "Description": f"uber ride {i}",
                         "Amount": -12.0, "Category": "Transport"})
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_df(descriptions: list[str], categories: list[str] | None = None) -> pd.DataFrame:
    data: dict = {
        "Date": ["2024-01-01"] * len(descriptions),
        "Description": descriptions,
        "Amount": [-10.0] * len(descriptions),
    }
    if categories:
        data["Category"] = categories
    return pd.DataFrame(data)


def _trained_pipeline(n: int = 25):
    """Return a pipeline trained on small synthetic data."""
    rows = []
    for i in range(n):
        if i % 2 == 0:
            rows.append((f"starbucks purchase {i}", "Food & Drink"))
        else:
            rows.append((f"uber trip {i}", "Transport"))
    X = pd.Series([r[0] for r in rows])
    y = pd.Series([r[1] for r in rows])
    return train_model(X, y)


# ---------------------------------------------------------------------------
# load_training_data
# ---------------------------------------------------------------------------

def test_load_training_data_excludes_uncategorized(tmp_path):
    csv = tmp_path / "jan_categorized.csv"
    df = pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "Description": ["Starbucks", "Some Random Shop", "Uber"],
        "Amount": [-5.0, -20.0, -12.0],
        "Category": ["Food & Drink", "Uncategorized", "Transport"],
    })
    df.to_csv(csv, index=False)

    X, y = load_training_data(tmp_path)
    assert "Uncategorized" not in y.values
    assert len(X) == 2


def test_load_training_data_empty_dir(tmp_path):
    X, y = load_training_data(tmp_path)
    assert len(X) == 0
    assert len(y) == 0


def test_load_training_data_nonexistent_dir(tmp_path):
    X, y = load_training_data(tmp_path / "does_not_exist")
    assert len(X) == 0
    assert len(y) == 0


# ---------------------------------------------------------------------------
# train_model
# ---------------------------------------------------------------------------

def test_train_model_returns_pipeline():
    from sklearn.pipeline import Pipeline
    pipeline = _trained_pipeline()
    assert isinstance(pipeline, Pipeline)


def test_train_model_raises_on_single_class():
    X = pd.Series([f"starbucks {i}" for i in range(25)])
    y = pd.Series(["Food & Drink"] * 25)
    with pytest.raises(RuntimeError, match="distinct categor"):
        train_model(X, y)


def test_train_model_raises_below_min_samples():
    X = pd.Series(["starbucks", "uber", "lyft"])
    y = pd.Series(["Food & Drink", "Transport", "Transport"])
    with pytest.raises(RuntimeError, match="Insufficient training data"):
        train_model(X, y)


# ---------------------------------------------------------------------------
# predict_with_confidence
# ---------------------------------------------------------------------------

def test_predict_with_confidence_above_threshold():
    """Model should recognise its own training data with high confidence."""
    pipeline = _trained_pipeline(30)
    descriptions = pd.Series(["starbucks purchase 0", "uber trip 1"])
    result = predict_with_confidence(pipeline, descriptions, threshold=0.50)
    assert list(result) == ["Food & Drink", "Transport"]


def test_predict_with_confidence_below_threshold_returns_uncategorized():
    """With threshold=1.1 (impossible), all predictions collapse to Uncategorized."""
    pipeline = _trained_pipeline(30)
    descriptions = pd.Series(["starbucks purchase 0", "uber trip 1"])
    result = predict_with_confidence(pipeline, descriptions, threshold=1.1)
    assert all(v == "Uncategorized" for v in result)


# ---------------------------------------------------------------------------
# save_model / load_model
# ---------------------------------------------------------------------------

def test_save_and_load_model_roundtrip(tmp_path):
    pipeline = _trained_pipeline()
    meta = {"trained_at": datetime.now(timezone.utc).isoformat(), "n_samples": 25}
    save_model(pipeline, meta, tmp_path / "models")

    loaded = load_model(tmp_path / "models")
    assert loaded is not None

    # Should produce same predictions
    desc = pd.Series(["starbucks purchase"])
    original_pred = pipeline.predict(desc)
    loaded_pred = loaded.predict(desc)
    assert list(original_pred) == list(loaded_pred)


def test_load_model_returns_none_when_no_file(tmp_path):
    result = load_model(tmp_path / "empty_models")
    assert result is None


# ---------------------------------------------------------------------------
# is_model_stale
# ---------------------------------------------------------------------------

def test_is_model_stale_true_when_new_csv(tmp_path):
    model_dir = tmp_path / "models"
    processed_dir = tmp_path / "processed"
    model_dir.mkdir()
    processed_dir.mkdir()

    # Write meta with a past timestamp
    past = "2000-01-01T00:00:00+00:00"
    meta = {"trained_at": past}
    (model_dir / "classifier_meta.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )

    # Write a CSV (its mtime will be now, newer than the past timestamp)
    _make_categorized_csv(processed_dir / "jan_categorized.csv")

    assert is_model_stale(model_dir, processed_dir) is True


def test_is_model_stale_false_when_model_fresh(tmp_path):
    model_dir = tmp_path / "models"
    processed_dir = tmp_path / "processed"
    model_dir.mkdir()
    processed_dir.mkdir()

    # Write a CSV first
    _make_categorized_csv(processed_dir / "jan_categorized.csv")

    # Write meta with a future timestamp
    future = "2099-12-31T23:59:59+00:00"
    meta = {"trained_at": future}
    (model_dir / "classifier_meta.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )

    assert is_model_stale(model_dir, processed_dir) is False


def test_is_model_stale_true_when_meta_missing(tmp_path):
    model_dir = tmp_path / "models"
    processed_dir = tmp_path / "processed"
    model_dir.mkdir()
    processed_dir.mkdir()
    _make_categorized_csv(processed_dir / "jan_categorized.csv")

    assert is_model_stale(model_dir, processed_dir) is True


def test_is_model_stale_false_when_model_dir_missing(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    assert is_model_stale(tmp_path / "no_models", processed_dir) is False


# ---------------------------------------------------------------------------
# classify_all_v2
# ---------------------------------------------------------------------------

def _base_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        "Description": [
            "starbucks grande latte",   # keyword → Food & Drink
            "uber pool ride",            # keyword → Transport
            "some mystery shop",         # uncategorized
            "another unknown vendor",    # uncategorized
        ],
        "Amount": [-5.0, -12.0, -30.0, -8.0],
    })


def test_classify_all_v2_keyword_takes_precedence(tmp_path):
    """Keyword-matched rows must not be overridden by ML."""
    pipeline = _trained_pipeline(30)
    model_dir = tmp_path / "models"
    save_model(pipeline, {"trained_at": "2099-01-01T00:00:00+00:00"}, model_dir)

    df = _base_df()
    result = classify_all_v2(df, _KEYWORDS, model_path=model_dir)

    assert result.iloc[0]["Category"] == "Food & Drink"
    assert result.iloc[0]["ML_Classified"] is False or result.iloc[0]["ML_Classified"] == False  # noqa: E712
    assert result.iloc[1]["Category"] == "Transport"


def test_classify_all_v2_ml_fills_uncategorized(tmp_path):
    """ML model should fill in rows that keywords left as Uncategorized."""
    pipeline = _trained_pipeline(30)
    model_dir = tmp_path / "models"
    save_model(pipeline, {"trained_at": "2099-01-01T00:00:00+00:00"}, model_dir)

    # Use descriptions the model trained on directly
    df = pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02"],
        "Description": ["starbucks purchase 0", "uber trip 1"],
        "Amount": [-5.0, -12.0],
    })
    result = classify_all_v2(df, {}, model_path=model_dir, threshold=0.50)

    # With no keywords, both start as Uncategorized; ML should fill them
    assert result.iloc[0]["Category"] == "Food & Drink"
    assert result.iloc[1]["Category"] == "Transport"
    assert result.iloc[0]["ML_Classified"] == True  # noqa: E712
    assert result.iloc[1]["ML_Classified"] == True  # noqa: E712


def test_classify_all_v2_pure_no_mutation(tmp_path):
    """classify_all_v2 must not mutate the input DataFrame."""
    df = _base_df()
    original_cols = list(df.columns)
    classify_all_v2(df, _KEYWORDS, model_path=tmp_path / "no_model")
    assert list(df.columns) == original_cols
    assert "Category" not in df.columns
    assert "ML_Classified" not in df.columns


def test_classify_all_v2_cold_start_no_model(tmp_path):
    """With no saved model, keyword results are returned with ML_Classified=False."""
    df = _base_df()
    result = classify_all_v2(df, _KEYWORDS, model_path=tmp_path / "empty_dir")

    assert "Category" in result.columns
    assert "ML_Classified" in result.columns
    assert not result["ML_Classified"].any()
    assert result.iloc[0]["Category"] == "Food & Drink"


def test_classify_all_v2_ml_classified_column_added(tmp_path):
    """ML_Classified column must always be present in output."""
    df = _base_df()
    result = classify_all_v2(df, _KEYWORDS, model_path=tmp_path / "empty_dir")
    assert "ML_Classified" in result.columns
    assert result["ML_Classified"].dtype == bool or result["ML_Classified"].isin([True, False]).all()
