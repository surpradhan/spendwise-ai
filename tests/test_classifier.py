"""Tests for scripts/classifier.py"""
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.classifier import (
    classify_all,
    classify_transaction,
    load_keywords,
    save_keywords,
)


SAMPLE_KEYWORDS = {
    "Food & Drink": ["starbucks", "mcdonald"],
    "Transport": ["uber", "lyft"],
    "Groceries": ["whole foods", "trader joe"],
}


# ---------------------------------------------------------------------------
# classify_transaction
# ---------------------------------------------------------------------------

def test_classify_transaction_exact_match():
    assert classify_transaction("STARBUCKS #1234", SAMPLE_KEYWORDS) == "Food & Drink"


def test_classify_transaction_case_insensitive():
    assert classify_transaction("Uber Trip", SAMPLE_KEYWORDS) == "Transport"


def test_classify_transaction_substring_match():
    assert classify_transaction("WHOLE FOODS MARKET", SAMPLE_KEYWORDS) == "Groceries"


def test_classify_transaction_no_match_returns_uncategorized():
    assert classify_transaction("Unknown Merchant XYZ", SAMPLE_KEYWORDS) == "Uncategorized"


def test_classify_transaction_empty_description():
    assert classify_transaction("", SAMPLE_KEYWORDS) == "Uncategorized"


def test_classify_transaction_empty_keywords():
    assert classify_transaction("Starbucks", {}) == "Uncategorized"


# ---------------------------------------------------------------------------
# classify_all
# ---------------------------------------------------------------------------

def _sample_df():
    return pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "Description": ["Starbucks latte", "Lyft ride home", "Random shop"],
        "Amount": [-5.0, -12.0, -30.0],
    })


def test_classify_all_adds_category_column():
    df = _sample_df()
    result = classify_all(df, SAMPLE_KEYWORDS)
    assert "Category" in result.columns


def test_classify_all_correct_categories():
    df = _sample_df()
    result = classify_all(df, SAMPLE_KEYWORDS)
    assert result.iloc[0]["Category"] == "Food & Drink"
    assert result.iloc[1]["Category"] == "Transport"
    assert result.iloc[2]["Category"] == "Uncategorized"


def test_classify_all_pure_does_not_mutate():
    df = _sample_df()
    assert "Category" not in df.columns
    classify_all(df, SAMPLE_KEYWORDS)
    assert "Category" not in df.columns


def test_classify_all_returns_new_dataframe():
    df = _sample_df()
    result = classify_all(df, SAMPLE_KEYWORDS)
    assert result is not df


# ---------------------------------------------------------------------------
# load_keywords / save_keywords
# ---------------------------------------------------------------------------

def test_load_keywords_valid_file(tmp_path):
    kw_file = tmp_path / "keywords.json"
    kw_file.write_text(json.dumps({"Food": ["pizza", "burger"]}), encoding="utf-8")
    result = load_keywords(kw_file)
    assert result == {"Food": ["pizza", "burger"]}


def test_load_keywords_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_keywords(tmp_path / "nonexistent.json")


def test_save_keywords_sorts_keys(tmp_path):
    kw_file = tmp_path / "keywords.json"
    keywords = {"Zebra": ["z"], "Apple": ["a"]}
    save_keywords(keywords, kw_file)
    saved = json.loads(kw_file.read_text(encoding="utf-8"))
    assert list(saved.keys()) == ["Apple", "Zebra"]


def test_save_keywords_sorts_keyword_lists(tmp_path):
    kw_file = tmp_path / "keywords.json"
    keywords = {"Food": ["pizza", "burger", "apple"]}
    save_keywords(keywords, kw_file)
    saved = json.loads(kw_file.read_text(encoding="utf-8"))
    assert saved["Food"] == sorted(["pizza", "burger", "apple"])


def test_save_keywords_deduplicates(tmp_path):
    kw_file = tmp_path / "keywords.json"
    keywords = {"Food": ["pizza", "pizza", "burger"]}
    save_keywords(keywords, kw_file)
    saved = json.loads(kw_file.read_text(encoding="utf-8"))
    assert saved["Food"].count("pizza") == 1


def test_save_then_load_roundtrip(tmp_path):
    kw_file = tmp_path / "keywords.json"
    keywords = {"Transport": ["uber", "lyft"], "Food": ["sushi"]}
    save_keywords(keywords, kw_file)
    loaded = load_keywords(kw_file)
    assert set(loaded["Transport"]) == {"uber", "lyft"}
    assert loaded["Food"] == ["sushi"]
