"""Tests for modeling.py — LightGBMFramePreprocessor."""
import numpy as np
import pandas as pd
import pytest

from modeling import LightGBMFramePreprocessor


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age":     [10.0, 20.0, 30.0, 40.0],
        "revenue": [100.0, 200.0, np.nan, 400.0],
        "sector":  ["J", "M", "K", "J"],
        "region":  ["Prague", "Brno", "Prague", "Ostrava"],
    })


CAT_COLS = ["sector", "region"]


# ── fit ───────────────────────────────────────────────────────────────────────

def test_fit_stores_feature_names(sample_df):
    pre = LightGBMFramePreprocessor(CAT_COLS)
    pre.fit(sample_df)
    assert list(pre.feature_names_in_) == list(sample_df.columns)


def test_fit_stores_only_present_categoricals(sample_df):
    pre = LightGBMFramePreprocessor(["sector", "nonexistent"])
    pre.fit(sample_df)
    assert pre.categorical_cols_ == ["sector"]


def test_fit_builds_category_values(sample_df):
    pre = LightGBMFramePreprocessor(CAT_COLS)
    pre.fit(sample_df)
    assert set(pre.category_values_["sector"]) >= {"J", "M", "K", "__missing__"}


# ── transform ─────────────────────────────────────────────────────────────────

def test_transform_output_has_same_columns(sample_df):
    pre = LightGBMFramePreprocessor(CAT_COLS)
    pre.fit(sample_df)
    out = pre.transform(sample_df)
    assert list(out.columns) == list(sample_df.columns)


def test_transform_categoricals_are_categorical_dtype(sample_df):
    pre = LightGBMFramePreprocessor(CAT_COLS)
    pre.fit(sample_df)
    out = pre.transform(sample_df)
    assert str(out["sector"].dtype) == "category"
    assert str(out["region"].dtype) == "category"


def test_transform_unknown_category_mapped_to_missing(sample_df):
    pre = LightGBMFramePreprocessor(CAT_COLS)
    pre.fit(sample_df)
    new = pd.DataFrame({
        "age":     [5.0],
        "revenue": [50.0],
        "sector":  ["Z"],   # never seen
        "region":  ["Plzen"],
    })
    out = pre.transform(new)
    assert out["sector"].iloc[0] == "__missing__"


def test_transform_nan_category_mapped_to_missing(sample_df):
    pre = LightGBMFramePreprocessor(["sector"])
    pre.fit(sample_df)
    new = sample_df.copy()
    new["sector"] = None
    out = pre.transform(new)
    assert (out["sector"] == "__missing__").all()


def test_transform_missing_column_filled_with_nan(sample_df):
    pre = LightGBMFramePreprocessor(CAT_COLS)
    pre.fit(sample_df)
    new = sample_df.drop(columns=["revenue"])
    out = pre.transform(new)
    assert "revenue" in out.columns
    assert out["revenue"].isna().all()


def test_transform_numeric_nan_passthrough(sample_df):
    pre = LightGBMFramePreprocessor(CAT_COLS)
    pre.fit(sample_df)
    out = pre.transform(sample_df)
    assert pd.isna(out["revenue"].iloc[2])


# ── get_feature_names_out ─────────────────────────────────────────────────────

def test_get_feature_names_out_matches_columns(sample_df):
    pre = LightGBMFramePreprocessor(CAT_COLS)
    pre.fit(sample_df)
    names = pre.get_feature_names_out()
    assert list(names) == list(sample_df.columns)


# ── sklearn compatibility ─────────────────────────────────────────────────────

def test_fit_transform_consistent(sample_df):
    pre = LightGBMFramePreprocessor(CAT_COLS)
    out_fit_transform = pre.fit_transform(sample_df)
    pre2 = LightGBMFramePreprocessor(CAT_COLS)
    pre2.fit(sample_df)
    out_separate = pre2.transform(sample_df)
    pd.testing.assert_frame_equal(out_fit_transform, out_separate)