"""Tests for contracts.py — column contracts, segment mapping, feature frames."""
import pandas as pd
import pytest

from contracts import (
    MODEL_DROP_COLS,
    STATUS_PROXY_COLS,
    WIN_MODEL_FEATURES,
    add_region_alias,
    model_feature_frame,
    present_columns,
    score_segment,
    win_feature_frame,
    HIGH_SCORE_THRESHOLD,
    MEDIUM_SCORE_THRESHOLD,
)
from feature_engineering import add_engineered_features


# ── score_segment ─────────────────────────────────────────────────────────────

def test_score_segment_high():
    assert score_segment(HIGH_SCORE_THRESHOLD) == "High"
    assert score_segment(100) == "High"


def test_score_segment_medium():
    assert score_segment(MEDIUM_SCORE_THRESHOLD) == "Medium"
    assert score_segment(HIGH_SCORE_THRESHOLD - 1) == "Medium"


def test_score_segment_low():
    assert score_segment(0) == "Low"
    assert score_segment(MEDIUM_SCORE_THRESHOLD - 1) == "Low"


# ── present_columns ────────────────────────────────────────────────────────────

def test_present_columns_returns_only_existing():
    cols = ["a", "b", "c"]
    excluded = ["b", "d"]
    assert present_columns(cols, excluded) == ["b"]


def test_present_columns_empty_when_none_match():
    assert present_columns(["x", "y"], ["a", "b"]) == []


def test_present_columns_preserves_order():
    cols = ["z", "a", "m"]
    excluded = ["m", "z"]
    result = present_columns(cols, excluded)
    assert result == ["m", "z"]


# ── model_feature_frame ────────────────────────────────────────────────────────

def test_model_feature_frame_drops_id_and_target(minimal_df):
    add_engineered_features(minimal_df)
    X = model_feature_frame(minimal_df)
    for col in MODEL_DROP_COLS:
        assert col not in X.columns, f"Column '{col}' should have been dropped"


def test_model_feature_frame_keeps_feature_cols(minimal_df):
    add_engineered_features(minimal_df)
    X = model_feature_frame(minimal_df)
    assert len(X.columns) > 0
    assert "Web_Interactions__c" in X.columns


def test_model_feature_frame_exclude_status_proxies(minimal_df):
    add_engineered_features(minimal_df)
    X = model_feature_frame(minimal_df, exclude_status_proxies=True)
    for col in STATUS_PROXY_COLS:
        assert col not in X.columns, f"Proxy column '{col}' should be excluded"


def test_model_feature_frame_retains_proxies_by_default(minimal_df):
    add_engineered_features(minimal_df)
    X = model_feature_frame(minimal_df)
    present = [c for c in STATUS_PROXY_COLS if c in minimal_df.columns]
    for col in present:
        assert col in X.columns


# ── win_feature_frame ──────────────────────────────────────────────────────────

def test_win_feature_frame_subset_of_win_features(minimal_df):
    add_engineered_features(minimal_df)
    X = win_feature_frame(minimal_df)
    for col in X.columns:
        assert col in WIN_MODEL_FEATURES


def test_win_feature_frame_exclude_proxies(minimal_df):
    add_engineered_features(minimal_df)
    X = win_feature_frame(minimal_df, exclude_status_proxies=True)
    for col in STATUS_PROXY_COLS:
        assert col not in X.columns


# ── add_region_alias ───────────────────────────────────────────────────────────

def test_add_region_alias_adds_column(minimal_df):
    out = add_region_alias(minimal_df)
    assert "Region" in out.columns
    assert (out["Region"] == out["Region_Name"]).all()


def test_add_region_alias_no_duplicate_if_present(minimal_df):
    minimal_df["Region"] = "existing"
    out = add_region_alias(minimal_df)
    assert (out["Region"] == "existing").all()


def test_add_region_alias_returns_copy_by_default(minimal_df):
    out = add_region_alias(minimal_df)
    assert out is not minimal_df