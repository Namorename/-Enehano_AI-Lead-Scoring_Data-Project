"""Tests for feature_engineering.py — add_engineered_features."""
import numpy as np
import pandas as pd
import pytest

from feature_engineering import add_engineered_features


# ── helper ────────────────────────────────────────────────────────────────────

def _base_df(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "CreatedDate":                ["2023-03-15"] * n,
        "LastActivityDate":           ["2023-09-01"] * n,
        "Time_to_First_Response_h__c": [0.5, 3.0, 10.0, 30.0, 50.0][:n],
        "Days_Since_Last_Activity__c": [1.0, 7.0, 20.0, 60.0, 200.0][:n],
        "Days_in_Pipeline__c":        [10.0, 30.0, 90.0, 180.0, 360.0][:n],
        "Web_Interactions__c":        [5.0, 15.0, 35.0, 2.0, 0.0][:n],
        "Email_Opens__c":             [2.0] * n,
        "Email_Clicks__c":            [1.0] * n,
        "Email_Open_Rate__c":         [0.3] * n,
        "Form_Submissions__c":        [1.0, 0.0, 2.0, 0.0, 0.0][:n],
        "Content_Downloads__c":       [0.0] * n,
        "Chatbot_Interactions__c":    [0.0] * n,
        "Meetings_Held__c":           [1.0, 0.0, 2.0, 0.0, 3.0][:n],
        "Demo_Requested__c":          [1, 0, 1, 0, 0][:n],
        "Proposal_Sent__c":           [1, 1, 0, 0, 0][:n],
        "Annual_Revenue_MCZK__c":     [100.0] * n,
        "Calls_Made__c":              [2.0] * n,
        "Rating":                     ["Hot", "Warm", "Cold", "Warm", "Hot"][:n],
    })


# ── column existence ──────────────────────────────────────────────────────────

EXPECTED_COLS = [
    "created_month", "created_quarter", "created_year", "created_day_of_week",
    "log1p_ttfr", "log1p_days_since", "log1p_web", "log1p_revenue",
    "meet_x_proposal", "engagement_composite", "fast_response_flag",
    "email_quality", "recency_tier", "pipeline_maturity", "activity_density",
    "rating_encoded", "meetings_capped", "rating_x_meetings", "total_calls_and_meets",
]


def test_all_engineered_columns_created():
    df = _base_df()
    add_engineered_features(df)
    for col in EXPECTED_COLS:
        assert col in df.columns, f"Missing engineered column: {col}"


def test_returns_same_dataframe():
    df = _base_df()
    result = add_engineered_features(df)
    assert result is df


# ── date features ─────────────────────────────────────────────────────────────

def test_created_month():
    df = _base_df(1)
    add_engineered_features(df)
    assert df["created_month"].iloc[0] == 3


def test_created_quarter():
    df = _base_df(1)
    add_engineered_features(df)
    assert df["created_quarter"].iloc[0] == 1


def test_created_year():
    df = _base_df(1)
    add_engineered_features(df)
    assert df["created_year"].iloc[0] == 2023


def test_missing_created_date_gives_zeros():
    df = _base_df(1).drop(columns=["CreatedDate"])
    add_engineered_features(df)
    assert df["created_month"].iloc[0] == 0
    assert df["created_year"].iloc[0] == 0


# ── log transforms ────────────────────────────────────────────────────────────

def test_log1p_ttfr_non_negative():
    df = _base_df()
    add_engineered_features(df)
    assert (df["log1p_ttfr"] >= 0).all()


def test_log1p_ttfr_zero_for_zero_input():
    df = pd.DataFrame({
        "Time_to_First_Response_h__c": [0.0],
        "Annual_Revenue_MCZK__c": [0.0],
    })
    add_engineered_features(df)
    assert df["log1p_ttfr"].iloc[0] == pytest.approx(0.0)


# ── fast_response_flag ────────────────────────────────────────────────────────

def test_fast_response_flag_under_4h():
    df = _base_df()
    add_engineered_features(df)
    assert df["fast_response_flag"].iloc[0] == 1   # 0.5h -> fast
    assert df["fast_response_flag"].iloc[1] == 1   # 3h   -> fast
    assert df["fast_response_flag"].iloc[2] == 0   # 10h  -> not fast


# ── rating_encoded ────────────────────────────────────────────────────────────

def test_rating_encoded_values():
    df = _base_df()
    add_engineered_features(df)
    enc = df["rating_encoded"].tolist()
    assert enc[0] == 2   # Hot
    assert enc[1] == 1   # Warm
    assert enc[2] == 0   # Cold


def test_rating_encoded_unknown_defaults_to_warm():
    df = _base_df(1)
    df["Rating"] = "Unknown"
    add_engineered_features(df)
    assert df["rating_encoded"].iloc[0] == 1


# ── recency_tier ──────────────────────────────────────────────────────────────

def test_recency_tier_most_recent_is_highest():
    df = _base_df()
    add_engineered_features(df)
    # days_since 1, 7, 20, 60, 200 → tiers 4, 3, 2, 1, 0
    assert df["recency_tier"].iloc[0] > df["recency_tier"].iloc[4]


# ── engagement_composite ──────────────────────────────────────────────────────

def test_engagement_composite_non_negative():
    df = _base_df()
    add_engineered_features(df)
    assert (df["engagement_composite"] >= 0).all()


# ── meetings_capped ───────────────────────────────────────────────────────────

def test_meetings_capped_at_four():
    df = _base_df(1)
    df["Meetings_Held__c"] = 10.0
    add_engineered_features(df)
    assert df["meetings_capped"].iloc[0] == 4


# ── missing columns gracefully handled ───────────────────────────────────────

def test_missing_numeric_cols_default_to_zero():
    df = pd.DataFrame({"CreatedDate": ["2023-01-01"]})
    add_engineered_features(df)
    assert df["log1p_revenue"].iloc[0] == pytest.approx(0.0)
    assert df["engagement_composite"].iloc[0] == pytest.approx(0.0)