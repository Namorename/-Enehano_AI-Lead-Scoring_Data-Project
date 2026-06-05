"""
Shared data and scoring contracts.

Keeping these constants in one place prevents the training pipeline, API, and
dashboard from quietly drifting apart as fields are added or renamed.
"""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

REGION_SOURCE_COL = "Region_Name"
REGION_ALIAS_COL = "Region"

ID_COLS = ["Lead_ID", "IČO", "Company_Name"]
TARGET_CONV = "Converted"
TARGET_WIN = "Closed_Won"
STATUS_COL = "Status"

LEAKY_COLS = [
    "Conversion_Probability",
    "Win_Probability__c",
    "Rule_Based_Score",
    "Rule_Based_Segment",
    "IČO_Duplicate_Flag",
]

DATE_COLS = ["CreatedDate", "LastActivityDate"]

REDUNDANT_COLS = [
    "CZ_NACE_Section_Label",  # identical information to CZ_NACE_Section
    "Industry",               # identical mapping to CZ_NACE_Section
    "Legal_Form_Label",       # text version of Legal_Form_Code
    "Region_Code",            # same regional information as Region_Name
    "Founding_Year",          # perfectly collinear with Company_Age_Years
    "Owner",                  # pure assignment/noise in the synthetic dataset
]

# Pipeline-stage fields (and their engineered children) that are driven by the
# excluded Status column in the synthetic generator. They behave as leakage
# proxies: the model can reconstruct Status from them, which inflates AUC on
# this data but will not generalise to real CRM pipelines. Dropping them is
# opt-in (see model_feature_frame/win_feature_frame `exclude_status_proxies`)
# so the default behaviour and existing artifacts are unchanged.
STATUS_PROXY_COLS = [
    "Days_in_Pipeline__c",
    "Days_Since_Last_Activity__c",
    "log1p_days_since",       # log1p(Days_Since_Last_Activity__c)
    "recency_tier",           # binned Days_Since_Last_Activity__c
    "pipeline_maturity",      # log1p(Days_in_Pipeline__c)
    "activity_density",       # engagement / (Days_in_Pipeline__c + 1)
]

DATE_ENGINEERED_FEATURES = [
    "created_month",          # 1-12, 0 when source date is unavailable
    "created_quarter",        # 1-4, 0 when source date is unavailable
    "created_year",           # source year, 0 when source date is unavailable
    "created_day_of_week",    # Monday=0, Sunday=6, 0 when source date is unavailable
]

LOG_ENGINEERED_FEATURES = [
    "log1p_ttfr",
    "log1p_days_since",
    "log1p_web",
    "log1p_revenue",
]

CONVERSION_ENGINEERED_FEATURES = [
    "meet_x_proposal",
    "engagement_composite",
    "fast_response_flag",
    "email_quality",
    "recency_tier",
    "pipeline_maturity",
    "activity_density",
]

WIN_ENGINEERED_FEATURES = [
    "rating_encoded",
    "meetings_capped",
    "rating_x_meetings",
    "total_calls_and_meets",
]

ENGINEERED_FEATURE_COLS = (
    DATE_ENGINEERED_FEATURES
    + LOG_ENGINEERED_FEATURES
    + CONVERSION_ENGINEERED_FEATURES
    + WIN_ENGINEERED_FEATURES
)

WIN_MODEL_FEATURES = [
    "Meetings_Held__c",
    "meetings_capped",
    "meet_x_proposal",
    "Proposal_Sent__c",
    "rating_encoded",
    "rating_x_meetings",
    "Time_to_First_Response_h__c",
    "fast_response_flag",
    "Company_Age_Years",
    "NumberOfEmployees",
    "Annual_Revenue_MCZK__c",
    "log1p_revenue",
    "CZ_NACE_Section",
    "Region_Name",
    "Days_in_Pipeline__c",
    "Days_Since_Last_Activity__c",
    "recency_tier",
    "total_calls_and_meets",
    "activity_density",
]

BASE_NON_FEATURE_COLS = ID_COLS + [TARGET_CONV, TARGET_WIN, STATUS_COL] + LEAKY_COLS + DATE_COLS
MODEL_DROP_COLS = BASE_NON_FEATURE_COLS + REDUNDANT_COLS

# Keep API payload filtering separate from training feature filtering. The app
# can still send redundant fields so existing model artifacts keep working; new
# preprocessors trained with MODEL_DROP_COLS will simply ignore those extras.
API_PAYLOAD_EXCLUDE_COLS = set(BASE_NON_FEATURE_COLS) | {REGION_ALIAS_COL}

HIGH_SCORE_THRESHOLD = 70
MEDIUM_SCORE_THRESHOLD = 40


def present_columns(columns: Iterable[str], excluded: Iterable[str]) -> list[str]:
    """Return excluded columns that exist in a provided column collection."""
    existing = set(columns)
    return [col for col in excluded if col in existing]


def model_feature_frame(
    df: pd.DataFrame, *, exclude_status_proxies: bool = False
) -> pd.DataFrame:
    """
    Return the model feature matrix by dropping all non-feature columns.

    When `exclude_status_proxies` is True, the Status-derived leakage proxies
    (STATUS_PROXY_COLS) are dropped as well, producing a feature set that
    generalises to real CRM data at the cost of some synthetic-benchmark AUC.
    """
    drop = list(MODEL_DROP_COLS)
    if exclude_status_proxies:
        drop = drop + STATUS_PROXY_COLS
    return df.drop(columns=present_columns(df.columns, drop))


def win_feature_frame(
    df: pd.DataFrame, *, exclude_status_proxies: bool = False
) -> pd.DataFrame:
    """
    Return the narrower opportunity-stage feature matrix for Closed_Won.

    When `exclude_status_proxies` is True, any Status-derived leakage proxies in
    WIN_MODEL_FEATURES are removed so the win model is trained on the same
    leakage-controlled basis as the conversion model.
    """
    excluded = set(STATUS_PROXY_COLS) if exclude_status_proxies else set()
    cols = [
        col for col in WIN_MODEL_FEATURES
        if col in df.columns and col not in excluded
    ]
    return df[cols].copy()


def add_region_alias(df: pd.DataFrame, *, copy: bool = True) -> pd.DataFrame:
    """Ensure a dataframe has the UI/baseline-friendly `Region` alias."""
    if REGION_SOURCE_COL not in df.columns or REGION_ALIAS_COL in df.columns:
        return df.copy() if copy else df

    out = df.copy() if copy else df
    out[REGION_ALIAS_COL] = out[REGION_SOURCE_COL]
    return out


def score_segment(score: float) -> str:
    """Map a 0-100 score to the High / Medium / Low segment contract."""
    if score >= HIGH_SCORE_THRESHOLD:
        return "High"
    if score >= MEDIUM_SCORE_THRESHOLD:
        return "Medium"
    return "Low"
