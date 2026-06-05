"""
Feature engineering shared by training and inference.

The function mutates the provided DataFrame in place and returns it so callers
can use it fluently while avoiding unnecessary copies for batch scoring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    """Return a numeric series, falling back to a scalar default when missing."""
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def _date(df: pd.DataFrame, column: str) -> pd.Series:
    """Return parsed datetimes; missing columns become NaT."""
    if column not in df.columns:
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    return pd.to_datetime(df[column], errors="coerce")


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add model-ready engineered features in place and return the same DataFrame.

    Training data has CreatedDate and LastActivityDate, but API scoring payloads
    may not. Date-derived features therefore use 0 as an explicit "unknown"
    bucket when dates are absent or invalid, keeping inference compatible with
    retrained preprocessors.
    """
    created = _date(df, "CreatedDate")
    last_activity = _date(df, "LastActivityDate")
    if "CreatedDate" in df.columns:
        df["CreatedDate"] = created
    if "LastActivityDate" in df.columns:
        df["LastActivityDate"] = last_activity

    df["created_month"] = created.dt.month.fillna(0).astype(int)
    df["created_quarter"] = created.dt.quarter.fillna(0).astype(int)
    df["created_year"] = created.dt.year.fillna(0).astype(int)
    df["created_day_of_week"] = created.dt.dayofweek.fillna(0).astype(int)

    ttfr = _numeric(df, "Time_to_First_Response_h__c")
    days_since = _numeric(df, "Days_Since_Last_Activity__c")
    web = _numeric(df, "Web_Interactions__c")
    revenue = _numeric(df, "Annual_Revenue_MCZK__c")

    df["log1p_ttfr"] = np.log1p(ttfr.clip(lower=0))
    df["log1p_days_since"] = np.log1p(days_since.clip(lower=0))
    df["log1p_web"] = np.log1p(web.clip(lower=0))
    df["log1p_revenue"] = np.log1p(revenue.clip(lower=0))

    meetings = _numeric(df, "Meetings_Held__c")
    proposal = _numeric(df, "Proposal_Sent__c").astype(int)
    demo = _numeric(df, "Demo_Requested__c").astype(int)

    df["meet_x_proposal"] = meetings * proposal
    df["engagement_composite"] = (
        web * 1.0
        + _numeric(df, "Email_Opens__c") * 1.5
        + _numeric(df, "Email_Clicks__c") * 3.0
        + _numeric(df, "Form_Submissions__c") * 4.0
        + _numeric(df, "Content_Downloads__c") * 2.0
        + _numeric(df, "Chatbot_Interactions__c") * 1.0
        + demo * 8.0
        + meetings * 5.0
    )
    df["fast_response_flag"] = (ttfr < 4).astype(int)
    df["email_quality"] = _numeric(df, "Email_Open_Rate__c") * _numeric(df, "Email_Clicks__c")
    df["recency_tier"] = pd.cut(
        days_since.clip(lower=0, upper=9999),
        bins=[-1, 3, 14, 30, 90, 9999],
        labels=[4, 3, 2, 1, 0],
    ).astype(int)
    pipeline_days = _numeric(df, "Days_in_Pipeline__c").clip(lower=0)
    df["pipeline_maturity"] = np.log1p(pipeline_days)
    df["activity_density"] = df["engagement_composite"] / (
        pipeline_days + 1
    )

    df["rating_encoded"] = df.get(
        "Rating",
        pd.Series("Warm", index=df.index, dtype=object),
    ).map({"Hot": 2, "Warm": 1, "Cold": 0}).fillna(1).astype(int)
    df["meetings_capped"] = meetings.clip(upper=4)
    df["rating_x_meetings"] = df["rating_encoded"] * df["meetings_capped"]
    df["total_calls_and_meets"] = _numeric(df, "Calls_Made__c") + meetings * 2

    return df
