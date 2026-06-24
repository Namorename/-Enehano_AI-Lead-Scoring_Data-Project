"""Tests for baseline.py — rule-based scoring."""
import numpy as np
import pandas as pd
import pytest

from baseline import predict_proba, score_frame, score_row, segment


# ── score_row ─────────────────────────────────────────────────────────────────

def _row(**kwargs) -> pd.Series:
    defaults = {
        "Time_to_First_Response_h__c": 24,
        "Web_Interactions__c": 0,
        "Meetings_Held__c": 0,
        "Form_Submissions__c": 0,
        "Demo_Requested__c": 0,
        "LinkedIn_Viewed__c": 0,
        "CZ_NACE_Section": "C",
        "LeadSource": "Cold Call",
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


def test_score_row_bounds_empty_lead():
    score = score_row(_row())
    assert 0 <= score <= 100


def test_score_row_max_lead():
    lead = _row(
        Time_to_First_Response_h__c=0.5,
        Web_Interactions__c=50,
        Meetings_Held__c=2,
        Form_Submissions__c=1,
        Demo_Requested__c=1,
        LinkedIn_Viewed__c=1,
        CZ_NACE_Section="J",
        LeadSource="Partner",
    )
    score = score_row(lead)
    assert score == 100


def test_score_row_fast_response_lt1h():
    s1 = score_row(_row(Time_to_First_Response_h__c=0.5))
    s2 = score_row(_row(Time_to_First_Response_h__c=5.0))
    assert s1 > s2


def test_score_row_high_web_interactions():
    s_high = score_row(_row(Web_Interactions__c=35))
    s_low  = score_row(_row(Web_Interactions__c=2))
    assert s_high > s_low


def test_score_row_nace_sector_bonus():
    s_j = score_row(_row(CZ_NACE_Section="J"))
    s_c = score_row(_row(CZ_NACE_Section="C"))
    assert s_j > s_c


def test_score_row_lead_source_partner():
    s_partner = score_row(_row(LeadSource="Partner"))
    s_cold    = score_row(_row(LeadSource="Cold Call"))
    assert s_partner > s_cold


def test_score_row_returns_int():
    assert isinstance(score_row(_row()), int)


# ── score_frame ────────────────────────────────────────────────────────────────

def _frame(n: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "Time_to_First_Response_h__c": rng.uniform(0, 48, n),
        "Web_Interactions__c":         rng.integers(0, 50, n).astype(float),
        "Meetings_Held__c":            rng.integers(0, 4, n).astype(float),
        "Form_Submissions__c":         rng.integers(0, 3, n).astype(float),
        "Demo_Requested__c":           rng.integers(0, 2, n).astype(int),
        "LinkedIn_Viewed__c":          rng.integers(0, 2, n).astype(int),
        "CZ_NACE_Section":             rng.choice(["J", "K", "M", "C"], n),
        "LeadSource":                  rng.choice(["Web", "Partner", "Event"], n),
    })


def test_score_frame_length():
    df = _frame(10)
    scores = score_frame(df)
    assert len(scores) == 10


def test_score_frame_bounds():
    df = _frame(50)
    scores = score_frame(df)
    assert (scores >= 0).all()
    assert (scores <= 100).all()


def test_score_frame_matches_score_row():
    df = _frame(8)
    frame_scores = score_frame(df).tolist()
    row_scores   = [score_row(df.iloc[i]) for i in range(len(df))]
    assert frame_scores == row_scores


def test_score_frame_missing_columns():
    df = pd.DataFrame({"CZ_NACE_Section": ["J", "C"]})
    scores = score_frame(df)
    assert len(scores) == 2
    assert (scores >= 0).all()


# ── segment ───────────────────────────────────────────────────────────────────

def test_segment_high():
    assert segment(100) == "High"
    assert segment(70) == "High"


def test_segment_medium():
    assert segment(40) == "Medium"
    assert segment(69) == "Medium"


def test_segment_low():
    assert segment(0) == "Low"
    assert segment(39) == "Low"


# ── predict_proba ──────────────────────────────────────────────────────────────

def test_predict_proba_shape():
    df = _frame(5)
    proba = predict_proba(df)
    assert proba.shape == (5, 2)


def test_predict_proba_sums_to_one():
    df = _frame(5)
    proba = predict_proba(df)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)


def test_predict_proba_bounds():
    df = _frame(20)
    proba = predict_proba(df)
    assert (proba >= 0).all()
    assert (proba <= 1).all()