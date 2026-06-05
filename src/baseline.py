"""
baseline.py
===========
Rule-based lead scoring used as a benchmark for the AI model.
 
The rules encode the kind of heuristics a sales manager would write by hand:
fast follow-up, high engagement, demo / meeting activity, attractive industry.
Output is a 0-100 score and a categorical segment (High / Medium / Low).
"""
from __future__ import annotations
 
import numpy as np
import pandas as pd

from contracts import score_segment
 
# Sectors with the strongest historical fit for Salesforce consulting.
_HIGH_VALUE_SECTORS   = {"J", "M"}
_MEDIUM_VALUE_SECTORS = {"K"}
 
# Lead source weights.
_SOURCE_POINTS = {"Partner": 15, "Event": 12, "Web": 8, "Cold Call": 0}
 
 
def score_row(row: pd.Series) -> int:
    """Return a 0-100 rule-based score for a single lead."""
    score = 0
 
    ttfr = float(row.get("Time_to_First_Response_h__c", 24))
    if ttfr < 1:    score += 25
    elif ttfr < 6:  score += 18
    elif ttfr < 24: score += 8
 
    web = float(row.get("Web_Interactions__c", 0))
    if web > 30:   score += 20
    elif web > 10: score += 10
 
    if int(row.get("Meetings_Held__c", 0))  > 0: score += 15
    if int(row.get("Form_Submissions__c", 0)) > 0: score += 8
    if int(row.get("Demo_Requested__c", 0)) == 1: score += 10
    if int(row.get("LinkedIn_Viewed__c",  0)) == 1: score += 4
 
    sec = str(row.get("CZ_NACE_Section", ""))
    if sec in _HIGH_VALUE_SECTORS:   score += 10
    elif sec in _MEDIUM_VALUE_SECTORS: score += 5
 
    score += _SOURCE_POINTS.get(str(row.get("LeadSource", "")), 0)
 
    return int(np.clip(score, 0, 100))
 
 
def score_frame(df: pd.DataFrame) -> pd.Series:
    """Return rule-based scores for a whole DataFrame using vectorized rules."""
    index = df.index

    def _num(name: str, default: float = 0.0) -> pd.Series:
        if name not in df.columns:
            return pd.Series(default, index=index, dtype=float)
        return pd.to_numeric(df[name], errors="coerce")

    def _text(name: str, default: str = "") -> pd.Series:
        if name not in df.columns:
            return pd.Series(default, index=index, dtype=object)
        return df[name].astype(str)

    score = pd.Series(0, index=index, dtype=float)

    ttfr = _num("Time_to_First_Response_h__c", 24)
    score += np.select([ttfr < 1, ttfr < 6, ttfr < 24], [25, 18, 8], default=0)

    web = _num("Web_Interactions__c", 0)
    score += np.select([web > 30, web > 10], [20, 10], default=0)

    score += (_num("Meetings_Held__c", 0) > 0).astype(int) * 15
    score += (_num("Form_Submissions__c", 0) > 0).astype(int) * 8
    score += (_num("Demo_Requested__c", 0) == 1).astype(int) * 10
    score += (_num("LinkedIn_Viewed__c", 0) == 1).astype(int) * 4

    section = _text("CZ_NACE_Section")
    score += section.isin(_HIGH_VALUE_SECTORS).astype(int) * 10
    score += section.isin(_MEDIUM_VALUE_SECTORS).astype(int) * 5

    score += _text("LeadSource").map(_SOURCE_POINTS).fillna(0)

    return pd.Series(np.clip(score, 0, 100), index=index).astype(int)
 
 
def segment(score: int) -> str:
    """Map a 0-100 score to a High / Medium / Low segment."""
    return score_segment(score)
 
 
def predict_proba(df: pd.DataFrame) -> np.ndarray:
    """Return a [P(not_convert), P(convert)] matrix on the 0-1 scale.
 
    Used so the baseline can be compared to sklearn classifiers via AUC.
    """
    p = score_frame(df).to_numpy() / 100.0
    return np.column_stack([1 - p, p])
