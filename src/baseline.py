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
    """Vectorised wrapper around `score_row` for a whole DataFrame."""
    return df.apply(score_row, axis=1).astype(int)
 
 
def segment(score: int) -> str:
    """Map a 0-100 score to a High / Medium / Low segment."""
    if score >= 70: return "High"
    if score >= 40: return "Medium"
    return "Low"
 
 
def predict_proba(df: pd.DataFrame) -> np.ndarray:
    """Return a [P(not_convert), P(convert)] matrix on the 0-1 scale.
 
    Used so the baseline can be compared to sklearn classifiers via AUC.
    """
    p = score_frame(df).to_numpy() / 100.0
    return np.column_stack([1 - p, p])