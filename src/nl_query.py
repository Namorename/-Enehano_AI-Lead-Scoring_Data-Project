"""
nl_query.py
===========
Lightweight rule-based NL query engine.
 
Column contract:
  - DIMENSIONS maps "Region" → the 'Region' column (alias set by load_data())
  - suggestions() uses df["Region"] — will work because of the alias
"""
from __future__ import annotations
 
import re
from dataclasses import dataclass, field
from typing import Optional
 
import pandas as pd
 
METRICS = {
    "win_rate":       ["win rate", "winrate", "win %", "win percentage", "conversion rate",
                       "convert rate", "win"],
    "ai_score":       ["ai score", "average score", "avg score", "score", "lead score"],
    "count":          ["count", "number of leads", "how many", "leads"],
    "expected_wins":  ["expected wins", "wins", "deals won", "expected deals"],
    "high_pct":       ["high potential %", "high potential share", "share of high",
                       "% high", "percent high"],
}
 
DIMENSIONS = {
    "Industry":   ["industry", "industries", "sector", "sectors"],
    "Region":     ["region", "regions", "kraj", "city", "cities"],  # matches alias col
    "LeadSource": ["source", "sources", "lead source", "channel", "channels"],
    "Size_Band":  ["size", "size band", "company size"],
    "Rating":     ["rating", "ratings"],
    "Segment":    ["segment", "segments", "potential", "tier", "tiers"],
}
 
SHORTCUTS = {
    "top_leads":     ["top leads", "best leads", "hottest leads"],
    "best_region":   ["best region", "best kraj", "top region", "which region"],
    "best_industry": ["best industry", "top industry", "which industry"],
    "best_source":   ["best source", "top source", "which source", "best channel"],
}
 
 
@dataclass
class Intent:
    metric:    str = "count"
    dimension: Optional[str] = None
    filters:   dict[str, str] = field(default_factory=dict)
    shortcut:  Optional[str]  = None
    top_n:     int = 10
 
 
def _find(text: str, vocab: dict[str, list[str]]) -> Optional[str]:
    best_key, best_len = None, 0
    for key, phrases in vocab.items():
        for phrase in phrases:
            if phrase in text and len(phrase) > best_len:
                best_key, best_len = key, len(phrase)
    return best_key
 
 
def _resolve_filter(text: str, df: pd.DataFrame, col: str) -> Optional[str]:
    if col not in df.columns:
        return None
    values = sorted({str(v) for v in df[col].dropna().unique()}, key=len, reverse=True)
    for v in values:
        if re.search(rf"\b{re.escape(v.lower())}\b", text):
            return v
    for v in values:
        for tok in re.findall(r"[a-zA-Záčďéěíňóřšťúůýž]{4,}", v.lower()):
            if re.search(rf"\b{re.escape(tok)}\b", text):
                return v
    return None
 
 
def parse(question: str, df: pd.DataFrame) -> Optional[Intent]:
    if not question or not question.strip():
        return None
    text   = question.lower().strip()
    intent = Intent()
 
    m = re.search(r"top\s+(\d{1,3})", text)
    if m:
        intent.top_n = max(1, min(int(m.group(1)), 50))
 
    if re.search(r"top\s+\d+\s+leads?", text):
        intent.shortcut = "top_leads"
    else:
        intent.shortcut = _find(text, SHORTCUTS)
 
    if intent.shortcut:
        for col in DIMENSIONS:
            v = _resolve_filter(text, df, col)
            if v:
                intent.filters[col] = v
        return intent
 
    metric = _find(text, METRICS)
    dim    = _find(text, DIMENSIONS)
    if metric is None and dim is None:
        return None
    if metric:
        intent.metric = metric
    intent.dimension = dim
 
    for col in DIMENSIONS:
        if col == intent.dimension:
            continue
        v = _resolve_filter(text, df, col)
        if v:
            intent.filters[col] = v
 
    return intent
 
 
def suggestions(df: pd.DataFrame) -> list[str]:
    industry = df["Industry"].mode().iloc[0] if "Industry" in df else "Banking"
    region   = df["Region"].mode().iloc[0]   if "Region"   in df else "Praha"
    return [
        f"Win rate by industry in {region}",
        "Average AI score by region",
        f"Number of leads by source for {industry}",
        f"Best region for {industry}",
        "Top 10 leads",
        "Expected wins by segment",
    ]