"""
scorer.py
=========
Embedded scoring module — runs the trained models in-process without a
separate FastAPI server.

Used when SCORING_API_URL is unset or set to "embedded", e.g. when
deploying to Streamlit Community Cloud as a single self-contained app.

The public surface is two functions:
  score_dataframe(df)     — score a full DataFrame, returns list[dict]
  score_lead(lead_dict)   — score one lead from a flat dict, returns dict

Models are loaded lazily on first call and cached for the process lifetime.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import baseline
from contracts import add_region_alias, score_segment
from feature_engineering import add_engineered_features
from paths import (
    CONV_MODEL, CONV_PRE, CONV_FEAT, CONV_EXPL,
    WIN_MODEL, WIN_PRE, WIN_EXPL,
    WIN_THRESHOLD,
)

_ASSETS: dict = {}


def _ensure_loaded() -> None:
    if _ASSETS:
        return
    import joblib
    _ASSETS["conv_model"] = joblib.load(CONV_MODEL)
    _ASSETS["conv_pre"]   = joblib.load(CONV_PRE)
    _ASSETS["feat_names"] = joblib.load(CONV_FEAT)
    _ASSETS["win_model"]  = joblib.load(WIN_MODEL)
    _ASSETS["win_pre"]    = joblib.load(WIN_PRE)
    if CONV_EXPL.exists():
        try:
            _ASSETS["conv_expl"] = joblib.load(CONV_EXPL)
        except Exception:
            pass
    if WIN_THRESHOLD.exists():
        payload = json.loads(WIN_THRESHOLD.read_text())
        _ASSETS["win_threshold"] = float(payload.get("optimal_win_threshold", 0.5))


def _positive_shap_row(shap_values, row_idx: int) -> np.ndarray:
    if isinstance(shap_values, list):
        values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        return np.asarray(values)[row_idx]
    values = np.asarray(shap_values)
    if values.ndim == 3:
        class_idx = 1 if values.shape[2] > 1 else 0
        return values[row_idx, :, class_idx]
    if values.ndim == 2:
        return values[row_idx]
    raise ValueError(f"Unsupported SHAP values shape: {values.shape}")


def score_dataframe(df: pd.DataFrame, include_drivers: bool = False) -> list[dict]:
    """Score a DataFrame of leads in-process. Returns list of score dicts."""
    if len(df) == 0:
        return []

    _ensure_loaded()

    df = add_region_alias(df.copy())
    add_engineered_features(df)

    X_conv = _ASSETS["conv_pre"].transform(df)
    X_win  = _ASSETS["win_pre"].transform(df)
    p_conv = _ASSETS["conv_model"].predict_proba(X_conv)[:, 1]
    p_win  = _ASSETS["win_model"].predict_proba(X_win)[:, 1]
    rule   = baseline.score_frame(df)

    shap_vals  = None
    explainer  = _ASSETS.get("conv_expl")
    feat_names = _ASSETS["feat_names"]
    if include_drivers and explainer is not None:
        try:
            shap_vals = explainer.shap_values(X_conv, check_additivity=False)
        except Exception as exc:
            print(f"[WARN] SHAP drivers unavailable: {type(exc).__name__}: {exc}")

    out = []
    for i in range(len(df)):
        ai = int(np.clip(round(float(p_conv[i] * 100)), 0, 100))

        drivers: list[str] = []
        if shap_vals is not None:
            sv_row     = _positive_shap_row(shap_vals, i)
            ranked     = list(np.argsort(sv_row)[::-1])
            positive   = [j for j in ranked if sv_row[j] > 0]
            remaining  = [j for j in ranked if j not in positive]
            top_idx    = (positive + remaining)[:3]
            drivers    = [feat_names[j].split("__")[0].replace("_", " ") for j in top_idx]

        rule_val = int(rule.iloc[i]) if hasattr(rule, "iloc") else int(rule[i])
        out.append({
            "ai_score":               ai,
            "segment":                score_segment(ai),
            "expected_win_pct":       round(float(p_conv[i] * p_win[i] * 100), 1),
            "p_win_if_converted_pct": round(float(p_win[i] * 100), 1),
            "rule_score":             rule_val,
            "top_drivers":            drivers,
        })
    return out


def score_lead(lead_dict: dict) -> dict:
    """Score one lead from a flat dict of field values."""
    results = score_dataframe(pd.DataFrame([lead_dict]), include_drivers=True)
    return results[0] if results else {}