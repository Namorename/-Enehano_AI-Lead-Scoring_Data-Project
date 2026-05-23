"""
business_helpers.py
===================
Pure-Python helpers for the Streamlit dashboard:
  - threshold-based confusion stats
  - lift / cumulative-gain table
  - ROI (revenue-uplift) calculator
  - plain-English copy generators

Kept independent of Streamlit so it's easy to unit-test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


# ---------- Curves ----------
def roc_points(y_true: np.ndarray, y_proba: np.ndarray) -> pd.DataFrame:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr})


def pr_points(y_true: np.ndarray, y_proba: np.ndarray) -> pd.DataFrame:
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    return pd.DataFrame({"recall": rec, "precision": prec})


def auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_proba))


# ---------- Lift / Gain ----------
def lift_table(y_true: np.ndarray, y_proba: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Cumulative gain table: top-k% of leads -> % of conversions captured."""
    df = pd.DataFrame({"y": y_true, "p": y_proba}).sort_values("p", ascending=False).reset_index(drop=True)
    n, total_pos = len(df), int(df["y"].sum())
    base_rate = total_pos / n if n else 0
    rows = []
    for b in range(1, bins + 1):
        k = int(np.ceil(n * b / bins))
        captured = int(df.iloc[:k]["y"].sum())
        rows.append({
            "top_pct": b * 100 // bins,
            "leads_worked": k,
            "conversions_captured": captured,
            "cum_recall": captured / total_pos if total_pos else 0,
            "precision_in_bucket": captured / k if k else 0,
            "lift": (captured / k) / base_rate if base_rate else 0,
        })
    return pd.DataFrame(rows)


# ---------- Threshold-based stats ----------
@dataclass
class ThresholdStats:
    threshold: float
    leads_to_work: int
    leads_to_work_pct: float
    captured_conversions: int
    captured_pct: float
    precision: float
    recall: float


def threshold_stats(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> ThresholdStats:
    pred = (y_proba >= threshold).astype(int)
    n, total_pos = len(y_true), int(np.sum(y_true))
    work = int(pred.sum())
    captured = int(((pred == 1) & (y_true == 1)).sum())
    return ThresholdStats(
        threshold=threshold,
        leads_to_work=work,
        leads_to_work_pct=work / n if n else 0,
        captured_conversions=captured,
        captured_pct=captured / total_pos if total_pos else 0,
        precision=captured / work if work else 0,
        recall=captured / total_pos if total_pos else 0,
    )


# ---------- ROI calculator ----------
@dataclass
class ROIInputs:
    pipeline_size: int        # leads in the funnel per period
    avg_deal_value: float     # CZK or EUR per won deal
    win_rate_if_converted: float  # share of converted leads that close-won
    rep_capacity: int         # leads each rep can handle per period
    rep_cost: float           # cost per rep per period
    ai_top_pct: float         # AI prioritisation: work top X% of leads


@dataclass
class ROIResult:
    leads_worked: int
    revenue_random: float
    revenue_baseline: float
    revenue_ai: float
    uplift_vs_baseline: float
    uplift_vs_random: float
    reps_needed: int
    reps_cost: float
    roi_ratio_vs_baseline: float


def compute_roi(inputs: ROIInputs, lift_ai: pd.DataFrame, lift_base: pd.DataFrame,
                base_conversion_rate: float) -> ROIResult:
    """Estimate revenue under three strategies for the same lead budget.

    `lift_ai` and `lift_base` are cumulative-gain tables; we look up the
    `top_pct` bucket closest to `inputs.ai_top_pct`.
    """
    pct = max(1, min(100, int(inputs.ai_top_pct)))
    leads_worked = int(round(inputs.pipeline_size * pct / 100))

    def _recall_at(table: pd.DataFrame) -> float:
        idx = (table["top_pct"] - pct).abs().idxmin()
        return float(table.loc[idx, "cum_recall"])

    total_conv = inputs.pipeline_size * base_conversion_rate
    deal_value = inputs.avg_deal_value * inputs.win_rate_if_converted

    rev_random = leads_worked * base_conversion_rate * deal_value
    rev_base = total_conv * _recall_at(lift_base) * deal_value
    rev_ai = total_conv * _recall_at(lift_ai) * deal_value

    reps_needed = max(1, int(np.ceil(leads_worked / max(1, inputs.rep_capacity))))
    reps_cost = reps_needed * inputs.rep_cost
    roi_ratio = (rev_ai - reps_cost) / max(1.0, reps_cost)

    return ROIResult(
        leads_worked=leads_worked,
        revenue_random=rev_random,
        revenue_baseline=rev_base,
        revenue_ai=rev_ai,
        uplift_vs_baseline=rev_ai - rev_base,
        uplift_vs_random=rev_ai - rev_random,
        reps_needed=reps_needed,
        reps_cost=reps_cost,
        roi_ratio_vs_baseline=roi_ratio,
    )


# ---------- Plain-language helpers ----------
def explain_segment(seg: str) -> str:
    return {
        "High": "Call today — strong intent and good fit.",
        "Medium": "Worth a touch — nurture with content / quick email.",
        "Low": "Park or automate — unlikely to convert in this cycle.",
    }.get(seg, "")



# ---------- Drift / stability ----------
def population_stability_index(reference: np.ndarray, current: np.ndarray,
                               bins: int = 10) -> float:
    """Population Stability Index between two score distributions.

    Industry rule of thumb:
      PSI < 0.10 → no significant change ("stable")
      0.10–0.25 → moderate shift, investigate
      > 0.25    → major shift, retrain
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    edges = np.linspace(min(ref.min(), cur.min()), max(ref.max(), cur.max()) + 1e-9, bins + 1)
    ref_pct = np.histogram(ref, bins=edges)[0] / max(len(ref), 1)
    cur_pct = np.histogram(cur, bins=edges)[0] / max(len(cur), 1)
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_verdict(psi: float) -> tuple[str, str]:
    """Return (label, hex_color) for a PSI value."""
    if psi < 0.10:
        return "✅ Stable", "#a6ce39"
    if psi < 0.25:
        return "⚠ Moderate shift", "#ffc107"
    return "🚨 Major drift — retrain", "#ff4b4b"

