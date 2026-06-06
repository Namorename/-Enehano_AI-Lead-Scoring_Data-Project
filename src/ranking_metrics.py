"""
Ranking metrics for lead prioritisation.

A sales team works the top of the list, so what matters is how many real
converters land in the top K% of the ranking, not just ROC-AUC. These helpers
turn held-out predictions into decile lift, capture@K and an AI-vs-rules
comparison, which get written into metrics.json. Reporting only; they don't
affect any score the service returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _as_arrays(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(float)
    s = np.asarray(y_score).astype(float)
    if y.shape != s.shape:
        raise ValueError(f"y_true {y.shape} and y_score {s.shape} must align.")
    if len(y) == 0:
        raise ValueError("Cannot compute ranking metrics on an empty set.")
    return y, s


def decile_lift(y_true, y_score, n_bands: int = 10) -> pd.DataFrame:
    """Rank leads high to low and report precision and lift for each equal band.

    Lift is the band's precision over the overall positive rate: 3.0 in the top
    band means those leads convert 3x more often than average. A stable sort
    keeps ties from inflating a band.
    """
    y, s = _as_arrays(y_true, y_score)
    base_rate = y.mean()
    order = np.argsort(-s, kind="stable")
    y_sorted = y[order]
    # array_split handles counts that don't divide evenly into n_bands.
    bands = np.array_split(y_sorted, n_bands)

    rows = []
    seen_pos = 0
    total_pos = y.sum()
    for i, band in enumerate(bands, start=1):
        precision = float(band.mean()) if len(band) else 0.0
        seen_pos += band.sum()
        rows.append({
            "decile": i,
            "n": int(len(band)),
            "precision": round(precision, 4),
            "lift": round(precision / base_rate, 3) if base_rate else 0.0,
            "cumulative_capture": round(seen_pos / total_pos, 4) if total_pos else 0.0,
        })
    return pd.DataFrame(rows)


def capture_at_k(y_true, y_score, k_fractions=(0.05, 0.10, 0.20, 0.30)) -> dict:
    """For each top-K fraction, report precision, lift and the share of all
    converters that fall in that top slice."""
    y, s = _as_arrays(y_true, y_score)
    base_rate = y.mean()
    total_pos = y.sum()
    order = np.argsort(-s, kind="stable")
    y_sorted = y[order]
    n = len(y_sorted)

    out = {}
    for frac in k_fractions:
        k = max(1, int(round(n * frac)))
        top = y_sorted[:k]
        precision = float(top.mean())
        out[f"top_{int(frac * 100)}pct"] = {
            "k": int(k),
            "precision": round(precision, 4),
            "lift": round(precision / base_rate, 3) if base_rate else 0.0,
            "capture": round(top.sum() / total_pos, 4) if total_pos else 0.0,
        }
    return out


def ranking_report(
    y_true,
    ai_score,
    baseline_score=None,
    k_fractions=(0.05, 0.10, 0.20, 0.30),
) -> dict:
    """Build the prioritisation block for metrics.json. With a baseline score it
    also reports the head-to-head gap: how many more converters the AI reaches at
    each top-K cut-off."""
    y, ai = _as_arrays(y_true, ai_score)
    report: dict = {
        "positive_rate": round(float(y.mean()), 4),
        "n": int(len(y)),
        "ai": {
            "decile_lift": decile_lift(y, ai).to_dict(orient="records"),
            "capture_at_k": capture_at_k(y, ai, k_fractions),
        },
    }
    if baseline_score is not None:
        _, bl = _as_arrays(y_true, baseline_score)
        report["baseline"] = {
            "decile_lift": decile_lift(y, bl).to_dict(orient="records"),
            "capture_at_k": capture_at_k(y, bl, k_fractions),
        }
        head_to_head = {}
        ai_cap = report["ai"]["capture_at_k"]
        bl_cap = report["baseline"]["capture_at_k"]
        total_pos = float(y.sum())
        for key in ai_cap:
            ai_c, bl_c = ai_cap[key]["capture"], bl_cap[key]["capture"]
            head_to_head[key] = {
                "ai_capture": ai_c,
                "baseline_capture": bl_c,
                "capture_uplift": round(ai_c - bl_c, 4),
                "extra_converters_reached": int(round((ai_c - bl_c) * total_pos)),
                "relative_uplift_pct": round((ai_c - bl_c) / bl_c * 100, 1) if bl_c else None,
            }
        report["head_to_head"] = head_to_head
    return report
