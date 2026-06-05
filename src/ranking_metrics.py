"""
ranking_metrics.py
==================
Business-facing evaluation for lead *prioritisation*.

Lead scoring is a ranking problem: a sales team works the top-K leads, so the
metric that matters is "how many real converters do we reach for a fixed amount
of outreach", not raw ROC-AUC. This module turns held-out predictions into the
KPIs a revenue team actually acts on:

  * decile lift      — precision and lift within each 10% score band
  * capture@K        — share of all converters found in the top-K% of leads
  * head-to-head     — the AI model vs the rule-based baseline on the same set

These are reporting-only helpers: they never touch the scoring path, so adding
them changes no served prediction. They exist so the claim "ML beats the rules"
is measured in money terms (converters reached per call made) and tracked over
time in metrics.json instead of being asserted from AUC alone.
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
    """
    Rank leads best→worst and report precision/lift per equal-sized band.

    `lift` is band precision divided by the overall positive rate: lift=3.0 in
    the top band means those leads convert 3x more often than a random lead.
    Ties are broken by a stable sort, so equal scores never inflate a band.
    """
    y, s = _as_arrays(y_true, y_score)
    base_rate = y.mean()
    order = np.argsort(-s, kind="stable")
    y_sorted = y[order]
    # np.array_split tolerates lengths that are not divisible by n_bands.
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
    """
    For each top-K fraction, report precision, lift, and the share of all
    converters captured — the "if we only call the top K%, how many real
    opportunities do we reach" curve.
    """
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
    """
    Full prioritisation report for metrics.json. When a baseline score is given,
    emit the head-to-head delta — extra converters the AI reaches over the rules
    at each top-K budget, which is the number a sales lead understands directly.
    """
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
