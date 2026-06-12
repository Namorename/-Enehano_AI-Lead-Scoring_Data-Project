
"""
outcomes.py
===========
Thread-safe read/write helpers for the two production feedback files:

  data/score_log.csv    — every scoring call (lead_id, score, timestamp)
  data/real_outcomes.csv — actual Salesforce closes fed back via /feedback

These two files are the bridge between demo and product.  The score log lets
you audit every prediction ever made.  The real-outcomes file feeds back into
train.py so the model improves as real deals close instead of learning only
from synthetic labels forever.

Design notes
------------
- Both files are append-only CSV.  Writes use a threading.Lock so concurrent
  FastAPI workers do not interleave partial rows.
- Reads return pandas DataFrames; callers decide what to do with them.
- No database dependency — just files, so the feedback loop works out of the
  box with zero extra infrastructure.
- When data/real_outcomes.csv has >= RETRAIN_THRESHOLD rows newer than the
  last retrain, scripts/auto_retrain.py will trigger a fresh train.py run.
"""
from __future__ import annotations

import csv
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from paths import OUTCOMES_CSV, SCORE_LOG_CSV, RETRAIN_STATE

# Minimum new real outcomes before auto-retrain is triggered.
RETRAIN_THRESHOLD = 200

_score_lock   = threading.Lock()
_outcome_lock = threading.Lock()

# ── Score log ─────────────────────────────────────────────────────────────────

_SCORE_COLS = [
    "timestamp", "lead_id", "source_endpoint",
    "ai_score", "segment", "expected_win_pct", "p_win_if_converted_pct",
    "rule_score",
]


def _ensure_score_log() -> None:
    if not SCORE_LOG_CSV.exists():
        SCORE_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(SCORE_LOG_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=_SCORE_COLS).writeheader()


def log_score(
    *,
    lead_id: str | None,
    source_endpoint: str,
    ai_score: float,
    segment: str,
    expected_win_pct: float,
    p_win_if_converted_pct: float,
    rule_score: int,
) -> None:
    """Append one row to data/score_log.csv (thread-safe)."""
    row = {
        "timestamp":              datetime.now(timezone.utc).isoformat(),
        "lead_id":                lead_id or "",
        "source_endpoint":        source_endpoint,
        "ai_score":               round(ai_score, 2),
        "segment":                segment,
        "expected_win_pct":       round(expected_win_pct, 2),
        "p_win_if_converted_pct": round(p_win_if_converted_pct, 2),
        "rule_score":             rule_score,
    }
    with _score_lock:
        _ensure_score_log()
        with open(SCORE_LOG_CSV, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=_SCORE_COLS).writerow(row)


def read_score_log() -> pd.DataFrame:
    if not SCORE_LOG_CSV.exists():
        return pd.DataFrame(columns=_SCORE_COLS)
    return pd.read_csv(SCORE_LOG_CSV, parse_dates=["timestamp"])


# ── Real outcomes ─────────────────────────────────────────────────────────────

_OUTCOME_COLS = [
    "received_at", "lead_id", "converted", "closed_won",
    "closed_date", "source",
]


def _ensure_outcomes() -> None:
    if not OUTCOMES_CSV.exists():
        OUTCOMES_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTCOMES_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=_OUTCOME_COLS).writeheader()


def record_outcome(
    *,
    lead_id: str,
    converted: bool,
    closed_won: bool | None,
    closed_date: str | None,
    source: str = "api",
) -> None:
    """Append one real outcome to data/real_outcomes.csv (thread-safe)."""
    row = {
        "received_at": datetime.now(timezone.utc).isoformat(),
        "lead_id":     lead_id,
        "converted":   int(converted),
        "closed_won":  int(closed_won) if closed_won is not None else "",
        "closed_date": closed_date or "",
        "source":      source,
    }
    with _outcome_lock:
        _ensure_outcomes()
        with open(OUTCOMES_CSV, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=_OUTCOME_COLS).writerow(row)


def read_outcomes() -> pd.DataFrame:
    if not OUTCOMES_CSV.exists():
        return pd.DataFrame(columns=_OUTCOME_COLS)
    return pd.read_csv(OUTCOMES_CSV)


def outcome_stats() -> dict:
    """Summary consumed by GET /feedback/stats and auto_retrain.py."""
    df = read_outcomes()
    if df.empty:
        return {
            "total_real_outcomes": 0,
            "converted_count":     0,
            "won_count":           0,
            "retrain_threshold":   RETRAIN_THRESHOLD,
            "ready_for_retrain":   False,
            "last_retrain":        _read_retrain_state().get("last_retrain"),
        }
    state = _read_retrain_state()
    last_retrain = state.get("last_retrain")
    new_since_retrain = (
        len(df[df["received_at"] > last_retrain])
        if last_retrain
        else len(df)
    )
    return {
        "total_real_outcomes":   int(len(df)),
        "converted_count":       int(df["converted"].sum()),
        "won_count":             int(df["closed_won"].replace("", 0).fillna(0).astype(int).sum()),
        "new_since_last_retrain": new_since_retrain,
        "retrain_threshold":     RETRAIN_THRESHOLD,
        "ready_for_retrain":     new_since_retrain >= RETRAIN_THRESHOLD,
        "last_retrain":          last_retrain,
    }


# ── Retrain state ─────────────────────────────────────────────────────────────

def _read_retrain_state() -> dict:
    if not RETRAIN_STATE.exists():
        return {}
    try:
        return json.loads(RETRAIN_STATE.read_text())
    except Exception:
        return {}


def mark_retrain_complete(model_auc: float | None = None) -> None:
    state = {
        "last_retrain": datetime.now(timezone.utc).isoformat(),
        "model_auc":    model_auc,
    }
    RETRAIN_STATE.parent.mkdir(parents=True, exist_ok=True)
    RETRAIN_STATE.write_text(json.dumps(state, indent=2))
