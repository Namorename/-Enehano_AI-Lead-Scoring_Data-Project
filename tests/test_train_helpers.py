"""
Tests for pure helper functions in train.py.

Imports the private helpers directly; the module-level side-effects
(mkdir, random.seed) are harmless in tests.
"""
import numpy as np
import pandas as pd
import pytest

from train import (
    _build_preprocessor,
    _compute_ece,
    _scores,
    _threshold_tuning,
    _select_candidate,
    _time_holdout_positions,
    _calibration_split_positions,
    _ordered_positions,
    CandidateResult,
    CandidateSpec,
    _make_baseline_gbm,
)


# ── _scores ───────────────────────────────────────────────────────────────────

def test_scores_perfect_classifier():
    y = np.array([1, 1, 0, 0])
    proba = np.array([0.9, 0.8, 0.1, 0.2])
    pred  = (proba >= 0.5).astype(int)
    m = _scores(y, pred, proba)
    assert m["auc_roc"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)


def test_scores_keys_present():
    y = np.array([1, 0, 1, 0, 1])
    proba = np.array([0.7, 0.3, 0.6, 0.4, 0.8])
    pred  = (proba >= 0.5).astype(int)
    m = _scores(y, pred, proba)
    for key in ["auc_roc", "pr_auc", "brier", "precision", "recall",
                "f1", "confusion_matrix", "positive_rate",
                "unique_scores_int", "unique_scores_1dp"]:
        assert key in m, f"Key '{key}' missing from _scores output"


def test_scores_positive_rate():
    y = np.array([1, 1, 0, 0, 0])
    proba = np.array([0.6, 0.7, 0.3, 0.2, 0.1])
    pred  = (proba >= 0.5).astype(int)
    m = _scores(y, pred, proba)
    assert m["positive_rate"] == pytest.approx(0.4)


def test_scores_confusion_matrix_shape():
    y = np.array([0, 1, 0, 1])
    proba = np.array([0.3, 0.7, 0.2, 0.8])
    pred  = (proba >= 0.5).astype(int)
    m = _scores(y, pred, proba)
    cm = m["confusion_matrix"]
    assert len(cm) == 2 and len(cm[0]) == 2


# ── _compute_ece ──────────────────────────────────────────────────────────────

def test_compute_ece_perfect_calibration():
    y = np.array([1, 0, 1, 0])
    proba = np.array([1.0, 0.0, 1.0, 0.0])
    from train import _compute_ece
    ece = _compute_ece(y, proba)
    assert ece == pytest.approx(0.0, abs=1e-6)


def test_compute_ece_range():
    rng = np.random.default_rng(7)
    y     = rng.integers(0, 2, 200)
    proba = rng.uniform(0, 1, 200)
    from train import _compute_ece
    ece = _compute_ece(y, proba)
    assert 0.0 <= ece <= 1.0


def test_compute_ece_custom_bins():
    y = np.array([1, 0] * 10)
    proba = np.linspace(0, 1, 20)
    from train import _compute_ece
    ece_5  = _compute_ece(y, proba, n_bins=5)
    ece_10 = _compute_ece(y, proba, n_bins=10)
    assert ece_5 >= 0 and ece_10 >= 0


# ── _threshold_tuning ─────────────────────────────────────────────────────────

def test_threshold_tuning_returns_tuple():
    y = pd.Series([1, 0, 1, 0, 1, 0])
    proba = np.array([0.8, 0.2, 0.7, 0.3, 0.9, 0.1])
    t, m = _threshold_tuning(y, proba)
    assert isinstance(t, float)
    assert "threshold" in m


def test_threshold_tuning_threshold_in_range():
    y = pd.Series([1, 0, 1, 0, 1, 1, 0, 0])
    proba = np.array([0.9, 0.1, 0.8, 0.2, 0.75, 0.85, 0.15, 0.05])
    t, _ = _threshold_tuning(y, proba)
    assert 0.0 <= t <= 1.0


def test_threshold_tuning_metrics_include_f1():
    y = pd.Series([1, 0, 1, 0])
    proba = np.array([0.8, 0.3, 0.7, 0.2])
    _, m = _threshold_tuning(y, proba)
    assert "f1" in m
    assert "precision" in m
    assert "recall" in m


# ── _build_preprocessor ───────────────────────────────────────────────────────

def _sample_X():
    return pd.DataFrame({
        "age":     [25.0, 30.0, 45.0, 20.0, 50.0],
        "revenue": [100.0, 200.0, 150.0, 300.0, 250.0],
        "sector":  ["J", "M", "K", "J", "C"],
    })


def test_build_preprocessor_returns_three_items():
    X = _sample_X()
    result = _build_preprocessor(X)
    assert len(result) == 3


def test_build_preprocessor_numeric_and_cat_lists():
    X = _sample_X()
    _, num, cat = _build_preprocessor(X)
    assert "age" in num
    assert "revenue" in num
    assert "sector" in cat


def test_build_preprocessor_fit_transform_shape():
    X = _sample_X()
    pre, _, _ = _build_preprocessor(X)
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == len(X)
    # numeric cols scaled + OHE categoricals
    assert Xt.shape[1] > len(X.columns)


# ── _time_holdout_positions ───────────────────────────────────────────────────

def test_time_holdout_no_dates():
    tr, te, cutoff = _time_holdout_positions(None, 10)
    assert len(tr) + len(te) == 10
    assert cutoff is None


def test_time_holdout_with_dates():
    dates = pd.Series(pd.date_range("2022-01-01", periods=20, freq="ME").astype(str))
    tr, te, cutoff = _time_holdout_positions(dates, 20)
    assert len(tr) + len(te) == 20
    assert cutoff is not None


def test_time_holdout_test_size_approx_20pct():
    tr, te, _ = _time_holdout_positions(None, 100)
    assert abs(len(te) - 20) <= 2


# ── _calibration_split_positions ──────────────────────────────────────────────

def test_calibration_split_returns_disjoint():
    train_pos = np.arange(80)
    fit_pos, cal_pos = _calibration_split_positions(train_pos, None, 0.10)
    assert len(set(fit_pos) & set(cal_pos)) == 0


def test_calibration_split_fraction():
    train_pos = np.arange(100)
    fit_pos, cal_pos = _calibration_split_positions(train_pos, None, 0.10)
    assert abs(len(cal_pos) - 10) <= 2


# ── _select_candidate ─────────────────────────────────────────────────────────

def _make_result(name: str, cv_auc_mean: float, cv_folds: list, test_auc: float) -> CandidateResult:
    dummy_spec = CandidateSpec(name, "ohe", _make_baseline_gbm)
    test_metrics = {
        "auc_roc": test_auc, "pr_auc": 0.5, "brier": 0.2,
        "precision": 0.5, "recall": 0.5, "f1": 0.5,
        "confusion_matrix": [[40, 10], [10, 40]],
        "positive_rate": 0.5,
        "unique_scores_int": 50, "unique_scores_1dp": 100,
        "time_split_test_auc": test_auc,
        "split_type": "date_cutoff_80_20",
        "cutoff_date": "2023-06-01",
        "train_size": 80, "test_size": 20,
    }
    return CandidateResult(
        name=name,
        spec=dummy_spec,
        algorithm="GradientBoostingClassifier",
        cv={"auc_mean": cv_auc_mean, "auc_std": 0.01,
            "folds": cv_folds, "cv_type": "time_series_5_fold_gap_30"},
        test=test_metrics,
        cv_auc_random_split={"auc_mean": cv_auc_mean, "auc_std": 0.01, "folds": cv_folds},
        model=None,
        preprocessor=None,
        feature_names=[],
        val_idx=pd.Index([]),
        val_proba=np.array([]),
        val_y=np.array([]),
        val_X=pd.DataFrame(),
        cutoff_date="2023-06-01",
        train_size=80,
        test_size=20,
        optimal_threshold=0.5,
        threshold_test={**test_metrics, "threshold": 0.5},
    )


def test_select_candidate_single_returns_it():
    r = _make_result("logreg", 0.80, [0.78, 0.80, 0.81, 0.79, 0.82], 0.81)
    best, rationale = _select_candidate([r])
    assert best.name == "logreg"


def test_select_candidate_prefers_simpler_within_1se():
    # logreg and lightgbm within 1 SE — logreg is simpler (lower complexity rank)
    logreg = _make_result("logreg",           0.820, [0.81, 0.82, 0.83, 0.82, 0.81], 0.82)
    lgbm   = _make_result("lightgbm_native",  0.825, [0.82, 0.83, 0.84, 0.82, 0.81], 0.83)
    best, rationale = _select_candidate([logreg, lgbm])
    # Both within 1 SE → simpler one (logreg) is preferred
    assert best.name == "logreg"


def test_select_candidate_picks_clearly_better_model():
    logreg = _make_result("logreg",          0.700, [0.68, 0.70, 0.71, 0.69, 0.70], 0.70)
    lgbm   = _make_result("lightgbm_native", 0.900, [0.88, 0.90, 0.91, 0.89, 0.92], 0.91)
    best, _ = _select_candidate([logreg, lgbm])
    assert best.name == "lightgbm_native"


def test_select_candidate_rationale_contains_rule():
    r = _make_result("hist_gbm", 0.85, [0.84, 0.85, 0.86, 0.84, 0.86], 0.85)
    _, rationale = _select_candidate([r])
    assert "rule" in rationale