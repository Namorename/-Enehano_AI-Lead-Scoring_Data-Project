"""
train.py
========
Trains and evaluates the Enehano lead-scoring models.

Two classifiers are trained:
  * conversion model : Lead -> Opportunity  (target = Converted)
  * win model        : Opportunity -> Closed Won  (target = Closed_Won,
                       trained only on rows where Converted == 1)

Evaluation:
  * Time-based 5-fold cross-validation (mean ± std AUC)
  * Time-based held-out test set: AUC-ROC, PR-AUC, precision, recall, F1,
    confusion matrix
  * Side-by-side comparison with the rule-based baseline
  * Top feature importances saved to CSV

Artifacts written:
  model.pkl, preprocessor.pkl, feature_names.pkl,
  conv_explainer.pkl (SHAP explainer for the conversion base model: a
    TreeExplainer for tree models, a LinearExplainer for linear ones),
  win_model.pkl, win_preprocessor.pkl,
  win_explainer.pkl (same idea for the win model; see _build_explainer, or None
    if the model type isn't supported),
  metrics.json, feature_importance.csv, val_predictions.csv

  Note: the served models are CalibratedClassifierCV wrappers, but each explainer
  is built on the base estimator, so SHAP values explain the uncalibrated ranking
  rather than the calibrated probability.

Column contract (matches data_generator.py output):
  ID / audit cols  -> dropped before building X (never seen by the model)
  Leaky cols       -> dropped (Conversion_Probability, Rule_Based_Score/Segment)
  Date strings     -> parsed by feature_engineering.py, then dropped from X
  Bool flags       -> explicitly cast to int (0/1), so StandardScaler handles
                     them correctly without dtype ambiguity
  Legal_Form_Code  -> generator writes it as str, so OHE handles it correctly
"""

import json
import os
import random
import warnings
from dataclasses import dataclass
from typing import Any, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from contracts import (
    HIGH_SCORE_THRESHOLD,
    MODEL_DROP_COLS,
    STATUS_PROXY_COLS,
    TARGET_CONV,
    TARGET_WIN,
    WIN_MODEL_FEATURES,
    model_feature_frame,
    win_feature_frame,
)
import baseline
from feature_engineering import add_engineered_features
from modeling import LightGBMFramePreprocessor
from ranking_metrics import ranking_report
from paths import (
    LEAD_TRAIN,
    MODELS_DIR,
    CONV_MODEL,
    CONV_PRE,
    CONV_FEAT,
    CONV_EXPL,
    WIN_MODEL,
    WIN_PRE,
    WIN_EXPL,
    WIN_THRESHOLD,
    METRICS_JSON,
    FEATURE_IMPORTANCE,
    VAL_PREDICTIONS,
)

try:
    import lightgbm as lgb
except ImportError:  # optional dependency; sklearn candidates still run
    lgb = None

try:
    import optuna
except ImportError:  # optional dependency; tuning is explicitly gated
    optuna = None

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)

# ──────────────────────────────────────────────────────────────────────────────
# PATHS  (resolve relative to this script so it works from any cwd)
# ──────────────────────────────────────────────────────────────────────────────

DATA_PATH = str(LEAD_TRAIN)
RANDOM_STATE = 42
CONVERSION_CALIBRATION_FRACTION = 0.10
# The win model has far fewer rows than the conversion model. A 10% isotonic
# holdout created endpoint overfit on this dataset, so the win calibrator uses
# a larger temporal holdout while still keeping calibration data fully separate.
WIN_CALIBRATION_FRACTION = 0.20
BOOL_COLS = ["Demo_Requested__c", "Proposal_Sent__c", "LinkedIn_Viewed__c"]
LGBM_CATEGORICAL_FEATURES = [
    "CZ_NACE_Section",
    "Legal_Form_Code",
    "Region_Name",
    "LeadSource",
    "Rating",
]
WIN_LGBM_CATEGORICAL_FEATURES = [
    "CZ_NACE_Section",
    "Region_Name",
]
RUN_OPTUNA = os.getenv("RUN_OPTUNA", "0").strip().lower() in {"1", "true", "yes"}
OPTUNA_TRIALS = int(os.getenv("OPTUNA_TRIALS", "50"))

# Leakage control, off by default (keeps the existing artifacts unchanged).
# Set EXCLUDE_STATUS_PROXIES=1 to drop the Status-derived proxy features
# (STATUS_PROXY_COLS); the resulting model holds up better on real CRM data.
EXCLUDE_STATUS_PROXIES = (
    os.getenv("EXCLUDE_STATUS_PROXIES", "0").strip().lower() in {"1", "true", "yes"}
)

# Complexity order for the one-standard-error rule below. When several candidates
# are within one SE of the best CV AUC, we keep the simpler one rather than the
# one that happens to win on the test split. Lower rank = preferred; unknown last.
MODEL_COMPLEXITY_RANK = {
    "logreg": 0,
    "hist_gbm": 1,
    "baseline_gbm": 2,
    "lightgbm_native": 3,
    "stacking_lgbm_hist_logreg": 4,
}

MODELS_DIR.mkdir(exist_ok=True)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ──────────────────────────────────────────────────────────────────────────────
# COLUMN EXCLUSION LISTS
# ──────────────────────────────────────────────────────────────────────────────

# Imported from contracts.py so training, API payloads, and the dashboard share
# one feature boundary.
DROP_COLS = MODEL_DROP_COLS

# IMPORTANT: Days_in_Pipeline__c and Days_Since_Last_Activity__c are partially
# correlated with the excluded Status column (which drives them in the data generator).
# This inflates synthetic AUC. On real CRM data these will behave more independently.
# Monitor feature importance: if they still dominate at >50% combined on real data,
# consider whether Status should be re-introduced as a categorical for pipeline scoring.

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, list, list]:
    """
    Build a ColumnTransformer that scales numeric columns and one-hot-encodes
    categoricals. Booleans stored as int end up in the numeric branch; string
    codes (Legal_Form_Code) end up in the categorical branch.
    """
    num = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    # "str" works in pandas 3+; "object" is the legacy alias - support both
    cat = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ],
        remainder="drop",   # explicitly drop anything not in num or cat
    )
    return pre, num, cat


def _scores(y_true, y_pred, y_proba) -> dict:
    score_0_100 = np.asarray(y_proba) * 100
    return {
        "auc_roc":          float(roc_auc_score(y_true, y_proba)),
        "pr_auc":           float(average_precision_score(y_true, y_proba)),
        "brier":            float(brier_score_loss(y_true, y_proba)),
        "precision":        float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":           float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":               float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "positive_rate":    float(np.mean(y_true)),
        "unique_scores_int": int(pd.Series(np.round(score_0_100).astype(int)).nunique()),
        "unique_scores_1dp": int(pd.Series(np.round(score_0_100, 1)).nunique()),
    }


def _compute_ece(y_true, y_proba, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_proba >= bins[i]) & (y_proba <= bins[i + 1])
        else:
            mask = (y_proba >= bins[i]) & (y_proba < bins[i + 1])
        if mask.sum() > 0:
            bin_accuracy = y_true[mask].mean()
            bin_confidence = y_proba[mask].mean()
            ece += mask.sum() / len(y_true) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def _threshold_tuning(y_true: pd.Series, y_proba: np.ndarray) -> tuple[float, dict]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    if len(thresholds) == 0:
        threshold = 0.5
    else:
        f1_scores = (
            2
            * precision[:-1]
            * recall[:-1]
            / (precision[:-1] + recall[:-1] + 1e-8)
        )
        threshold = float(thresholds[int(np.nanargmax(f1_scores))])
    tuned_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tuned_metrics = _scores(y_true, tuned_pred, y_proba)
    tuned_metrics["threshold"] = threshold
    return threshold, tuned_metrics


def _legacy_cross_val_auc(model_factory, pre_factory, X: pd.DataFrame, y: pd.Series) -> dict:
    """5-fold stratified CV - each fold refits its own preprocessor to avoid leakage."""
    skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr, va in skf.split(X, y):
        pre = pre_factory()
        Xt  = pre.fit_transform(X.iloc[tr])
        Xv  = pre.transform(X.iloc[va])
        m   = model_factory()
        weights = compute_sample_weight(class_weight="balanced", y=y.iloc[tr])
        m.fit(Xt, y.iloc[tr], sample_weight=weights)
        aucs.append(roc_auc_score(y.iloc[va], m.predict_proba(Xv)[:, 1]))
    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std":  float(np.std(aucs)),
        "folds":    [float(a) for a in aucs],
    }


def _train_legacy_gbm(name: str, X: pd.DataFrame, y: pd.Series) -> dict:
    print(f"\n--- Training: {name} (n={len(X):,}, positive rate={y.mean():.1%}) ---")

    # Verify no excluded columns slipped through
    remaining_excluded = [c for c in MODEL_DROP_COLS if c in X.columns]
    if remaining_excluded:
        raise ValueError(f"Excluded columns still in X: {remaining_excluded}")

    def _pre_factory():
        pre, _, _ = _build_preprocessor(X)
        return pre

    def _model_factory():
        return GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.06,
            max_depth=3,
            min_samples_leaf=35,
            subsample=0.85,
            validation_fraction=0.12,
            n_iter_no_change=12,
            tol=1e-4,
            random_state=42,
        )

    cv = _legacy_cross_val_auc(_model_factory, _pre_factory, X, y)
    print(f"  CV AUC: {cv['auc_mean']:.4f} ± {cv['auc_std']:.4f}")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pre, num, cat = _build_preprocessor(Xtr)
    Xtr_p = pre.fit_transform(Xtr)
    Xte_p = pre.transform(Xte)
    train_weights = compute_sample_weight(class_weight="balanced", y=ytr)
    model  = _model_factory().fit(Xtr_p, ytr, sample_weight=train_weights)

    # Build a SHAP TreeExplainer tied to this model. Gradient boosting's
    # shallow additive trees give smoother scores than a RandomForest vote
    # average while keeping single-lead explanations fast.
    explainer = shap.TreeExplainer(model)

    proba  = model.predict_proba(Xte_p)[:, 1]
    pred   = (proba >= 0.5).astype(int)
    test_m = _scores(yte, pred, proba)
    print(f"  Test AUC: {test_m['auc_roc']:.4f}  F1: {test_m['f1']:.4f}")

    # Feature names: numeric first, then OHE-expanded categoricals
    feat_names = num + pre.named_transformers_["cat"].get_feature_names_out(cat).tolist()

    return {
        "algorithm": type(model).__name__,
        "cv": cv, "test": test_m,
        "model": model, "preprocessor": pre, "feature_names": feat_names,
        "explainer": explainer,
        "val_idx": Xte.index, "val_proba": proba, "val_y": yte.values,
        "val_X": Xte,
    }


def _baseline_metrics(df: pd.DataFrame, y: pd.Series) -> dict:
    """
    Rule-based baseline: uses Rule_Based_Score (0-100) normalised to [0,1]
    as the probability proxy. The score column is read from the raw df so it
    is never part of the model feature matrix.
    """
    proba = (df.loc[y.index, "Rule_Based_Score"] / 100).values
    pred  = (proba >= 0.5).astype(int)
    return _scores(y, pred, proba)


@dataclass
class CandidateSpec:
    name: str
    kind: str
    estimator_factory: Any
    use_sample_weight: bool = False
    categorical_features: list[str] | None = None


@dataclass
class CandidateResult:
    name: str
    spec: CandidateSpec
    algorithm: str
    cv: dict
    test: dict
    cv_auc_random_split: dict
    model: Any
    preprocessor: Any
    feature_names: list[str]
    val_idx: pd.Index
    val_proba: np.ndarray
    val_y: np.ndarray
    val_X: pd.DataFrame
    cutoff_date: str | None
    train_size: int
    test_size: int
    optimal_threshold: float
    threshold_test: dict


def _make_baseline_gbm() -> GradientBoostingClassifier:
    """Historical model kept as the direct comparison baseline."""
    return GradientBoostingClassifier(
        n_estimators=180,
        learning_rate=0.06,
        max_depth=3,
        min_samples_leaf=35,
        subsample=0.85,
        validation_fraction=0.12,
        n_iter_no_change=12,
        tol=1e-4,
        random_state=42,
    )


def _make_hist_gbm() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=30,
        l2_regularization=1.0,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
    )


def _make_logreg() -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=500,
        random_state=42,
    )


def _make_lgbm(params: dict | None = None):
    if lgb is None:
        raise RuntimeError("LightGBM is not installed. Install requirements.txt first.")
    base = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 5,
        "num_leaves": 31,
        "min_child_samples": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.5,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
        "verbose": -1,
    }
    if params:
        base.update(params)
    return lgb.LGBMClassifier(**base)


def _make_win_lgbm(params: dict | None = None):
    if lgb is None:
        raise RuntimeError("LightGBM is not installed. Install requirements.txt first.")
    base = {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 3,
        "num_leaves": 15,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if params:
        base.update(params)
    return lgb.LGBMClassifier(**base)


def _make_stacking(model_profile: str = "conversion") -> StackingClassifier:
    if lgb is None:
        raise RuntimeError("LightGBM is not installed. Install requirements.txt first.")
    lgb_base = _make_win_lgbm() if model_profile == "win" else lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    return StackingClassifier(
        estimators=[
            ("lgbm", lgb_base),
            ("hist_gbm", _make_hist_gbm()),
        ],
        final_estimator=_make_logreg(),
        cv=5,
        n_jobs=1,
    )


def _candidate_specs(
    include_stacking: bool = False,
    model_profile: str = "conversion",
) -> list[CandidateSpec]:
    specs = [
        CandidateSpec("baseline_gbm", "ohe", _make_baseline_gbm, use_sample_weight=True),
        CandidateSpec("hist_gbm", "ohe", _make_hist_gbm),
        CandidateSpec("logreg", "ohe", _make_logreg),
    ]
    if lgb is not None:
        lgbm_factory = _make_win_lgbm if model_profile == "win" else _make_lgbm
        cat_features = (
            WIN_LGBM_CATEGORICAL_FEATURES
            if model_profile == "win"
            else LGBM_CATEGORICAL_FEATURES
        )
        specs.insert(
            1,
            CandidateSpec(
                "lightgbm_native",
                "lightgbm",
                lgbm_factory,
                categorical_features=cat_features,
            ),
        )
        if include_stacking:
            specs.append(
                CandidateSpec(
                    "stacking_lgbm_hist_logreg",
                    "ohe",
                    lambda: _make_stacking(model_profile),
                )
            )
    return specs


def _ordered_positions(dates: pd.Series | None, n_rows: int) -> np.ndarray:
    if dates is None:
        return np.arange(n_rows)
    parsed = pd.to_datetime(dates, errors="coerce")
    if parsed.notna().sum() == 0:
        return np.arange(n_rows)
    sortable = parsed.fillna(parsed.max() + pd.Timedelta(days=1))
    return np.argsort(sortable.to_numpy())


def _time_holdout_positions(
    dates: pd.Series | None, n_rows: int, test_size: float = 0.2
) -> tuple[np.ndarray, np.ndarray, pd.Timestamp | None]:
    if dates is None:
        ordered = np.arange(n_rows)
        split = int(np.floor(n_rows * (1 - test_size)))
        split = max(1, min(split, n_rows - 1))
        return ordered[:split], ordered[split:], None

    parsed = pd.to_datetime(dates, errors="coerce")
    if parsed.notna().sum() == 0:
        ordered = np.arange(n_rows)
        split = int(np.floor(n_rows * (1 - test_size)))
        split = max(1, min(split, n_rows - 1))
        return ordered[:split], ordered[split:], None

    sortable = pd.DataFrame({
        "pos": np.arange(n_rows),
        "date": parsed.fillna(parsed.max() + pd.Timedelta(days=1)).to_numpy(),
    }).sort_values("date", kind="mergesort")
    cutoff_idx = int(len(sortable) * (1 - test_size))
    cutoff_idx = max(0, min(cutoff_idx, len(sortable) - 1))
    cutoff_date = pd.Timestamp(sortable["date"].iloc[cutoff_idx])

    train_pos = sortable.loc[sortable["date"] <= cutoff_date, "pos"].to_numpy()
    test_pos = sortable.loc[sortable["date"] > cutoff_date, "pos"].to_numpy()
    if len(test_pos) == 0:
        split = max(1, min(cutoff_idx, n_rows - 1))
        ordered = sortable["pos"].to_numpy()
        train_pos = ordered[:split]
        test_pos = ordered[split:]
    return train_pos, test_pos, cutoff_date


def _calibration_split_positions(
    train_pos: np.ndarray,
    dates: pd.Series | None,
    calibration_fraction: float = CONVERSION_CALIBRATION_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    train_pos = np.asarray(train_pos)
    if len(train_pos) < 20:
        split = max(1, len(train_pos) - 1)
        return train_pos[:split], train_pos[split:]

    if dates is None:
        ordered = train_pos
    else:
        parsed = pd.to_datetime(dates, errors="coerce")
        if parsed.notna().sum() == 0:
            ordered = train_pos
        else:
            sortable = pd.DataFrame({
                "pos": train_pos,
                "date": parsed.iloc[train_pos].fillna(
                    parsed.max() + pd.Timedelta(days=1)
                ).to_numpy(),
            }).sort_values("date", kind="mergesort")
            ordered = sortable["pos"].to_numpy()

    split = int(np.floor(len(ordered) * (1 - calibration_fraction)))
    split = max(1, min(split, len(ordered) - 1))
    return ordered[:split], ordered[split:]


def _candidate_feature_names(
    preprocessor: Any,
    num: list[str] | None,
    cat: list[str] | None,
) -> list[str]:
    if isinstance(preprocessor, LightGBMFramePreprocessor):
        return preprocessor.get_feature_names_out().tolist()
    if num is not None and cat is not None:
        return num + preprocessor.named_transformers_["cat"].get_feature_names_out(cat).tolist()
    return []


def _fit_candidate_once(
    spec: CandidateSpec,
    Xtr: pd.DataFrame,
    ytr: pd.Series,
    Xva: pd.DataFrame,
    yva: pd.Series,
) -> tuple[Any, Any, list[str], np.ndarray]:
    if spec.kind == "lightgbm":
        pre = LightGBMFramePreprocessor(spec.categorical_features or LGBM_CATEGORICAL_FEATURES)
        Xtr_p = pre.fit_transform(Xtr)
        Xva_p = pre.transform(Xva)
        model = spec.estimator_factory()
        model.fit(
            Xtr_p,
            ytr,
            eval_set=[(Xva_p, yva)],
            categorical_feature=pre.categorical_cols_,
        )
        proba = model.predict_proba(Xva_p)[:, 1]
        return pre, model, _candidate_feature_names(pre, None, None), proba

    pre, num, cat = _build_preprocessor(Xtr)
    Xtr_p = pre.fit_transform(Xtr)
    Xva_p = pre.transform(Xva)
    model = spec.estimator_factory()
    fit_kwargs = {}
    if spec.use_sample_weight:
        fit_kwargs["sample_weight"] = compute_sample_weight(
            class_weight="balanced", y=ytr
        )
    model.fit(Xtr_p, ytr, **fit_kwargs)
    proba = model.predict_proba(Xva_p)[:, 1]
    return pre, model, _candidate_feature_names(pre, num, cat), proba


def _fit_candidate_final_base(
    spec: CandidateSpec,
    Xtr: pd.DataFrame,
    ytr: pd.Series,
    dates: pd.Series | None,
) -> tuple[Any, Any, list[str]]:
    if spec.kind == "lightgbm":
        pre = LightGBMFramePreprocessor(spec.categorical_features or LGBM_CATEGORICAL_FEATURES)
        ordered_train, inner_val, _ = _time_holdout_positions(dates, len(Xtr), test_size=0.15)
        if len(inner_val) == 0 or ytr.iloc[inner_val].nunique() < 2:
            ordered_train = np.arange(len(Xtr))
            inner_val = ordered_train[-max(1, int(len(ordered_train) * 0.15)):]
            ordered_train = ordered_train[:-len(inner_val)]
        if len(ordered_train) == 0:
            raise ValueError("Not enough rows to fit the LightGBM final base model.")

        Xtr_p = pre.fit_transform(Xtr.iloc[ordered_train])
        Xva_p = pre.transform(Xtr.iloc[inner_val])
        model = spec.estimator_factory()
        model.fit(
            Xtr_p,
            ytr.iloc[ordered_train],
            eval_set=[(Xva_p, ytr.iloc[inner_val])],
            categorical_feature=pre.categorical_cols_,
        )
        return pre, model, _candidate_feature_names(pre, None, None)

    pre, num, cat = _build_preprocessor(Xtr)
    Xtr_p = pre.fit_transform(Xtr)
    model = spec.estimator_factory()
    fit_kwargs = {}
    if spec.use_sample_weight:
        fit_kwargs["sample_weight"] = compute_sample_weight(
            class_weight="balanced", y=ytr
        )
    model.fit(Xtr_p, ytr, **fit_kwargs)
    return pre, model, _candidate_feature_names(pre, num, cat)


def _fit_isotonic_calibrator(model: Any, X_cal, y_cal: pd.Series) -> Any:
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(model),
        method="isotonic",
    )
    calibrated.fit(X_cal, y_cal)
    return calibrated


def _time_cv_auc(
    spec: CandidateSpec,
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series | None,
) -> dict:
    ordered = _ordered_positions(dates, len(X))
    X_ord = X.iloc[ordered].reset_index(drop=True)
    y_ord = y.iloc[ordered].reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=5, gap=30)
    aucs = []
    for tr, va in tscv.split(X_ord):
        if y_ord.iloc[va].nunique() < 2:
            continue
        _, _, _, proba = _fit_candidate_once(
            spec, X_ord.iloc[tr], y_ord.iloc[tr], X_ord.iloc[va], y_ord.iloc[va]
        )
        aucs.append(roc_auc_score(y_ord.iloc[va], proba))
    return {
        "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "auc_std": float(np.std(aucs)) if aucs else float("nan"),
        "folds": [float(a) for a in aucs],
        "cv_type": "time_series_5_fold_gap_30",
    }


def _random_cv_auc(spec: CandidateSpec, X: pd.DataFrame, y: pd.Series) -> dict:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr, va in skf.split(X, y):
        if y.iloc[va].nunique() < 2:
            continue
        _, _, _, proba = _fit_candidate_once(
            spec, X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va]
        )
        aucs.append(roc_auc_score(y.iloc[va], proba))
    return {
        "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "auc_std": float(np.std(aucs)) if aucs else float("nan"),
        "folds": [float(a) for a in aucs],
        "cv_type": "random_stratified_5_fold",
    }


def _fit_candidate(
    spec: CandidateSpec,
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series | None,
) -> CandidateResult:
    tr, te, cutoff_date = _time_holdout_positions(dates, len(X))
    cv = _time_cv_auc(spec, X, y, dates)
    cv_random = _random_cv_auc(spec, X, y)
    pre, model, feature_names, proba = _fit_candidate_once(
        spec, X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]
    )
    pred = (proba >= 0.5).astype(int)
    test_m = _scores(y.iloc[te], pred, proba)
    test_m["time_split_test_auc"] = test_m["auc_roc"]
    test_m["split_type"] = "date_cutoff_80_20"
    test_m["cutoff_date"] = cutoff_date.date().isoformat() if cutoff_date is not None else None
    test_m["train_size"] = int(len(tr))
    test_m["test_size"] = int(len(te))
    optimal_threshold, threshold_test = _threshold_tuning(y.iloc[te], proba)
    threshold_test["time_split_test_auc"] = threshold_test["auc_roc"]
    threshold_test["split_type"] = "date_cutoff_80_20"
    threshold_test["cutoff_date"] = test_m["cutoff_date"]
    threshold_test["train_size"] = int(len(tr))
    threshold_test["test_size"] = int(len(te))
    return CandidateResult(
        name=spec.name,
        spec=spec,
        algorithm=type(model).__name__,
        cv=cv,
        test=test_m,
        cv_auc_random_split=cv_random,
        model=model,
        preprocessor=pre,
        feature_names=feature_names,
        val_idx=X.iloc[te].index,
        val_proba=proba,
        val_y=y.iloc[te].values,
        val_X=X.iloc[te],
        cutoff_date=test_m["cutoff_date"],
        train_size=int(len(tr)),
        test_size=int(len(te)),
        optimal_threshold=optimal_threshold,
        threshold_test=threshold_test,
    )


def _refit_with_calibration(
    best: CandidateResult,
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series | None,
    model_profile: str,
) -> tuple[CandidateResult, Any, dict]:
    full_train_pos, test_pos, cutoff_date = _time_holdout_positions(dates, len(X))
    calibration_fraction = (
        WIN_CALIBRATION_FRACTION
        if model_profile == "win"
        else CONVERSION_CALIBRATION_FRACTION
    )
    model_fit_pos, calibration_pos = _calibration_split_positions(
        full_train_pos,
        dates,
        calibration_fraction,
    )
    if y.iloc[calibration_pos].nunique() < 2:
        raise ValueError(
            f"Calibration slice for {best.name} has only one target class; "
            "cannot fit isotonic calibration."
        )

    fit_dates = (
        pd.Series(dates).iloc[model_fit_pos].reset_index(drop=True)
        if dates is not None
        else None
    )
    pre, base_model, feature_names = _fit_candidate_final_base(
        best.spec,
        X.iloc[model_fit_pos],
        y.iloc[model_fit_pos],
        fit_dates,
    )

    X_cal_p = pre.transform(X.iloc[calibration_pos])
    calibrated_model = _fit_isotonic_calibrator(
        base_model,
        X_cal_p,
        y.iloc[calibration_pos],
    )

    Xte_p = pre.transform(X.iloc[test_pos])
    uncalibrated_proba = base_model.predict_proba(Xte_p)[:, 1]
    proba = calibrated_model.predict_proba(Xte_p)[:, 1]
    pred = (proba >= 0.5).astype(int)
    y_test = y.iloc[test_pos]

    test_m = _scores(y_test, pred, proba)
    test_m["time_split_test_auc"] = test_m["auc_roc"]
    test_m["split_type"] = "date_cutoff_80_20"
    test_m["cutoff_date"] = cutoff_date.date().isoformat() if cutoff_date is not None else None
    test_m["train_size"] = int(len(full_train_pos))
    test_m["test_size"] = int(len(test_pos))
    test_m["model_fit_size"] = int(len(model_fit_pos))
    test_m["calibration_size"] = int(len(calibration_pos))
    test_m["calibrated"] = True
    test_m["calibration_method"] = "isotonic_prefit_frozen_estimator"

    uncalibrated_ece = _compute_ece(y_test, uncalibrated_proba)
    calibrated_ece = _compute_ece(y_test, proba)
    test_m["uncalibrated_ece"] = uncalibrated_ece
    test_m["calibration_ece"] = calibrated_ece

    optimal_threshold, threshold_test = _threshold_tuning(y_test, proba)
    threshold_test["time_split_test_auc"] = threshold_test["auc_roc"]
    threshold_test["split_type"] = "date_cutoff_80_20"
    threshold_test["cutoff_date"] = test_m["cutoff_date"]
    threshold_test["train_size"] = int(len(full_train_pos))
    threshold_test["test_size"] = int(len(test_pos))
    threshold_test["model_fit_size"] = int(len(model_fit_pos))
    threshold_test["calibration_size"] = int(len(calibration_pos))
    threshold_test["calibrated"] = True
    threshold_test["calibration_ece"] = calibrated_ece

    calibration = {
        "method": "isotonic",
        "fit_strategy": "prefit_frozen_estimator",
        "holdout_fraction_of_training_window": float(calibration_fraction),
        "model_fit_size": int(len(model_fit_pos)),
        "calibration_size": int(len(calibration_pos)),
        "calibration_positive_rate": float(y.iloc[calibration_pos].mean()),
        "test_positive_rate": float(y_test.mean()),
        "uncalibrated_ece": uncalibrated_ece,
        "calibration_ece": calibrated_ece,
        "ece_target": 0.05,
        "ece_target_met": bool(calibrated_ece < 0.05),
    }
    if model_profile == "conversion":
        score_int = np.clip(np.rint(proba * 100), 0, 100).astype(int)
        high_fraction = float(np.mean(score_int >= HIGH_SCORE_THRESHOLD))
        calibration.update({
            "contract_high_threshold": int(HIGH_SCORE_THRESHOLD),
            "high_score_fraction": high_fraction,
            "high_threshold_adjustment_required": bool(high_fraction > 0.40),
            "recommended_high_threshold_if_adjustment_required": 75,
        })

    final_result = CandidateResult(
        name=best.name,
        spec=best.spec,
        algorithm=type(base_model).__name__,
        cv=best.cv,
        test=test_m,
        cv_auc_random_split=best.cv_auc_random_split,
        model=base_model,
        preprocessor=pre,
        feature_names=feature_names,
        val_idx=X.iloc[test_pos].index,
        val_proba=proba,
        val_y=y_test.values,
        val_X=X.iloc[test_pos],
        cutoff_date=test_m["cutoff_date"],
        train_size=int(len(full_train_pos)),
        test_size=int(len(test_pos)),
        optimal_threshold=optimal_threshold,
        threshold_test=threshold_test,
    )
    return final_result, calibrated_model, calibration


def _build_explainer(model: Any, preprocessor: Any = None, background_X: Any = None):
    """
    Pick a SHAP explainer for the model type:

      - tree models   -> TreeExplainer
      - linear models -> LinearExplainer (needs the preprocessor and a sample of
                         X for the background, so pass both)
      - anything else (e.g. a StackingClassifier) -> None

    Callers already check for None, so returning None just means no SHAP for that
    model rather than an error.
    """
    base = getattr(model, "estimator", model)  # unwrap calibrated/frozen wrappers
    try:
        # Linear models have coef_ but no feature_importances_; send those to
        # LinearExplainer so a logistic-regression win model stays explainable.
        if hasattr(base, "coef_") and not hasattr(base, "feature_importances_"):
            if preprocessor is None or background_X is None:
                print(f"  [WARN] No background data for LinearExplainer on "
                      f"{type(base).__name__}; SHAP explanations disabled.")
                return None
            background = preprocessor.transform(background_X)
            return shap.LinearExplainer(base, background)
        return shap.TreeExplainer(base)
    except Exception as exc:
        print(f"  [WARN] SHAP explainer unavailable for {type(base).__name__}: {exc}")
        return None


def _positive_shap_matrix(shap_values) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])

    values = np.asarray(shap_values)
    if values.ndim == 3:
        class_idx = 1 if values.shape[2] > 1 else 0
        return values[:, :, class_idx]
    return values


def _feature_importances(result: CandidateResult, explainer: Any | None = None) -> np.ndarray:
    model = result.model
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    if hasattr(model, "coef_"):
        return np.abs(np.asarray(model.coef_)).ravel()
    if explainer is not None:
        transformed = result.preprocessor.transform(result.val_X)
        try:
            shap_values = explainer.shap_values(transformed, check_additivity=False)
        except TypeError:
            shap_values = explainer.shap_values(transformed)
        return np.mean(np.abs(_positive_shap_matrix(shap_values)), axis=0)
    return np.zeros(len(result.feature_names), dtype=float)


def _segment_auc_report(
    df: pd.DataFrame,
    test_idx: pd.Index,
    proba: np.ndarray,
    target_col: str = TARGET_CONV,
) -> tuple[dict, list[dict]]:
    segment_aucs = {}
    weaknesses = []
    test_df = df.loc[test_idx]
    proba_by_index = pd.Series(proba, index=test_idx)
    for segment_col in ["CZ_NACE_Section", "Region_Name", "Size_Band", "LeadSource", "Rating"]:
        if segment_col not in test_df.columns:
            continue
        for segment_val in sorted(test_df[segment_col].dropna().unique()):
            mask = test_df[segment_col] == segment_val
            y_seg = test_df.loc[mask, target_col]
            if mask.sum() < 50 or y_seg.sum() < 5 or y_seg.nunique() < 2:
                continue
            segment_idx = test_df.index[mask.to_numpy()]
            seg_auc = roc_auc_score(y_seg, proba_by_index.loc[segment_idx])
            key = f"{segment_col}_{segment_val}"
            row = {"auc": float(seg_auc), "n": int(mask.sum())}
            segment_aucs[key] = row
            if seg_auc < 0.70:
                weaknesses.append({
                    "segment": key,
                    "auc": float(seg_auc),
                    "n": int(mask.sum()),
                })
    return segment_aucs, weaknesses


def _top_ranked_features(names: list[str], values: np.ndarray, top_n: int = 10) -> list[dict]:
    values = np.asarray(values, dtype=float)
    if len(values) < len(names):
        values = np.pad(values, (0, len(names) - len(values)))
    elif len(values) > len(names):
        values = values[:len(names)]
    order = np.argsort(values)[::-1][:top_n]
    return [
        {"feature": names[i], "importance": float(values[i]), "rank": rank}
        for rank, i in enumerate(order, start=1)
    ]


def _split_importance_reference(
    results: list[CandidateResult],
    selected: CandidateResult,
) -> dict | None:
    candidates = []
    selected_features = list(selected.feature_names)
    for result in results:
        if not hasattr(result.model, "feature_importances_"):
            continue
        if list(result.feature_names) != selected_features:
            continue
        candidates.append(result)
    if not candidates:
        return None

    best = max(candidates, key=lambda r: r.test["time_split_test_auc"])
    return {
        "model_name": best.name,
        "algorithm": best.algorithm,
        "time_split_test_auc": best.test["time_split_test_auc"],
        "feature_names": best.feature_names,
        "split_importances": np.asarray(best.model.feature_importances_, dtype=float),
    }


def _shap_vs_split_report(
    result: CandidateResult,
    explainer: Any | None,
    split_reference: dict | None = None,
) -> dict:
    if explainer is None:
        return {
            "status": "skipped",
            "reason": "SHAP explainer is unavailable for the selected model.",
        }
    if hasattr(result.model, "feature_importances_"):
        split_values = np.asarray(result.model.feature_importances_, dtype=float)
        split_source = {
            "type": "selected_model",
            "model_name": result.name,
            "algorithm": result.algorithm,
            "time_split_test_auc": result.test["time_split_test_auc"],
        }
    elif split_reference is not None:
        split_values = np.asarray(split_reference["split_importances"], dtype=float)
        split_source = {
            "type": "same_feature_space_reference",
            "model_name": split_reference["model_name"],
            "algorithm": split_reference["algorithm"],
            "time_split_test_auc": split_reference["time_split_test_auc"],
        }
    else:
        return {
            "status": "skipped",
            "reason": (
                f"{type(result.model).__name__} does not expose split importance "
                "and no same-feature-space GBM reference was available."
            ),
        }

    transformed = result.preprocessor.transform(result.val_X)
    try:
        shap_values = explainer.shap_values(transformed, check_additivity=False)
    except TypeError:
        shap_values = explainer.shap_values(transformed)
    shap_values = np.mean(np.abs(_positive_shap_matrix(shap_values)), axis=0)

    top_split = _top_ranked_features(result.feature_names, split_values)
    top_shap = _top_ranked_features(result.feature_names, shap_values)
    shap_rank = {row["feature"]: row["rank"] for row in top_shap}
    split_rank = {row["feature"]: row["rank"] for row in top_split}
    rank_warnings = []
    for row in top_split:
        feature = row["feature"]
        if feature not in shap_rank:
            rank_warnings.append({
                "feature": feature,
                "split_rank": row["rank"],
                "shap_rank": None,
                "message": "Top split-importance feature is absent from top 10 SHAP features.",
            })
            continue
        delta = shap_rank[feature] - row["rank"]
        if delta >= 7:
            rank_warnings.append({
                "feature": feature,
                "split_rank": row["rank"],
                "shap_rank": shap_rank[feature],
                "message": "Feature ranks much lower by SHAP than by split importance.",
            })

    return {
        "status": "ok",
        "split_importance_source": split_source,
        "top_10_split_importance": top_split,
        "top_10_shap_mean_abs": top_shap,
        "rank_warnings": rank_warnings,
        "split_only_top_10": [
            feature for feature in split_rank
            if feature not in shap_rank
        ],
        "shap_only_top_10": [
            feature for feature in shap_rank
            if feature not in split_rank
        ],
    }


def _run_optuna_lgbm(X: pd.DataFrame, y: pd.Series, dates: pd.Series | None) -> dict | None:
    if not RUN_OPTUNA:
        return {"enabled": False, "reason": "Set RUN_OPTUNA=1 to enable tuning."}
    if lgb is None or optuna is None:
        return {"enabled": False, "reason": "LightGBM and Optuna must be installed."}

    def objective(trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        spec = CandidateSpec(
            "lightgbm_optuna_trial",
            "lightgbm",
            lambda: _make_lgbm(params),
        )
        return _time_cv_auc(spec, X, y, dates)["auc_mean"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    return {
        "enabled": True,
        "n_trials": OPTUNA_TRIALS,
        "best_auc": float(study.best_value),
        "best_params": study.best_params,
    }


def _select_candidate(
    eligible: list[CandidateResult],
) -> tuple[CandidateResult, dict]:
    """
    Pick the model using a one-standard-error rule on cross-validated AUC rather
    than the raw test-split AUC.

    The top candidates' test AUCs differ by less than the CV standard error, so
    picking the single best on the test split just tracks noise and changes from
    run to run. Instead we keep every candidate within one SE of the best CV mean
    and take the simplest of those (MODEL_COMPLEXITY_RANK). Same performance, more
    stable choice. Complexity ties break on the higher CV mean.
    """
    def _mean(r: CandidateResult) -> float:
        return r.cv.get("auc_mean", float("nan"))

    def _se(r: CandidateResult) -> float:
        folds = r.cv.get("folds") or []
        if len(folds) < 2:
            return 0.0
        return float(np.std(folds, ddof=1) / np.sqrt(len(folds)))

    def _rank(r: CandidateResult) -> int:
        return MODEL_COMPLEXITY_RANK.get(r.name, 99)

    scored = [r for r in eligible if not np.isnan(_mean(r))]
    if not scored:
        # Degenerate CV (e.g. a tiny win slice). Fall back to held-out AUC.
        best = max(eligible, key=lambda r: r.test["auc_roc"])
        return best, {"rule": "fallback_test_auc", "reason": "no valid CV means"}

    best_cv = max(scored, key=_mean)
    threshold = _mean(best_cv) - _se(best_cv)
    within_1se = [r for r in scored if _mean(r) >= threshold]
    selected = min(within_1se, key=lambda r: (_rank(r), -_mean(r)))
    rationale = {
        "rule": "one_standard_error_on_cv_auc",
        "best_cv_candidate": best_cv.name,
        "best_cv_auc_mean": round(_mean(best_cv), 4),
        "best_cv_auc_se": round(_se(best_cv), 4),
        "within_1se_threshold": round(threshold, 4),
        "within_1se_candidates": sorted(r.name for r in within_1se),
        "selected": selected.name,
        "selected_cv_auc_mean": round(_mean(selected), 4),
        "selected_test_auc": round(selected.test["auc_roc"], 4),
        "argmax_test_auc_candidate": max(
            eligible, key=lambda r: r.test["auc_roc"]
        ).name,
    }
    return selected, rationale


def _train_one(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series | None = None,
    model_profile: str = "conversion",
) -> dict:
    print(f"\n--- Training: {name} (n={len(X):,}, positive rate={y.mean():.1%}) ---")

    remaining_excluded = [c for c in MODEL_DROP_COLS if c in X.columns]
    if remaining_excluded:
        raise ValueError(f"Excluded columns still in X: {remaining_excluded}")

    results: list[CandidateResult] = []
    skipped: dict[str, str] = {}
    candidate_notes: dict[str, str] = {}
    if lgb is None:
        skipped["lightgbm_native"] = "LightGBM is not installed."
        skipped["stacking_lgbm_hist_logreg"] = "Requires LightGBM."

    for spec in _candidate_specs(include_stacking=False, model_profile=model_profile):
        print(f"  Candidate: {spec.name}")
        result = _fit_candidate(spec, X, y, dates)
        print(
            f"    CV AUC: {result.cv['auc_mean']:.4f} +/- {result.cv['auc_std']:.4f} | "
            f"Test AUC: {result.test['auc_roc']:.4f} | F1: {result.test['f1']:.4f}"
        )
        results.append(result)

    by_name = {result.name: result for result in results}
    lgb_result = by_name.get("lightgbm_native")
    logreg_result = by_name.get("logreg")
    stacking_allowed_for_selection = False
    if lgb_result and logreg_result:
        if lgb_result.test["auc_roc"] > logreg_result.test["auc_roc"] + 0.02:
            stacking_allowed_for_selection = True
            candidate_notes["stacking_lgbm_hist_logreg"] = (
                "Recommended because LightGBM beat Logistic Regression by >0.02 AUC."
            )
        else:
            candidate_notes["stacking_lgbm_hist_logreg"] = (
                "Trained for comparison, but excluded from production selection by the A-vs-C gate."
            )
        spec = CandidateSpec(
            "stacking_lgbm_hist_logreg",
            "ohe",
            lambda: _make_stacking(model_profile),
        )
        print(f"  Candidate: {spec.name}")
        result = _fit_candidate(spec, X, y, dates)
        print(
            f"    CV AUC: {result.cv['auc_mean']:.4f} +/- {result.cv['auc_std']:.4f} | "
            f"Test AUC: {result.test['auc_roc']:.4f} | F1: {result.test['f1']:.4f}"
        )
        results.append(result)

    eligible_results = [
        result for result in results
        if result.name != "stacking_lgbm_hist_logreg" or stacking_allowed_for_selection
    ]
    best, selection_rationale = _select_candidate(eligible_results)
    print(f"  Selected: {best.name} ({best.algorithm}) via {selection_rationale['rule']}")
    if selection_rationale.get("within_1se_candidates"):
        argmax = selection_rationale["argmax_test_auc_candidate"]
        note = "" if argmax == best.name else f" (raw test-AUC argmax was {argmax})"
        print(
            f"    1-SE set {selection_rationale['within_1se_candidates']}; "
            f"best CV {selection_rationale['best_cv_candidate']}="
            f"{selection_rationale['best_cv_auc_mean']:.4f}"
            f"±{selection_rationale['best_cv_auc_se']:.4f}{note}"
        )

    optuna_result = None
    best_single = max(
        [r for r in results if r.name != "stacking_lgbm_hist_logreg"],
        key=lambda r: r.test["auc_roc"],
    )
    if best_single.name == "lightgbm_native":
        optuna_result = _run_optuna_lgbm(X, y, dates)

    final_base, calibrated_model, calibration = _refit_with_calibration(
        best,
        X,
        y,
        dates,
        model_profile,
    )
    print(
        "  Calibration: "
        f"ECE {calibration['uncalibrated_ece']:.4f} -> "
        f"{calibration['calibration_ece']:.4f}"
    )

    explainer = _build_explainer(
        final_base.model, final_base.preprocessor, final_base.val_X
    )
    split_reference = _split_importance_reference(results, final_base)
    shap_split_report = _shap_vs_split_report(final_base, explainer, split_reference)
    return {
        "algorithm": f"CalibratedClassifierCV({final_base.algorithm})",
        "base_algorithm": final_base.algorithm,
        "selected_candidate": best.name,
        "selection": selection_rationale,
        "candidates": {
            result.name: {
                "algorithm": result.algorithm,
                "cv": result.cv,
                "cv_auc_random_split": result.cv_auc_random_split,
                "test": result.test,
                "optimal_threshold": result.optimal_threshold,
                "threshold_test": result.threshold_test,
            }
            for result in results
        },
        "skipped_candidates": skipped,
        "candidate_notes": candidate_notes,
        "optuna": optuna_result,
        "cv": final_base.cv,
        "cv_auc_random_split": final_base.cv_auc_random_split,
        "test": final_base.test,
        "time_split_test_auc": final_base.test["time_split_test_auc"],
        "cutoff_date": final_base.cutoff_date,
        "train_size": final_base.train_size,
        "test_size": final_base.test_size,
        "optimal_threshold": final_base.optimal_threshold,
        "threshold_test": final_base.threshold_test,
        "calibration": calibration,
        "calibration_ece": calibration["calibration_ece"],
        "model": calibrated_model,
        "base_model": final_base.model,
        "preprocessor": final_base.preprocessor,
        "feature_names": final_base.feature_names,
        "explainer": explainer,
        "val_idx": final_base.val_idx,
        "val_proba": final_base.val_proba,
        "val_y": final_base.val_y,
        "val_X": final_base.val_X,
        "feature_importances": _feature_importances(final_base, explainer),
        "shap_vs_split_importance": shap_split_report,
    }

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Run data_generator.py first.")
        return

    df = pd.read_csv(DATA_PATH, dtype={"IČO": str, "Legal_Form_Code": str})
    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(int)
    add_engineered_features(df)
    print(f"Loaded {len(df):,} rows x {len(df.columns)} columns from {DATA_PATH}")

    # ── Build feature matrix (drop all excluded columns) ─────────────────
    if EXCLUDE_STATUS_PROXIES:
        print(
            "\n[feature mode] EXCLUDE_STATUS_PROXIES=1 - dropping Status leakage "
            f"proxies: {STATUS_PROXY_COLS}"
        )
    X = model_feature_frame(df, exclude_status_proxies=EXCLUDE_STATUS_PROXIES)
    y_conv = df[TARGET_CONV]
    missing_win_features = [col for col in WIN_MODEL_FEATURES if col not in df.columns]
    if missing_win_features:
        raise ValueError(f"Missing win model features: {missing_win_features}")
    X_win_all = win_feature_frame(df, exclude_status_proxies=EXCLUDE_STATUS_PROXIES)

    print(f"\nFeature matrix: {X.shape[1]} columns")
    print(f"  numeric:     {X.select_dtypes(include='number').shape[1]}")
    print(f"  categorical: {X.select_dtypes(include=['object', 'string', 'category']).shape[1]}")
    print(f"Win feature matrix: {X_win_all.shape[1]} columns")
    # ── 1. Conversion model ───────────────────────────────────────────────
    date_series = df["CreatedDate"] if "CreatedDate" in df.columns else None
    conv = _train_one("Conversion (Lead -> Opportunity)", X, y_conv, date_series)
    base_conv = _baseline_metrics(df, y_conv.loc[conv["val_idx"]])
    print(f"  Baseline  AUC: {base_conv['auc_roc']:.4f}  F1: {base_conv['f1']:.4f}")

    # ── 2. Win model (converted leads only) ──────────────────────────────
    won_mask = df[TARGET_CONV] == 1
    if won_mask.sum() < 50:
        print("\n  Warning: fewer than 50 converted leads - win model may be unreliable.")
    Xw = X_win_all.loc[won_mask].reset_index(drop=True)
    yw = df.loc[won_mask, TARGET_WIN].reset_index(drop=True)
    date_w = (
        df.loc[won_mask, "CreatedDate"].reset_index(drop=True)
        if "CreatedDate" in df.columns
        else None
    )
    win = _train_one(
        "Win (Opportunity -> Closed Won)",
        Xw,
        yw,
        date_w,
        model_profile="win",
    )
    segment_aucs, segment_auc_weaknesses = _segment_auc_report(
        df,
        conv["val_idx"],
        conv["val_proba"],
    )

    # ── 3. Persist model artifacts ────────────────────────────────────────
    joblib.dump(conv["model"],        CONV_MODEL)
    joblib.dump(conv["preprocessor"], CONV_PRE)
    joblib.dump(conv["feature_names"],CONV_FEAT)
    joblib.dump(conv["explainer"],    CONV_EXPL)
    joblib.dump(win["model"],         WIN_MODEL)
    joblib.dump(win["preprocessor"],  WIN_PRE)
    joblib.dump(win["explainer"],     WIN_EXPL)
    print(f"\nModels saved to {MODELS_DIR}/")

    # ── 4. Feature importance ─────────────────────────────────────────────
    importances = np.asarray(conv["feature_importances"], dtype=float)
    if len(importances) < len(conv["feature_names"]):
        importances = np.pad(
            importances,
            (0, len(conv["feature_names"]) - len(importances)),
        )
    elif len(importances) > len(conv["feature_names"]):
        importances = importances[:len(conv["feature_names"])]
    pd.DataFrame({
        "feature":    conv["feature_names"],
        "importance": importances,
    }).sort_values("importance", ascending=False).to_csv(FEATURE_IMPORTANCE, index=False)

    # ── 5. Held-out predictions (for ROC / PR / threshold explorer in app) ─
    # Two baselines, on purpose:
    #   stored_baseline - the generator's Rule_Based_Score, which includes the
    #     excluded Status field. Kept so older reports still line up.
    #   live_baseline   - baseline.score_frame(), the rules the API actually runs
    #     (no Status). We measure ranking lift against this one so the comparison
    #     matches what's deployed, not a stronger rule set we never ship.
    val_rows = df.loc[conv["val_idx"]]
    stored_baseline_val = (val_rows["Rule_Based_Score"] / 100).values
    live_baseline_val = (baseline.score_frame(val_rows) / 100).values
    pd.DataFrame({
        "y_true":          conv["val_y"],
        "ai_proba":        conv["val_proba"],
        "baseline_proba":  stored_baseline_val,
        "live_baseline_proba": live_baseline_val,
    }).to_csv(VAL_PREDICTIONS, index=False)

    # How many converters the AI reaches vs the deployed rules at each top-K cut.
    ranking = ranking_report(conv["val_y"], conv["val_proba"], live_baseline_val)

    # ── 6. Metrics JSON ───────────────────────────────────────────────────
    metrics = {
        "conversion_model": {
            "algorithm": conv["algorithm"],
            "base_algorithm": conv["base_algorithm"],
            "selected_candidate": conv["selected_candidate"],
            "selection": conv["selection"],
            "cv": conv["cv"],
            "cv_auc_random_split": conv["cv_auc_random_split"],
            "test": conv["test"],
            "time_split_test_auc": conv["time_split_test_auc"],
            "calibration": conv["calibration"],
            "calibration_ece": conv["calibration_ece"],
            "cutoff_date": conv["cutoff_date"],
            "train_size": conv["train_size"],
            "test_size": conv["test_size"],
            "optimal_threshold": conv["optimal_threshold"],
            "threshold_test": conv["threshold_test"],
            "shap_vs_split_importance": conv["shap_vs_split_importance"],
            "candidates": conv["candidates"],
            "skipped_candidates": conv["skipped_candidates"],
            "candidate_notes": conv["candidate_notes"],
            "optuna": conv["optuna"],
        },
        "win_model": {
            "algorithm": win["algorithm"],
            "base_algorithm": win["base_algorithm"],
            "selected_candidate": win["selected_candidate"],
            "selection": win["selection"],
            "cv": win["cv"],
            "cv_auc_random_split": win["cv_auc_random_split"],
            "test": win["test"],
            "time_split_test_auc": win["time_split_test_auc"],
            "calibration": win["calibration"],
            "calibration_ece": win["calibration_ece"],
            "cutoff_date": win["cutoff_date"],
            "train_size": win["train_size"],
            "test_size": win["test_size"],
            "optimal_threshold": win["optimal_threshold"],
            "threshold_test": win["threshold_test"],
            "shap_vs_split_importance": win["shap_vs_split_importance"],
            "candidates": win["candidates"],
            "skipped_candidates": win["skipped_candidates"],
            "candidate_notes": win["candidate_notes"],
            "optuna": win["optuna"],
        },
        "baseline_conversion":   base_conv,
        "comparison_conversion": {
            "ai_auc":       conv["test"]["auc_roc"],
            "baseline_auc": base_conv["auc_roc"],
            "ai_f1":        conv["test"]["f1"],
            "baseline_f1":  base_conv["f1"],
            "uplift_auc":   conv["test"]["auc_roc"] - base_conv["auc_roc"],
        },
        "ranking_lift_vs_live_baseline": ranking,
        "feature_mode": {
            "exclude_status_proxies": bool(EXCLUDE_STATUS_PROXIES),
            "status_proxy_cols": STATUS_PROXY_COLS,
            "note": (
                "Status leakage proxies excluded - uplift reflects real-data "
                "signal." if EXCLUDE_STATUS_PROXIES else
                "Default mode: Status leakage proxies retained; benchmark AUC is "
                "inflated relative to real CRM data."
            ),
        },
        "segment_aucs": segment_aucs,
        "segment_auc_weaknesses": segment_auc_weaknesses,
        "optimal_win_threshold": win["optimal_threshold"],
        "n_rows":       int(len(df)),
        "n_converted":  int(df[TARGET_CONV].sum()),
        "n_won":        int(df[TARGET_WIN].sum()),
        "feature_count": int(X.shape[1]),
        "win_feature_count": int(X_win_all.shape[1]),
    }
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(WIN_THRESHOLD, "w") as f:
        json.dump(
            {
                "optimal_win_threshold": float(win["optimal_threshold"]),
                "source": "win_model.threshold_test",
                "calibrated": True,
            },
            f,
            indent=2,
        )

    # ── 7. Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(
        f"Conversion model ({conv['selected_candidate']}) | "
        f"time-split AUC: {conv['time_split_test_auc']:.4f} | "
        f"F1: {conv['test']['f1']:.4f}"
    )
    print(
        f"Conversion cutoff | {conv['cutoff_date']} | "
        f"train={conv['train_size']:,}, test={conv['test_size']:,}"
    )
    print(
        "Conversion calibration | "
        f"ECE: {conv['calibration']['uncalibrated_ece']:.4f} -> "
        f"{conv['calibration_ece']:.4f} | "
        f"High share: {conv['calibration'].get('high_score_fraction', 0):.1%}"
    )
    if conv["calibration"].get("high_threshold_adjustment_required"):
        print(
            "Conversion threshold warning | >40% of test leads are High; "
            "raise HIGH_SCORE_THRESHOLD to 75 in contracts.py."
        )
    print(
        "Conversion random CV | "
        f"AUC: {conv['cv_auc_random_split']['auc_mean']:.4f} +/- "
        f"{conv['cv_auc_random_split']['auc_std']:.4f}"
    )
    print(f"Rule-based base. | AUC: {base_conv['auc_roc']:.4f} | F1: {base_conv['f1']:.4f}")
    print(f"Uplift (AUC):      {metrics['comparison_conversion']['uplift_auc']:+.4f}")
    print(
        f"Win model ({win['selected_candidate']})        | "
        f"time-split AUC: {win['time_split_test_auc']:.4f} | "
        f"F1: {win['test']['f1']:.4f}"
    )
    print(
        f"Win tuned threshold | {win['optimal_threshold']:.3f} | "
        f"F1: {win['threshold_test']['f1']:.4f}"
    )
    print(
        "Win calibration | "
        f"ECE: {win['calibration']['uncalibrated_ece']:.4f} -> "
        f"{win['calibration_ece']:.4f}"
    )
    print(
        "Score granularity | "
        f"{conv['test']['unique_scores_int']} integer buckets, "
        f"{conv['test']['unique_scores_1dp']} one-decimal buckets"
    )
    if segment_auc_weaknesses:
        print("\nSegment AUC weaknesses (<0.70):")
        for item in segment_auc_weaknesses[:10]:
            print(f"  {item['segment']}: AUC={item['auc']:.4f}, n={item['n']}")
    else:
        print("\nSegment AUC weaknesses (<0.70): none")

    shap_report = conv["shap_vs_split_importance"]
    if shap_report["status"] != "ok":
        print(f"SHAP vs split importance: {shap_report['status']} - {shap_report['reason']}")
    elif shap_report["rank_warnings"]:
        source = shap_report["split_importance_source"]
        print(
            "SHAP vs split importance warnings "
            f"(split source: {source['model_name']} / {source['algorithm']}):"
        )
        for item in shap_report["rank_warnings"]:
            print(
                f"  {item['feature']}: split_rank={item['split_rank']}, "
                f"shap_rank={item['shap_rank']}"
            )
    else:
        source = shap_report["split_importance_source"]
        print(
            "SHAP vs split importance: no dramatic top-10 rank differences "
            f"(split source: {source['model_name']} / {source['algorithm']})"
        )
    print(f"\nArtifacts written to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
