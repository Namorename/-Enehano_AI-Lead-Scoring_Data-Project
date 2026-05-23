"""
train.py
========
Trains and evaluates the Enehano lead-scoring models.

Two classifiers are trained:
  * conversion model : Lead → Opportunity  (target = Converted)
  * win model        : Opportunity → Closed Won  (target = Closed_Won,
                       trained only on rows where Converted == 1)

Evaluation:
  * 5-fold stratified cross-validation (mean ± std AUC)
  * Held-out test set: AUC-ROC, PR-AUC, precision, recall, F1, confusion matrix
  * Side-by-side comparison with the rule-based baseline
  * Top feature importances saved to CSV

Artifacts written:
  model.pkl, preprocessor.pkl, feature_names.pkl,
  win_model.pkl, win_preprocessor.pkl,
  metrics.json, feature_importance.csv, val_predictions.csv

Column contract (matches data_generator.py output):
  ID / audit cols  → dropped before building X (never seen by the model)
  Leaky cols       → dropped (Conversion_Probability, Rule_Based_Score/Segment)
  Date strings     → dropped (CreatedDate, LastActivityDate — Days_in_Pipeline__c
                     is kept as the numeric proxy)
  Bool flags       → generator writes them as int (0/1), so StandardScaler
                     handles them correctly without special casing
  Legal_Form_Code  → generator writes it as str, so OHE handles it correctly
"""

import json
import os
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ──────────────────────────────────────────────────────────────────────────────
# PATHS  (resolve relative to this script so it works from any cwd)
# ──────────────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent

DATA_PATH      = str(_HERE / "lead_train.csv")
MODELS_DIR     = _HERE / "models"
CONV_MODEL_OUT = str(MODELS_DIR / "model.pkl")
CONV_PRE_OUT   = str(MODELS_DIR / "preprocessor.pkl")
CONV_FEAT_OUT  = str(MODELS_DIR / "feature_names.pkl")
WIN_MODEL_OUT  = str(MODELS_DIR / "win_model.pkl")
WIN_PRE_OUT    = str(MODELS_DIR / "win_preprocessor.pkl")
METRICS_OUT    = str(MODELS_DIR / "metrics.json")
IMPORTANCE_OUT = str(MODELS_DIR / "feature_importance.csv")
VAL_PRED_OUT   = str(MODELS_DIR / "val_predictions.csv")

MODELS_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# COLUMN EXCLUSION LISTS
# ──────────────────────────────────────────────────────────────────────────────

# Columns that identify a record but carry no predictive signal.
ID_COLS = ["Lead_ID", "IČO", "Company_Name"]

# Target variables — must never appear in X.
TARGET_CONV = "Converted"
TARGET_WIN  = "Closed_Won"

# Status is the single strongest leaky predictor of Converted (it literally
# encodes the outcome), so it is excluded from the feature set.
STATUS_COL = "Status"

# These columns are derived directly from the conversion label or from the
# causal model used to generate the label — including them would let the model
# "cheat" and inflate AUC without learning anything real.
LEAKY_COLS = [
    "Conversion_Probability",   # the latent probability that generated Converted
    "Rule_Based_Score",         # computed from the same features + label direction
    "Rule_Based_Segment",       # bucketed Rule_Based_Score
    "IČO_Duplicate_Flag",       # data-quality flag, not a business signal
]

# Raw date strings that cannot be used by ColumnTransformer without parsing.
# Days_in_Pipeline__c is already the numeric proxy and is kept.
DATE_COLS = ["CreatedDate", "LastActivityDate"]

# All columns to remove before building the feature matrix.
DROP_COLS = ID_COLS + [TARGET_CONV, TARGET_WIN, STATUS_COL] + LEAKY_COLS + DATE_COLS

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
    # "str" works in pandas 3+; "object" is the legacy alias — support both
    cat = X.select_dtypes(include=["object", "str"]).columns.tolist()
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ],
        remainder="drop",   # explicitly drop anything not in num or cat
    )
    return pre, num, cat


def _scores(y_true, y_pred, y_proba) -> dict:
    return {
        "auc_roc":          float(roc_auc_score(y_true, y_proba)),
        "pr_auc":           float(average_precision_score(y_true, y_proba)),
        "precision":        float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":           float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":               float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "positive_rate":    float(np.mean(y_true)),
    }


def _cross_val_auc(model_factory, pre_factory, X: pd.DataFrame, y: pd.Series) -> dict:
    """5-fold stratified CV — each fold refits its own preprocessor to avoid leakage."""
    skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr, va in skf.split(X, y):
        pre = pre_factory()
        Xt  = pre.fit_transform(X.iloc[tr])
        Xv  = pre.transform(X.iloc[va])
        m   = model_factory()
        m.fit(Xt, y.iloc[tr])
        aucs.append(roc_auc_score(y.iloc[va], m.predict_proba(Xv)[:, 1]))
    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std":  float(np.std(aucs)),
        "folds":    [float(a) for a in aucs],
    }


def _train_one(name: str, X: pd.DataFrame, y: pd.Series) -> dict:
    print(f"\n--- Training: {name} (n={len(X):,}, positive rate={y.mean():.1%}) ---")

    # Verify no unexpected columns slipped through
    remaining_leaky = [c for c in LEAKY_COLS + DATE_COLS if c in X.columns]
    if remaining_leaky:
        raise ValueError(f"Leaky/invalid columns still in X: {remaining_leaky}")

    def _pre_factory():
        pre, _, _ = _build_preprocessor(X)
        return pre

    def _model_factory():
        return RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=20,
            random_state=42, class_weight="balanced", n_jobs=-1,
        )

    cv = _cross_val_auc(_model_factory, _pre_factory, X, y)
    print(f"  CV AUC: {cv['auc_mean']:.4f} ± {cv['auc_std']:.4f}")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pre, num, cat = _build_preprocessor(Xtr)
    Xtr_p = pre.fit_transform(Xtr)
    Xte_p = pre.transform(Xte)
    model  = _model_factory().fit(Xtr_p, ytr)

    proba  = model.predict_proba(Xte_p)[:, 1]
    pred   = (proba >= 0.5).astype(int)
    test_m = _scores(yte, pred, proba)
    print(f"  Test AUC: {test_m['auc_roc']:.4f}  F1: {test_m['f1']:.4f}")

    # Feature names: numeric first, then OHE-expanded categoricals
    feat_names = num + pre.named_transformers_["cat"].get_feature_names_out(cat).tolist()

    return {
        "cv": cv, "test": test_m,
        "model": model, "preprocessor": pre, "feature_names": feat_names,
        "val_idx": Xte.index, "val_proba": proba, "val_y": yte.values,
        "val_X": Xte,
    }


def _baseline_metrics(df: pd.DataFrame, y: pd.Series) -> dict:
    """
    Rule-based baseline: uses Rule_Based_Score (0–100) normalised to [0,1]
    as the probability proxy. The score column is read from the raw df so it
    is never part of the model feature matrix.
    """
    proba = (df.loc[y.index, "Rule_Based_Score"] / 100).values
    pred  = (proba >= 0.5).astype(int)
    return _scores(y, pred, proba)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Run data_generator.py first.")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns from {DATA_PATH}")

    # ── Build feature matrix (drop all excluded columns) ─────────────────
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    y_conv = df[TARGET_CONV]

    print(f"\nFeature matrix: {X.shape[1]} columns")
    print(f"  numeric:     {X.select_dtypes(include='number').shape[1]}")
    print(f"  categorical: {X.select_dtypes(include=['object', 'str']).shape[1]}")

    # ── 1. Conversion model ───────────────────────────────────────────────
    conv      = _train_one("Conversion (Lead → Opportunity)", X, y_conv)
    base_conv = _baseline_metrics(df, y_conv)
    print(f"  Baseline  AUC: {base_conv['auc_roc']:.4f}  F1: {base_conv['f1']:.4f}")

    # ── 2. Win model (converted leads only) ──────────────────────────────
    won_mask = df[TARGET_CONV] == 1
    if won_mask.sum() < 50:
        print("\n  Warning: fewer than 50 converted leads — win model may be unreliable.")
    Xw  = X.loc[won_mask].reset_index(drop=True)
    yw  = df.loc[won_mask, TARGET_WIN].reset_index(drop=True)
    win = _train_one("Win (Opportunity → Closed Won)", Xw, yw)

    # ── 3. Persist model artifacts ────────────────────────────────────────
    joblib.dump(conv["model"],        CONV_MODEL_OUT)
    joblib.dump(conv["preprocessor"], CONV_PRE_OUT)
    joblib.dump(conv["feature_names"],CONV_FEAT_OUT)
    joblib.dump(win["model"],         WIN_MODEL_OUT)
    joblib.dump(win["preprocessor"],  WIN_PRE_OUT)
    print(f"\nModels saved to {MODELS_DIR}/")

    # ── 4. Feature importance ─────────────────────────────────────────────
    pd.DataFrame({
        "feature":    conv["feature_names"],
        "importance": conv["model"].feature_importances_,
    }).sort_values("importance", ascending=False).to_csv(IMPORTANCE_OUT, index=False)

    # ── 5. Held-out predictions (for ROC / PR / threshold explorer in app) ─
    base_proba_val = (
        df.loc[conv["val_idx"], "Rule_Based_Score"] / 100
    ).values
    pd.DataFrame({
        "y_true":          conv["val_y"],
        "ai_proba":        conv["val_proba"],
        "baseline_proba":  base_proba_val,
    }).to_csv(VAL_PRED_OUT, index=False)

    # ── 6. Metrics JSON ───────────────────────────────────────────────────
    metrics = {
        "conversion_model":      {"cv": conv["cv"], "test": conv["test"]},
        "win_model":             {"cv": win["cv"],  "test": win["test"]},
        "baseline_conversion":   base_conv,
        "comparison_conversion": {
            "ai_auc":       conv["test"]["auc_roc"],
            "baseline_auc": base_conv["auc_roc"],
            "ai_f1":        conv["test"]["f1"],
            "baseline_f1":  base_conv["f1"],
            "uplift_auc":   conv["test"]["auc_roc"] - base_conv["auc_roc"],
        },
        "n_rows":       int(len(df)),
        "n_converted":  int(df[TARGET_CONV].sum()),
        "n_won":        int(df[TARGET_WIN].sum()),
        "feature_count": int(X.shape[1]),
    }
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    # ── 7. Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Conversion model | AUC: {conv['test']['auc_roc']:.4f} | F1: {conv['test']['f1']:.4f}")
    print(f"Rule-based base. | AUC: {base_conv['auc_roc']:.4f} | F1: {base_conv['f1']:.4f}")
    print(f"Uplift (AUC):      {metrics['comparison_conversion']['uplift_auc']:+.4f}")
    print(f"Win model        | AUC: {win['test']['auc_roc']:.4f} | F1: {win['test']['f1']:.4f}")
    print(f"\nArtifacts written to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
