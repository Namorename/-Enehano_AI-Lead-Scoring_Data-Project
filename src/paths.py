"""
paths.py
========
Single source of truth for filesystem locations.
All other modules import constants from here so the project works the same
whether you run it from the repo root, from `src/`, or inside Docker.
"""
from __future__ import annotations
from pathlib import Path
 
ROOT: Path = Path(__file__).resolve().parent.parent
 
DATA_DIR:   Path = ROOT / "data"
MODELS_DIR: Path = ROOT / "src" / "models"
ASSETS_DIR: Path = ROOT / "assets"
 
# Data files
RES_SAMPLE = DATA_DIR / "res_open_data_sample.csv"
LEAD_TRAIN  = DATA_DIR / "lead_train.csv"
 
# Trained model artifacts
CONV_MODEL = MODELS_DIR / "model.pkl"
CONV_PRE   = MODELS_DIR / "preprocessor.pkl"
CONV_FEAT  = MODELS_DIR / "feature_names.pkl"
CONV_EXPL  = MODELS_DIR / "conv_explainer.pkl"
WIN_MODEL  = MODELS_DIR / "win_model.pkl"
WIN_PRE    = MODELS_DIR / "win_preprocessor.pkl"
WIN_EXPL   = MODELS_DIR / "win_explainer.pkl" 
# Reports / metrics
METRICS_JSON       = MODELS_DIR / "metrics.json"
FEATURE_IMPORTANCE = MODELS_DIR / "feature_importance.csv"
VAL_PREDICTIONS    = MODELS_DIR / "val_predictions.csv"
 
# Assets
LOGO_SVG = ASSETS_DIR / "enehano_logo.svg"