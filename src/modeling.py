"""
Reusable model/preprocessing helpers.

Classes in this module are importable by joblib when model artifacts are loaded
from the API process. Avoid defining artifact classes inside train.py, because
scripts executed as __main__ do not pickle cleanly across processes.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class LightGBMFramePreprocessor(BaseEstimator, TransformerMixin):
    """Prepare a pandas DataFrame for LightGBM native categorical handling."""

    def __init__(self, categorical_cols: Iterable[str] | None = None):
        self.categorical_cols = list(categorical_cols or [])

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = list(X.columns)
        self.categorical_cols_ = [
            col for col in self.categorical_cols if col in self.feature_names_in_
        ]
        self.category_values_ = {}
        for col in self.categorical_cols_:
            values = pd.Series(X[col], copy=False).astype("string").fillna("__missing__")
            categories = sorted(set(values.tolist()) | {"__missing__"})
            self.category_values_[col] = categories
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for col in self.feature_names_in_:
            if col in X.columns:
                out[col] = X[col]
            else:
                out[col] = np.nan

        for col in self.categorical_cols_:
            categories = self.category_values_[col]
            values = out[col].astype("string").fillna("__missing__")
            values = values.where(values.isin(categories), "__missing__")
            out[col] = pd.Categorical(values, categories=categories)

        numeric_cols = [col for col in out.columns if col not in self.categorical_cols_]
        for col in numeric_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.feature_names_in_, dtype=object)
