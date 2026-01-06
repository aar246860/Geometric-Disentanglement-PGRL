"""Residual learning models for lag parameter prediction."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import xgboost as xgb


class ResidualLagModel:
    """Physics-guided residual learning for tau parameters."""

    DEFAULT_FEATURES = [
        "Phys_Perc_Betti0",      # beta_0: topological obstacles
        "Phys_Perc_Backbone",    # flow backbone
        "Phys_Morph_Perimeter",  # surface area proxy
        "Phys_Perc_ClusterSize",
        "Phys_Info_Entropy",
        "Phys_Info_Div",
        "Phys_Morph_Lacunarity",
        "Geostat_Mu_K",
        "Geostat_Mu_Ss",
        "Geostat_Rho",
    ]

    def __init__(
        self,
        cutoff_n: int = 9200,
        params: Optional[Dict] = None,
        seed: int = 42,
        theory_col: str = "Phys_Theory",
    ) -> None:
        self.cutoff_n = int(cutoff_n)
        self.params = params or {}
        self.seed = int(seed)
        self.theory_col = theory_col

        self.base_model_ = None
        self.residual_model_ = None
        self.features_: List[str] = []
        self.metrics_: Dict[str, float] = {}

    @staticmethod
    def infer_feature_columns(df: pd.DataFrame) -> List[str]:
        missing = [c for c in ResidualLagModel.DEFAULT_FEATURES if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required feature columns: {missing}")
        return list(ResidualLagModel.DEFAULT_FEATURES)

    def fit(self, df: pd.DataFrame, target_col: str, feature_cols: Optional[List[str]] = None) -> float:
        if feature_cols is None:
            feature_cols = self.infer_feature_columns(df)

        X_theory = df[self.theory_col].values.reshape(-1, 1)
        y_val = df[target_col].values

        base = LinearRegression().fit(X_theory, y_val)
        resid_sq = (y_val - base.predict(X_theory)) ** 2

        df_temp = df.copy()
        df_temp["Resid_Sq"] = resid_sq
        df_golden = df_temp.sort_values("Resid_Sq").head(self.cutoff_n).copy()

        X_base = df_golden[self.theory_col].values.reshape(-1, 1)
        y_true = df_golden[target_col].values
        base_fit = LinearRegression().fit(X_base, y_true)
        y_base = base_fit.predict(X_base)
        residuals = y_true - y_base

        # TODO: Scientific Check - manuscript mentions ExtraTrees, code uses XGBoost. Verify alignment.
        params = {"n_jobs": -1, "random_state": self.seed}
        params.update(self.params)
        model = xgb.XGBRegressor(**params)
        model.fit(df_golden[feature_cols], residuals)

        y_resid_pred = model.predict(df_golden[feature_cols])
        final_r2 = r2_score(y_true, y_base + y_resid_pred)

        self.base_model_ = base_fit
        self.residual_model_ = model
        self.features_ = list(feature_cols)
        self.metrics_ = {"r2": float(final_r2)}

        return float(final_r2)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.base_model_ is None or self.residual_model_ is None:
            raise RuntimeError("Model is not fitted.")
        X_base = df[self.theory_col].values.reshape(-1, 1)
        y_base = self.base_model_.predict(X_base)
        y_res = self.residual_model_.predict(df[self.features_])
        return y_base + y_res
