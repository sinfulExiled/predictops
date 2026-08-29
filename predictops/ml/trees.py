"""Gradient-boosted trees over the engineered lag features.

These are the honest competitor to the sequence models: given good rolling
features, a tree ensemble is often all a tabular time-series problem needs.
If XGBoost wins here, the project should say so.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TreeModel:
    booster: object
    kind: str
    columns: list[str]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.kind == "xgboost":
            import xgboost as xgb
            return self.booster.predict(xgb.DMatrix(x, feature_names=self.columns))
        return self.booster.predict(x)

    def feature_importance(self) -> dict[str, float]:
        if self.kind == "xgboost":
            raw = self.booster.get_score(importance_type="gain")
            total = sum(raw.values()) or 1.0
            return {k: v / total for k, v in
                    sorted(raw.items(), key=lambda kv: -kv[1])}
        imp = self.booster.feature_importance(importance_type="gain")
        total = float(imp.sum()) or 1.0
        pairs = sorted(zip(self.columns, imp), key=lambda kv: -kv[1])
        return {k: float(v) / total for k, v in pairs}


def train_tree(kind: str, x_tr: np.ndarray, y_tr: np.ndarray,
               x_va: np.ndarray, y_va: np.ndarray, columns: list[str],
               seed: int = 42, n_rounds: int = 600) -> TreeModel:
    pos = max(int((y_tr == 1).sum()), 1)
    neg = int((y_tr == 0).sum())
    spw = neg / pos

    if kind == "xgboost":
        import xgboost as xgb
        dtr = xgb.DMatrix(x_tr, label=y_tr, feature_names=columns)
        dva = xgb.DMatrix(x_va, label=y_va, feature_names=columns)
        params = {
            "objective": "binary:logistic", "eval_metric": "aucpr",
            "eta": 0.05, "max_depth": 6, "subsample": 0.85,
            "colsample_bytree": 0.7, "min_child_weight": 5,
            "reg_lambda": 2.0, "scale_pos_weight": spw,
            "seed": seed, "nthread": 0,
        }
        booster = xgb.train(params, dtr, num_boost_round=n_rounds,
                            evals=[(dva, "val")], early_stopping_rounds=40,
                            verbose_eval=False)
        return TreeModel(booster, kind, columns)

    if kind == "lightgbm":
        import lightgbm as lgb
        dtr = lgb.Dataset(x_tr, label=y_tr, feature_name=columns)
        dva = lgb.Dataset(x_va, label=y_va, feature_name=columns, reference=dtr)
        params = {
            "objective": "binary", "metric": "average_precision",
            "learning_rate": 0.05, "num_leaves": 48, "min_data_in_leaf": 40,
            "feature_fraction": 0.7, "bagging_fraction": 0.85,
            "bagging_freq": 1, "lambda_l2": 2.0, "scale_pos_weight": spw,
            "seed": seed, "verbosity": -1, "num_threads": 0,
        }
        booster = lgb.train(params, dtr, num_boost_round=n_rounds,
                            valid_sets=[dva],
                            callbacks=[lgb.early_stopping(40, verbose=False)])
        return TreeModel(booster, kind, columns)

    raise ValueError(f"unknown tree model: {kind}")
