from __future__ import annotations

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np


class LambdaMARTModel:
    """LightGBM LambdaMART wrapper for learning-to-rank."""

    DEFAULT_PARAMS = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5, 10],
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "min_child_samples": 10,
        "verbose": -1,
    }

    def __init__(self, params: dict | None = None) -> None:
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self._model: lgb.Booster | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        groups_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        groups_val: np.ndarray | None = None,
    ) -> None:
        train_data = lgb.Dataset(X_train, label=y_train, group=groups_train)
        valid_sets = []
        if X_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, group=groups_val, reference=train_data)
            valid_sets = [val_data]

        self._model = lgb.train(
            params={k: v for k, v in self.params.items() if k != "n_estimators"},
            train_set=train_data,
            num_boost_round=self.params["n_estimators"],
            valid_sets=valid_sets or None,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self._model.predict(X)

    def rank_documents(
        self,
        query_features: np.ndarray,
        doc_texts: list[str],
    ) -> list[tuple[str, float]]:
        """Rank documents for a single query.

        Returns:
            List of (doc_text, score) sorted by descending score.
        """
        scores = self.predict(query_features)
        ranked = sorted(zip(doc_texts, scores.tolist()), key=lambda x: x[1], reverse=True)
        return ranked

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            self._model = pickle.load(f)
