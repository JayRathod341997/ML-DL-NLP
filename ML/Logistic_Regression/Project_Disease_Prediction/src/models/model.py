"""Logistic Regression Model for Disease Prediction"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import joblib
import os


class LogisticRegressionClassifier:
    """Logistic Regression classifier for disease prediction"""

    def __init__(
        self,
        solver="lbfgs",
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        random_state=42,
    ):
        """Initialize Logistic Regression classifier

        Args:
            solver: Optimization algorithm
            max_iter: Maximum number of iterations
            C: Regularization parameter
            class_weight: Class weights for imbalanced data
            random_state: Random seed
        """
        self.solver = solver
        self.max_iter = max_iter
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        """Train the logistic regression model

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            self
        """
        self.model = LogisticRegression(
            solver=self.solver,
            max_iter=self.max_iter,
            C=self.C,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Predict class labels

        Args:
            X: Feature matrix

        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities

        Args:
            X: Feature matrix

        Returns:
            Probability predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        """Evaluate model performance

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]

        results = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(
                y, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y, y_pred, average="weighted", zero_division=0),
            "roc_auc": roc_auc_score(y, y_proba),
        }

        return results

    def get_model_info(self):
        """Get model information

        Returns:
            Dictionary with model details
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        return {
            "coefficients": self.model.coef_.tolist(),
            "intercept": self.model.intercept_.tolist(),
            "n_features": self.model.coef_.shape[1],
            "classes": self.model.classes_.tolist(),
        }

    def save(self, filepath):
        """Save model to file

        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)

    def load(self, filepath):
        """Load model from file

        Args:
            filepath: Path to the saved model
        """
        self.model = joblib.load(filepath)
