"""
Linear Regression Model Module
==============================
Implements Linear Regression with Ridge/Lasso regularization options.
"""

import pickle
import joblib
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.base import BaseEstimator, RegressorMixin

# Add project root to path
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)


class HousePriceModel:
    """
    Linear Regression model for house price prediction.
    Supports plain Linear Regression, Ridge, Lasso, and ElasticNet.
    """

    MODEL_TYPES = {
        "linear": LinearRegression,
        "ridge": Ridge,
        "lasso": Lasso,
        "elasticnet": ElasticNet,
    }

    def __init__(self, model_type: str = "ridge", **kwargs):
        """
        Initialize the model.

        Args:
            model_type: Type of model ('linear', 'ridge', 'lasso', 'elasticnet')
            **kwargs: Additional parameters for the model
        """
        self.model_type = model_type.lower()
        self.model_params = kwargs
        self.model = None
        self.feature_names = None

        if self.model_type not in self.MODEL_TYPES:
            raise ValueError(
                f"Unknown model type: {model_type}. Choose from {list(self.MODEL_TYPES.keys())}"
            )

        logger.info(f"Initialized {model_type} model with params: {kwargs}")

    def _create_model(self):
        """Create the underlying model instance."""
        model_class = self.MODEL_TYPES[self.model_type]

        if self.model_type == "linear":
            # LinearRegression doesn't have alpha parameter
            self.model = model_class(**self.model_params)
        else:
            # Ridge, Lasso, ElasticNet have alpha parameter
            self.model = model_class(**self.model_params)

    def fit(self, X, y, feature_names: Optional[list] = None):
        """
        Train the model.

        Args:
            X: Training features (numpy array or pandas DataFrame)
            y: Target variable
            feature_names: List of feature names
        """
        logger.info(f"Training {self.model_type} model...")

        # Create model if not exists
        if self.model is None:
            self._create_model()

        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        elif hasattr(X, "columns"):
            self.feature_names = list(X.columns)

        # Fit the model
        self.model.fit(X, y)

        # Log model coefficients
        if hasattr(self.model, "coef_"):
            logger.info(f"Model coefficients shape: {self.model.coef_.shape}")
            if self.feature_names:
                top_features = np.argsort(np.abs(self.model.coef_))[-5:][::-1]
                logger.info("Top 5 important features:")
                for idx in top_features:
                    logger.info(
                        f"  {self.feature_names[idx]}: {self.model.coef_[idx]:.4f}"
                    )

        logger.info("Model training completed")

        return self

    def predict(self, X):
        """
        Make predictions.

        Args:
            X: Features to predict on

        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        predictions = self.model.predict(X)

        # Ensure predictions are non-negative (house prices can't be negative)
        predictions = np.maximum(predictions, 0)

        logger.info(f"Made {len(predictions)} predictions")

        return predictions

    def score(self, X, y) -> float:
        """
        Calculate R² score.

        Args:
            X: Features
            y: True target values

        Returns:
            R² score
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        return self.model.score(X, y)

    def get_coefficients(self) -> Dict[str, float]:
        """
        Get feature coefficients.

        Returns:
            Dictionary mapping feature names to coefficients
        """
        if self.model is None or not hasattr(self.model, "coef_"):
            return {}

        if self.feature_names:
            return dict(zip(self.feature_names, self.model.coef_))
        else:
            return {f"feature_{i}": coef for i, coef in enumerate(self.model.coef_)}

    def get_intercept(self) -> float:
        """Get model intercept."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        return self.model.intercept_ if hasattr(self.model, "intercept_") else 0.0

    def save(self, filepath: str):
        """
        Save model to file.

        Args:
            filepath: Path to save model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self, filepath)
        logger.info(f"Model saved to: {filepath}")

    @staticmethod
    def load(filepath: str) -> "HousePriceModel":
        """
        Load model from file.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded model instance
        """
        model = joblib.load(filepath)
        logger.info(f"Model loaded from: {filepath}")
        return model

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "model_type": self.model_type,
            "model_params": self.model_params,
            "feature_names": self.feature_names,
        }


def create_model(config: Optional[Dict[str, Any]] = None) -> HousePriceModel:
    """
    Factory function to create a model from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        HousePriceModel instance
    """
    if config is None:
        config = load_config()

    model_config = config.get("model", {})
    model_type = model_config.get("name", "Ridge").lower()
    model_params = model_config.get("params", {})

    # Map model name
    name_mapping = {
        "linear": "linear",
        "linearregression": "linear",
        "ridge": "ridge",
        "lasso": "lasso",
        "elasticnet": "elasticnet",
    }

    model_type = name_mapping.get(model_type, "ridge")

    logger.info(f"Creating {model_type} model with params: {model_params}")

    return HousePriceModel(model_type=model_type, **model_params)
