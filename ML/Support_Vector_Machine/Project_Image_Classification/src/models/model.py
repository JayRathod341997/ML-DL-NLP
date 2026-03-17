"""SVM Classifier model module"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from ..utils.logger import get_logger

logger = get_logger(__name__)


class SVMClassifier:
    """Support Vector Machine classifier for image classification"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SVM Classifier

        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        self.model_config = config.get("model", {})

        # Initialize model with configuration
        self.model = SVC(
            kernel=self.model_config.get("kernel", "rbf"),
            C=self.model_config.get("C", 1.0),
            gamma=self.model_config.get("gamma", "scale"),
            degree=self.model_config.get("degree", 3),
            probability=self.model_config.get("probability", True),
            random_state=42,
            cache_size=500,
        )

        self.best_params_ = None
        self.label_encoder_ = None
        self.is_trained_ = False

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        tune_hyperparameters: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the SVM classifier

        Args:
            X_train: Training features
            y_train: Training labels
            tune_hyperparameters: Whether to perform hyperparameter tuning

        Returns:
            Dictionary containing training results
        """
        logger.info("Starting SVM training...")

        results = {"training_accuracy": 0.0, "cv_scores": [], "best_params": {}}

        if tune_hyperparameters:
            logger.info("Performing hyperparameter tuning...")
            best_params = self._tune_hyperparameters(X_train, y_train)
            results["best_params"] = best_params

            # Update model with best parameters
            self.model = SVC(
                kernel=best_params.get("kernel", "rbf"),
                C=best_params.get("C", 1.0),
                gamma=best_params.get("gamma", "scale"),
                degree=best_params.get("degree", 3),
                probability=True,
                random_state=42,
            )

        # Train final model
        logger.info("Training final model with best parameters...")
        self.model.fit(X_train, y_train)

        # Calculate training accuracy
        train_predictions = self.model.predict(X_train)
        training_accuracy = accuracy_score(y_train, train_predictions)
        results["training_accuracy"] = training_accuracy

        # Cross-validation
        cv_folds = self.config.get("training", {}).get("cv_folds", 5)
        cv_scores = cross_val_score(
            self.model, X_train, y_train, cv=cv_folds, scoring="accuracy"
        )
        results["cv_scores"] = cv_scores.tolist()

        logger.info(f"Training completed. Training accuracy: {training_accuracy:.4f}")
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(
            f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
        )

        self.is_trained_ = True
        return results

    def _tune_hyperparameters(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Dictionary of best hyperparameters
        """
        param_grid = self.config.get("hyperparameters", {})

        cv_folds = self.config.get("training", {}).get("cv_folds", 5)

        grid_search = GridSearchCV(
            SVC(probability=True, random_state=42),
            param_grid,
            cv=cv_folds,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        self.best_params_ = grid_search.best_params_
        logger.info(f"Best parameters: {self.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

        return self.best_params_

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model on test data

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained_:
            raise ValueError("Model has not been trained yet")

        logger.info("Evaluating model on test data...")

        y_pred = self.model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test Precision: {metrics['precision']:.4f}")
        logger.info(f"Test Recall: {metrics['recall']:.4f}")
        logger.info(f"Test F1-Score: {metrics['f1_score']:.4f}")

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data

        Args:
            X: Features to predict on

        Returns:
            Array of predictions
        """
        if not self.is_trained_:
            raise ValueError("Model has not been trained yet")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities

        Args:
            X: Features to predict on

        Returns:
            Array of prediction probabilities
        """
        if not self.is_trained_:
            raise ValueError("Model has not been trained yet")

        return self.model.predict_proba(X)

    def save(self, filepath: str):
        """
        Save the model to disk

        Args:
            filepath: Path to save the model
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """
        Load the model from disk

        Args:
            filepath: Path to the saved model
        """
        self.model = joblib.load(filepath)
        self.is_trained_ = True
        logger.info(f"Model loaded from {filepath}")
