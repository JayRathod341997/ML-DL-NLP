"""
Decision Tree Model Module
Credit Scoring Project
"""

import pickle
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import logging

logger = logging.getLogger(__name__)


class CreditDecisionTree:
    """Decision Tree Classifier for Credit Scoring."""

    def __init__(self, config: dict):
        """
        Initialize the model with configuration.

        Args:
            config: Dictionary containing model parameters
        """
        self.config = config
        model_config = config.get("model", {})

        self.model = DecisionTreeClassifier(
            max_depth=model_config.get("max_depth", 5),
            min_samples_split=model_config.get("min_samples_split", 10),
            min_samples_leaf=model_config.get("min_samples_leaf", 5),
            criterion=model_config.get("criterion", "gini"),
            random_state=model_config.get("random_state", 42),
        )

        self.best_params = None
        logger.info(f"Initialized DecisionTreeClassifier with params: {model_config}")

    def train(self, X_train, y_train):
        """
        Train the Decision Tree model.

        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info("Starting model training")

        self.model.fit(X_train, y_train)

        train_score = self.model.score(X_train, y_train)
        logger.info(f"Training completed. Training accuracy: {train_score:.4f}")

        return self

    def tune_hyperparameters(self, X_train, y_train):
        """
        Tune hyperparameters using GridSearchCV.

        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info("Starting hyperparameter tuning")

        param_grid = {
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "criterion": ["gini", "entropy"],
        }

        cv_folds = self.config.get("training", {}).get("cv_folds", 5)

        grid_search = GridSearchCV(
            DecisionTreeClassifier(random_state=42),
            param_grid,
            cv=cv_folds,
            scoring="accuracy",
            n_jobs=-1,
        )

        grid_search.fit(X_train, y_train)

        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_

        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        return self

    def cross_validate(self, X, y, cv: int = 5):
        """
        Perform cross-validation.

        Args:
            X: Features
            y: Labels
            cv: Number of folds

        Returns:
            Cross-validation scores
        """
        logger.info(f"Performing {cv}-fold cross-validation")

        scores = cross_val_score(self.model, X, y, cv=cv, scoring="accuracy")

        logger.info(f"CV Scores: {scores}")
        logger.info(f"Mean CV Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

        return scores

    def predict(self, X):
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            Predictions
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X: Features

        Returns:
            Probability predictions
        """
        return self.model.predict_proba(X)

    def get_feature_importance(self, feature_names):
        """
        Get feature importance scores.

        Args:
            feature_names: List of feature names

        Returns:
            Dictionary of feature importances
        """
        importances = self.model.feature_importances_

        feature_importance = dict(zip(feature_names, importances))

        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        logger.info("Feature Importance:")
        for feature, importance in feature_importance.items():
            logger.info(f"  {feature}: {importance:.4f}")

        return feature_importance

    def save_model(self, filepath: str):
        """
        Save the trained model to a file.

        Args:
            filepath: Path to save the model
        """
        logger.info(f"Saving model to {filepath}")
        joblib.dump(self.model, filepath)
        logger.info("Model saved successfully")

    def load_model(self, filepath: str):
        """
        Load a trained model from a file.

        Args:
            filepath: Path to the saved model
        """
        logger.info(f"Loading model from {filepath}")
        self.model = joblib.load(filepath)
        logger.info("Model loaded successfully")
