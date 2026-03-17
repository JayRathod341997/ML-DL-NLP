"""Naive Bayes Classifier model module"""

import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import joblib
from pathlib import Path
from typing import Dict, Any

from ..utils.logger import get_logger

logger = get_logger(__name__)


class NaiveBayesClassifier:
    """Naive Bayes classifier for spam detection"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config.get("model", {})
        variant = self.model_config.get("variant", "multinomial")
        alpha = self.model_config.get("alpha", 1.0)

        if variant == "multinomial":
            self.model = MultinomialNB(alpha=alpha)
        elif variant == "bernoulli":
            self.model = BernoulliNB(alpha=alpha)
        else:
            self.model = GaussianNB()

        self.is_trained_ = False

    def train(self, X_train, y_train):
        """Train the classifier"""
        logger.info("Training Naive Bayes classifier...")
        self.model.fit(X_train, y_train)
        self.is_trained_ = True

        train_pred = self.model.predict(X_train)
        accuracy = accuracy_score(y_train, train_pred)
        logger.info(f"Training accuracy: {accuracy:.4f}")

        return {"training_accuracy": accuracy}

    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        if not self.is_trained_:
            raise ValueError("Model not trained yet")

        y_pred = self.model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        return metrics

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)

    def save(self, filepath):
        """Save model"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)

    def load(self, filepath):
        """Load model"""
        self.model = joblib.load(filepath)
        self.is_trained_ = True
