"""Training pipeline for SVM Image Classification"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

from ..data.data_loader import DataLoader
from ..models.model import SVMClassifier
from ..utils.logger import get_logger
from ..utils.config import load_config

logger = get_logger(__name__)


def train_model(config_path: str = "config/settings.yaml") -> SVMClassifier:
    """
    Train the SVM model

    Args:
        config_path: Path to configuration file

    Returns:
        Trained SVMClassifier instance
    """
    logger.info("=" * 50)
    logger.info("Starting SVM Training Pipeline")
    logger.info("=" * 50)

    # Load configuration
    config = load_config(config_path)

    # Initialize data loader
    data_loader = DataLoader(config)

    # Load and prepare data
    train_path = config.get("data", {}).get("train_path", "data/raw/train")
    test_path = config.get("data", {}).get("test_path", "data/raw/test")

    logger.info(f"Loading data from {train_path} and {test_path}")
    X_train, X_test, y_train, y_test, label_encoder = data_loader.load_and_prepare_data(
        train_path, test_path
    )

    # Initialize and train model
    classifier = SVMClassifier(config)

    # Train with hyperparameter tuning
    train_results = classifier.train(X_train, y_train, tune_hyperparameters=True)

    logger.info("Training results:")
    logger.info(f"  Training Accuracy: {train_results['training_accuracy']:.4f}")
    logger.info(f"  CV Scores: {train_results['cv_scores']}")

    # Evaluate on test set
    test_metrics = classifier.evaluate(X_test, y_test)

    logger.info("Test Set Evaluation:")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall: {test_metrics['recall']:.4f}")
    logger.info(f"  F1-Score: {test_metrics['f1_score']:.4f}")

    # Save model
    model_path = config.get("model_path", "models/svm_model.pkl")
    classifier.save(model_path)

    # Save scaler
    scaler_path = config.get("scaler_path", "models/scaler.pkl")
    data_loader.save_scaler(scaler_path)

    logger.info("Training pipeline completed successfully!")
    logger.info("=" * 50)

    return classifier


def evaluate_model(
    classifier: SVMClassifier, X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluate the trained model

    Args:
        classifier: Trained SVMClassifier
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating model...")

    metrics = classifier.evaluate(X_test, y_test)

    logger.info("Evaluation Metrics:")
    for metric, value in metrics.items():
        if metric != "confusion_matrix":
            logger.info(f"  {metric}: {value:.4f}")

    return metrics


def predict(classifier: SVMClassifier, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions on new data

    Args:
        classifier: Trained SVMClassifier
        X: Features to predict on

    Returns:
        Tuple of (predictions, probabilities)
    """
    predictions = classifier.predict(X)
    probabilities = classifier.predict_proba(X)

    return predictions, probabilities
