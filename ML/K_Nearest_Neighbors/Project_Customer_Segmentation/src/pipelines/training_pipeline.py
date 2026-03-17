"""Training pipeline for KNN Customer Segmentation"""

import os
from ..data.data_loader import DataLoader
from ..models.model import KNNClassifier
from ..utils.logger import get_logger
from ..utils.config import load_config


def train_model(config_path=None):
    """Train the KNN model for customer segmentation"""
    logger = get_logger(__name__)
    config = load_config(config_path) if config_path else load_config()

    logger.info("Starting KNN model training")

    # Load data
    data_loader = DataLoader(config)
    X_train, X_test, y_train, y_test = data_loader.get_train_test_data()

    # Initialize model
    model = KNNClassifier(
        n_neighbors=config.get("model", {}).get("n_neighbors", 5),
        weights=config.get("model", {}).get("weights", "distance"),
        metric=config.get("model", {}).get("metric", "euclidean"),
    )

    # Train
    logger.info("Training KNN model")
    model.fit(X_train, y_train)

    # Save model
    model_path = config.get("model", {}).get("save_path", "models/knn_model.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model"""
    logger = get_logger(__name__)
    logger.info("Evaluating model")

    metrics = model.evaluate(X_test, y_test)
    logger.info(f"Evaluation metrics: {metrics}")

    return metrics
