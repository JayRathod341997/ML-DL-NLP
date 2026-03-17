"""Training pipeline for Logistic Regression Disease Prediction"""

import os
from ..data.data_loader import DataLoader
from ..models.model import LogisticRegressionClassifier
from ..utils.logger import get_logger
from ..utils.config import load_config


logger = get_logger(__name__)


def train_model(config=None):
    """Train Logistic Regression model

    Args:
        config: Configuration dictionary

    Returns:
        Trained model and evaluation results
    """
    config = config or load_config()
    logger.info("Starting Logistic Regression training pipeline")

    # Initialize data loader
    data_loader = DataLoader(config)

    # Load dataset
    dataset_name = config.get("data", {}).get("dataset", "breast_cancer")
    logger.info(f"Loading dataset: {dataset_name}")
    X, y = data_loader.load_builtin_dataset(dataset_name)

    # Preprocess data
    logger.info("Preprocessing data with StandardScaler")
    X_scaled = data_loader.preprocess(apply_scaling=True)

    # Split data
    test_size = config.get("data", {}).get("test_size", 0.2)
    random_state = config.get("data", {}).get("random_state", 42)
    X_train, X_test, y_train, y_test = data_loader.split_data(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    # Initialize model
    model_params = config.get("model", {})
    model = LogisticRegressionClassifier(
        solver=model_params.get("solver", "lbfgs"),
        max_iter=model_params.get("max_iter", 1000),
        C=model_params.get("C", 1.0),
        class_weight=model_params.get("class_weight", "balanced"),
        random_state=model_params.get("random_state", 42),
    )

    # Train model
    logger.info("Training Logistic Regression model")
    model.fit(X_train, y_train)

    # Evaluate
    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)

    logger.info(f"Training metrics: {train_metrics}")
    logger.info(f"Test metrics: {test_metrics}")

    # Save model
    model_path = model_params.get("save_path", "models/logistic_model.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

    return model, test_metrics


def evaluate_model(model, X, y):
    """Evaluate model performance

    Args:
        model: Trained model
        X: Feature matrix
        y: True labels

    Returns:
        Evaluation metrics
    """
    logger.info("Evaluating model")
    metrics = model.evaluate(X, y)
    logger.info(f"Evaluation metrics: {metrics}")
    return metrics
