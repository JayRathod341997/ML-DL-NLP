"""Training pipeline for Random Forest Fraud Detection"""

import os
from ..data.data_loader import DataLoader
from ..models.model import RandomForestClassifier
from ..utils.logger import get_logger
from ..utils.config import load_config


logger = get_logger(__name__)


def train_model(config=None):
    """Train Random Forest model"""
    config = config or load_config()
    logger.info("Starting Random Forest training pipeline")

    data_loader = DataLoader(config)
    X, y = data_loader.load_builtin_dataset("creditcard")
    X_scaled = data_loader.preprocess(apply_scaling=True)

    test_size = config.get("data", {}).get("test_size", 0.2)
    random_state = config.get("data", {}).get("random_state", 42)
    X_train, X_test, y_train, y_test = data_loader.split_data(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    model_params = config.get("model", {})
    model = RandomForestClassifier(
        n_estimators=model_params.get("n_estimators", 100),
        max_depth=model_params.get("max_depth", 10),
        class_weight=model_params.get("class_weight", "balanced"),
        random_state=model_params.get("random_state", 42),
    )

    model.fit(X_train, y_train)

    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)

    logger.info(f"Training metrics: {train_metrics}")
    logger.info(f"Test metrics: {test_metrics}")

    model_path = model_params.get("save_path", "models/rf_model.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

    return model, test_metrics


def evaluate_model(model, X, y):
    """Evaluate model performance"""
    return model.evaluate(X, y)
