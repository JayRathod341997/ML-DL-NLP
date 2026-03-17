"""
Training Pipeline Module
Credit Scoring Project - Decision Tree
"""

import yaml
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.data_loader import DataLoader
from src.models.model import CreditDecisionTree
from src.utils.logger import get_logger
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = get_logger(__name__)


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config",
            "settings.yaml",
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Configuration loaded from {config_path}")
    return config


def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """
    Evaluate model performance.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
    """
    logger.info("Evaluating model performance")

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
    }

    if y_pred_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])

    for metric, value in metrics.items():
        logger.info(f"{metric.capitalize()}: {value:.4f}")

    logger.info("\nClassification Report:")
    logger.info(f"\n{classification_report(y_true, y_pred)}")

    logger.info("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"\n{cm}")

    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path: str = None):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Confusion matrix saved to {save_path}")

    plt.close()


def plot_feature_importance(feature_importance, save_path: str = None):
    """Plot and save feature importance."""
    plt.figure(figsize=(10, 6))

    features = list(feature_importance.keys())
    importance = list(feature_importance.values())

    plt.barh(features, importance)
    plt.xlabel("Importance")
    plt.title("Feature Importance")

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Feature importance plot saved to {save_path}")

    plt.close()


def run_pipeline(data_path: str = None, config_path: str = None):
    """
    Run the complete training pipeline.

    Args:
        data_path: Path to data file
        config_path: Path to config file
    """
    logger.info("=" * 50)
    logger.info("Starting Credit Scoring Training Pipeline")
    logger.info("=" * 50)

    # Load configuration
    config = load_config(config_path)

    # Initialize data loader
    logger.info("Step 1: Loading and preprocessing data")
    data_loader = DataLoader(config)

    if data_path is None:
        data_path = config.get("data", {}).get("source")

    X_train, X_test, y_train, y_test = data_loader.preprocess(data_path)

    # Initialize and train model
    logger.info("Step 2: Training Decision Tree model")
    model = CreditDecisionTree(config)

    # Optionally tune hyperparameters
    # model.tune_hyperparameters(X_train, y_train)

    model.train(X_train, y_train)

    # Cross-validation
    logger.info("Step 3: Performing cross-validation")
    cv_scores = model.cross_validate(X_train, y_train, cv=5)

    # Make predictions
    logger.info("Step 4: Making predictions on test set")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Evaluate model
    logger.info("Step 5: Evaluating model")
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)

    # Get feature importance
    logger.info("Step 6: Extracting feature importance")
    feature_names = X_train.columns.tolist()
    feature_importance = model.get_feature_importance(feature_names)

    # Save model
    logger.info("Step 7: Saving model")
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "models",
        "credit_decision_tree.joblib",
    )
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)

    # Generate plots
    logger.info("Step 8: Generating visualization plots")
    plots_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "plots"
    )
    os.makedirs(plots_dir, exist_ok=True)

    plot_confusion_matrix(
        y_test, y_pred, os.path.join(plots_dir, "confusion_matrix.png")
    )

    plot_feature_importance(
        feature_importance, os.path.join(plots_dir, "feature_importance.png")
    )

    logger.info("=" * 50)
    logger.info("Training Pipeline Completed Successfully!")
    logger.info("=" * 50)

    return model, metrics


if __name__ == "__main__":
    # Run the pipeline
    run_pipeline()
