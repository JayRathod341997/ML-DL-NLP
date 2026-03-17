"""Training pipeline for K-Means Customer Segmentation"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple

from ..data.data_loader import DataLoader
from ..models.model import KMeansClustering
from ..utils.logger import get_logger
from ..utils.config import load_config

logger = get_logger(__name__)


def train_model(config_path: str = "config/settings.yaml") -> KMeansClustering:
    """
    Train the K-Means model

    Args:
        config_path: Path to configuration file

    Returns:
        Trained KMeansClustering instance
    """
    logger.info("=" * 50)
    logger.info("Starting K-Means Training Pipeline")
    logger.info("=" * 50)

    # Load configuration
    config = load_config(config_path)

    # Initialize data loader
    data_loader = DataLoader(config)

    # Load and prepare data
    data_path = config.get("data", {}).get("train_path", "data/raw/customers.csv")

    logger.info(f"Loading data from {data_path}")
    X, df = data_loader.load_and_prepare_data(data_path)

    # Get feature names
    feature_names = config.get("features", {}).get("numerical", [])

    # Initialize and train model
    clustering = KMeansClustering(config)

    # Train with optimal K finding
    train_results = clustering.train(X, find_optimal_k=True)

    logger.info("Training results:")
    logger.info(f"  Optimal K: {train_results['optimal_k']}")
    logger.info(f"  Inertia: {train_results['inertia']:.4f}")

    metrics = train_results["metrics"]
    logger.info(f"  Silhouette Score: {metrics.get('silhouette_score', 0):.4f}")

    # Get cluster profiles
    profiles = clustering.get_cluster_profiles(X, feature_names)
    logger.info("Cluster Profiles:")
    for cluster_id, profile in profiles.items():
        logger.info(f"  Cluster {cluster_id}: {profile}")

    # Save model
    model_path = config.get("model_path", "models/kmeans_model.pkl")
    clustering.save(model_path)

    # Save scaler
    scaler_path = config.get("scaler_path", "models/scaler.pkl")
    data_loader.save_scaler(scaler_path)

    # Add cluster labels to original data
    df["cluster"] = clustering.labels_
    output_path = config.get("data", {}).get("output_path", "data/processed")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{output_path}/customers_with_clusters.csv", index=False)

    logger.info("Training pipeline completed successfully!")
    logger.info("=" * 50)

    return clustering


def evaluate_model(clustering: KMeansClustering, X: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate the trained model

    Args:
        clustering: Trained KMeansClustering
        X: Feature matrix

    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating model...")

    metrics = clustering.evaluate(X)

    logger.info("Evaluation Metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    return metrics


def predict(clustering: KMeansClustering, X: np.ndarray) -> np.ndarray:
    """
    Predict clusters for new data

    Args:
        clustering: Trained KMeansClustering
        X: Features to predict on

    Returns:
        Array of cluster labels
    """
    predictions = clustering.predict(X)

    return predictions
