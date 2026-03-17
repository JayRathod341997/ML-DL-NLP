"""K-Means Clustering model module"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from ..utils.logger import get_logger

logger = get_logger(__name__)


class KMeansClustering:
    """K-Means clustering model for customer segmentation"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize K-Means Clustering model

        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        self.model_config = config.get("model", {})

        # Initialize model with configuration
        self.model = KMeans(
            n_clusters=self.model_config.get("n_clusters", 5),
            init=self.model_config.get("init", "k-means++"),
            n_init=self.model_config.get("n_init", 10),
            max_iter=self.model_config.get("max_iter", 300),
            tol=self.model_config.get("tol", 0.0001),
            algorithm=self.model_config.get("algorithm", "lloyd"),
            random_state=42,
        )

        self.n_clusters_ = self.model_config.get("n_clusters", 5)
        self.cluster_centers_ = None
        self.labels_ = None
        self.is_trained_ = False

    def train(self, X: np.ndarray, find_optimal_k: bool = False) -> Dict[str, Any]:
        """
        Train the K-Means model

        Args:
            X: Training features
            find_optimal_k: Whether to find optimal K using elbow/silhouette

        Returns:
            Dictionary containing training results
        """
        logger.info("Starting K-Means training...")

        results = {"inertia": 0.0, "metrics": {}, "optimal_k": self.n_clusters_}

        if find_optimal_k:
            logger.info("Finding optimal number of clusters...")
            optimal_k = self._find_optimal_k(X)
            results["optimal_k"] = optimal_k
            self.n_clusters_ = optimal_k
            self.model = KMeans(
                n_clusters=optimal_k,
                init=self.model_config.get("init", "k-means++"),
                n_init=self.model_config.get("n_init", 10),
                max_iter=self.model_config.get("max_iter", 300),
                random_state=42,
            )

        # Train final model
        logger.info(f"Training K-Means with {self.n_clusters_} clusters...")
        self.model.fit(X)

        # Get cluster labels and centers
        self.labels_ = self.model.labels_
        self.cluster_centers_ = self.model.cluster_centers_

        results["inertia"] = self.model.inertia_

        # Calculate clustering metrics
        metrics = self.evaluate(X)
        results["metrics"] = metrics

        logger.info(f"Training completed. Inertia: {self.model.inertia_:.4f}")
        logger.info(f"Silhouette Score: {metrics['silhouette_score']:.4f}")

        self.is_trained_ = True
        return results

    def _find_optimal_k(self, X: np.ndarray) -> int:
        """
        Find optimal number of clusters using elbow method and silhouette score

        Args:
            X: Feature matrix

        Returns:
            Optimal number of clusters
        """
        hyperparameters = self.config.get("hyperparameters", {})
        k_range = hyperparameters.get("n_clusters_range", [2, 10])

        inertias = []
        silhouettes = []
        K_range = range(k_range[0], k_range[1] + 1)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(X, kmeans.labels_))

        # Find optimal K using silhouette score
        optimal_k = K_range[np.argmax(silhouettes)]
        logger.info(f"Optimal K based on silhouette score: {optimal_k}")

        return optimal_k

    def evaluate(self, X: np.ndarray) -> Dict[str, float]:
        """
        Evaluate clustering quality

        Args:
            X: Feature matrix

        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained_:
            raise ValueError("Model has not been trained yet")

        labels = self.model.labels_

        metrics = {
            "silhouette_score": silhouette_score(X, labels),
            "calinski_harabasz_score": calinski_harabasz_score(X, labels),
            "davies_bouldin_score": davies_bouldin_score(X, labels),
        }

        logger.info(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
        logger.info(
            f"Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.4f}"
        )
        logger.info(f"Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f}")

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data

        Args:
            X: Features to predict on

        Returns:
            Array of cluster labels
        """
        if not self.is_trained_:
            raise ValueError("Model has not been trained yet")

        return self.model.predict(X)

    def get_cluster_profiles(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Dict[int, Dict[str, float]]:
        """
        Get profiles for each cluster

        Args:
            X: Feature matrix
            feature_names: Names of features

        Returns:
            Dictionary mapping cluster IDs to their profiles
        """
        if not self.is_trained_:
            raise ValueError("Model has not been trained yet")

        profiles = {}
        for cluster_id in range(self.n_clusters_):
            cluster_mask = self.labels_ == cluster_id
            cluster_data = X[cluster_mask]

            profile = {}
            for i, feature_name in enumerate(feature_names):
                profile[feature_name] = float(cluster_data[:, i].mean())

            profiles[cluster_id] = profile

        return profiles

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
        self.n_clusters_ = self.model.n_clusters
        self.is_trained_ = True
        logger.info(f"Model loaded from {filepath}")
