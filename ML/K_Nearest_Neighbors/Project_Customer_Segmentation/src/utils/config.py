"""Configuration utility for KNN Customer Segmentation project"""

import yaml
import os


def load_config(config_path="config/settings.yaml"):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        # Return default config
        return {
            "data": {
                "test_size": 0.2,
                "random_state": 42,
                "n_samples": 1000,
                "n_features": 5,
                "n_clusters": 4,
            },
            "model": {
                "n_neighbors": 5,
                "weights": "distance",
                "metric": "euclidean",
                "save_path": "models/knn_model.pkl",
            },
            "logging": {"level": "INFO"},
        }

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config
