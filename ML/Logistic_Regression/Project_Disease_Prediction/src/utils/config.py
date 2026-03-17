"""Configuration utilities"""

import yaml
import os


DEFAULT_CONFIG = {
    "data": {"dataset": "breast_cancer", "test_size": 0.2, "random_state": 42},
    "model": {
        "solver": "lbfgs",
        "max_iter": 1000,
        "C": 1.0,
        "save_path": "models/logistic_model.pkl",
    },
    "preprocessing": {"apply_scaling": True},
    "logging": {"level": "INFO", "log_dir": "logs"},
}


def load_config(config_path="config/settings.yaml"):
    """Load configuration from YAML file"""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return {**DEFAULT_CONFIG, **config}
    return DEFAULT_CONFIG.copy()


def save_config(config, config_path="config/settings.yaml"):
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
