"""Configuration utilities"""

import yaml
import os

DEFAULT_CONFIG = {
    "data": {"dataset": "creditcard", "test_size": 0.2, "random_state": 42},
    "model": {"n_estimators": 100, "max_depth": 10, "save_path": "models/rf_model.pkl"},
    "preprocessing": {"apply_scaling": True},
    "logging": {"level": "INFO", "log_dir": "logs"},
}


def load_config(config_path="config/settings.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return {**DEFAULT_CONFIG, **yaml.safe_load(f)}
    return DEFAULT_CONFIG.copy()


def save_config(config, config_path="config/settings.yaml"):
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
