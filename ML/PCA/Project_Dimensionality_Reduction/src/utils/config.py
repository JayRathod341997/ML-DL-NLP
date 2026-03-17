"""Configuration utilities for PCA Dimensionality Reduction project"""

import yaml
import os


DEFAULT_CONFIG = {
    "data": {"dataset": "iris", "test_size": 0.2, "random_state": 42},
    "model": {
        "n_components": None,
        "random_state": 42,
        "save_path": "models/pca_model.pkl",
    },
    "preprocessing": {"apply_scaling": True, "scaler_type": "standard"},
    "visualization": {"output_dir": "outputs", "save_plots": True},
    "logging": {"level": "INFO", "log_dir": "logs"},
}


def load_config(config_path="config/settings.yaml"):
    """Load configuration from YAML file

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Merge with default config
        return merge_configs(DEFAULT_CONFIG, config)

    return DEFAULT_CONFIG.copy()


def merge_configs(default, override):
    """Merge two configuration dictionaries

    Args:
        default: Default configuration
        override: Override configuration

    Returns:
        Merged configuration
    """
    result = default.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def save_config(config, config_path="config/settings.yaml"):
    """Save configuration to YAML file

    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def get_config_value(config, key_path, default=None):
    """Get configuration value by dot-separated key path

    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., "model.n_components")
        default: Default value if key not found

    Returns:
        Configuration value
    """
    keys = key_path.split(".")
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value
