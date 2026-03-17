"""
Configuration Management Module
================================
Handles loading and managing configuration from YAML files.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default settings.yaml

    Returns:
        Dictionary with configuration
    """
    if config_path is None:
        config_path = CONFIG_DIR / "settings.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        # Return default config
        return get_default_config()

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.

    Returns:
        Default configuration dictionary
    """
    return {
        "model": {
            "name": "LinearRegression",
            "params": {"fit_intercept": True, "normalize": False, "alpha": 1.0},
        },
        "data": {
            "train_path": "data/raw/train.csv",
            "test_path": "data/raw/test.csv",
            "target_column": "SalePrice",
            "test_size": 0.2,
            "random_state": 42,
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "preprocessing": {
            "handle_missing": True,
            "encode_categorical": True,
            "scale_features": True,
        },
    }


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        config_path: Path to save config
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
