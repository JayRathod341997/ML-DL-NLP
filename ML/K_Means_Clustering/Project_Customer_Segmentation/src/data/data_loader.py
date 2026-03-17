"""Data loader module for customer segmentation"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Handles loading and preprocessing of customer data for K-Means clustering"""

    def __init__(self, config: dict):
        """
        Initialize the DataLoader with configuration

        Args:
            config: Configuration dictionary containing data paths and parameters
        """
        self.config = config
        self.features_config = config.get("features", {})
        self.preprocessing_config = config.get("preprocessing", {})
        self.scaler = None

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load customer data from CSV file

        Args:
            filepath: Path to the CSV file

        Returns:
            DataFrame with customer data
        """
        logger.info(f"Loading data from {filepath}")

        file_path = Path(filepath)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")

        return df

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select features for clustering

        Args:
            df: Input dataframe

        Returns:
            DataFrame with selected features
        """
        numerical_features = self.features_config.get("numerical", [])
        categorical_features = self.features_config.get("categorical", [])

        selected_features = numerical_features + categorical_features
        missing_features = [f for f in selected_features if f not in df.columns]

        if missing_features:
            logger.warning(f"Missing features: {missing_features}")

        available_features = [f for f in selected_features if f in df.columns]
        logger.info(f"Using features: {available_features}")

        return df[available_features]

    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess the data for clustering

        Args:
            df: Input dataframe

        Returns:
            Preprocessed feature matrix
        """
        logger.info("Preprocessing data...")

        # Select features
        df_selected = self.select_features(df)

        # Handle missing values
        if df_selected.isnull().any().any():
            logger.warning("Missing values found, filling with median")
            df_selected = df_selected.fillna(df_selected.median())

        # Convert to numpy array
        X = df_selected.values

        # Scale features
        if self.preprocessing_config.get("scale", True):
            scale_method = self.preprocessing_config.get("scale_method", "standard")

            if scale_method == "standard":
                self.scaler = StandardScaler()
            elif scale_method == "minmax":
                self.scaler = MinMaxScaler()
            else:
                self.scaler = StandardScaler()

            X = self.scaler.fit_transform(X)
            logger.info(f"Scaled features using {scale_method} scaling")

        logger.info(f"Preprocessed data shape: {X.shape}")

        return X

    def load_and_prepare_data(self, filepath: str) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load and prepare data for clustering

        Args:
            filepath: Path to the data file

        Returns:
            Tuple of (feature matrix, original dataframe)
        """
        df = self.load_data(filepath)
        X = self.preprocess(df)

        return X, df

    def save_scaler(self, filepath: str):
        """Save the feature scaler"""
        import joblib

        joblib.dump(self.scaler, filepath)
        logger.info(f"Scaler saved to {filepath}")

    def load_scaler(self, filepath: str):
        """Load the feature scaler"""
        import joblib

        self.scaler = joblib.load(filepath)
        logger.info(f"Scaler loaded from {filepath}")
