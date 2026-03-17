"""
Data Loading and Preprocessing Module
=====================================
This module handles data downloading, loading, and preprocessing for the House Price Prediction model.

Dataset Source: Kaggle House Prices Competition
URL: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
Alternative: scikit-learn fetch_openml
"""

import os
import sys
import logging
import urllib.request
import zipfile
from typing import Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.config import load_config

# Initialize logger
logger = get_logger(__name__)


class DataLoader:
    """
    Handles data loading, preprocessing, and feature engineering.

    Attributes:
        config: Configuration dictionary
        data_path: Path to data directory
    """

    # Dataset URLs
    DATASET_URLS = {
        "house_prices": "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv",
        "boston": "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv",
    }

    def __init__(self, config_path: Optional[str] = None):
        """Initialize DataLoader with configuration."""
        self.config = load_config(config_path)
        self.data_path = PROJECT_ROOT / "data" / "raw"
        self.data_path.mkdir(parents=True, exist_ok=True)

        logger.info("DataLoader initialized successfully")
        logger.info(f"Data path: {self.data_path}")

    def download_dataset(self, dataset_name: str = "house_prices") -> str:
        """
        Download dataset from URL.

        Args:
            dataset_name: Name of the dataset to download

        Returns:
            Path to downloaded file
        """
        logger.info(f"Downloading dataset: {dataset_name}")

        if dataset_name not in self.DATASET_URLS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        url = self.DATASET_URLS[dataset_name]
        filename = f"{dataset_name}.csv"
        filepath = self.data_path / filename

        try:
            # Download with progress logging
            urllib.request.urlretrieve(url, filepath)
            logger.info(f"Dataset downloaded successfully to: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise

    def load_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            filepath: Path to CSV file

        Returns:
            DataFrame with loaded data
        """
        if filepath is None:
            # Try to load default dataset
            default_path = self.data_path / "house_prices.csv"
            if default_path.exists():
                filepath = str(default_path)
            else:
                logger.info("No local data found, downloading...")
                filepath = self.download_dataset()

        logger.info(f"Loading data from: {filepath}")

        try:
            df = pd.read_csv(filepath)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with handled missing values
        """
        logger.info("Handling missing values...")

        # Log missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({"count": missing, "percentage": missing_pct})
        missing_df = missing_df[missing_df["count"] > 0]

        if len(missing_df) > 0:
            logger.info(f"Missing values found:\n{missing_df}")

        # Numerical columns: fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy="median")
        df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

        # Categorical columns: fill with mode
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)

        logger.info("Missing values handled successfully")
        return df

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with encoded categorical variables
        """
        logger.info("Encoding categorical variables...")

        categorical_cols = df.select_dtypes(include=["object"]).columns
        label_encoders = {}

        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            logger.debug(f"Encoded column: {col}, unique values: {len(le.classes_)}")

        logger.info(f"Encoded {len(categorical_cols)} categorical columns")
        return df

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature engineering.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with engineered features
        """
        logger.info("Performing feature engineering...")

        # Create new features
        if "GrLivArea" in df.columns and "TotalBsmtSF" in df.columns:
            df["TotalSF"] = df["GrLivArea"] + df["TotalBsmtSF"]
            logger.debug("Created feature: TotalSF")

        if "YearBuilt" in df.columns:
            df["Age"] = pd.Timestamp.now().year - df["YearBuilt"]
            logger.debug("Created feature: Age")

        if "GarageArea" in df.columns and "GarageCars" in df.columns:
            df["GarageScore"] = df["GarageArea"] * df["GarageCars"]
            logger.debug("Created feature: GarageScore")

        logger.info(f"Feature engineering complete. New shape: {df.shape}")
        return df

    def preprocess_data(
        self, df: pd.DataFrame, target_column: str = "SalePrice"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Full preprocessing pipeline.

        Args:
            df: Input DataFrame
            target_column: Name of target column

        Returns:
            Tuple of (features, target)
        """
        logger.info("Starting full preprocessing pipeline...")

        # Handle missing values
        df = self.handle_missing_values(df)

        # Feature engineering
        df = self.feature_engineering(df)

        # Separate target
        if target_column in df.columns:
            y = df[target_column]
            X = df.drop(columns=[target_column])
        else:
            logger.warning(
                f"Target column '{target_column}' not found. Using last column as target."
            )
            y = df.iloc[:, -1]
            X = df.iloc[:, :-1]

        # Encode categorical variables
        X = self.encode_categorical(X)

        logger.info(f"Preprocessing complete. X shape: {X.shape}, y shape: {y.shape}")

        return X, y

    def get_train_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple:
        """
        Split data into train and test sets.

        Args:
            X: Features
            y: Target
            test_size: Proportion of test set
            random_state: Random seed

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(
            f"Splitting data with test_size={test_size}, random_state={random_state}"
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        logger.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

        return X_train, X_test, y_train, y_test

    def get_scaler(self, X_train: np.ndarray) -> StandardScaler:
        """
        Get and fit scaler for numerical features.

        Args:
            X_train: Training features

        Returns:
            Fitted StandardScaler
        """
        logger.info("Fitting StandardScaler...")
        scaler = StandardScaler()
        scaler.fit(X_train)
        logger.info(
            f"Scaler fitted. Mean: {scaler.mean_[:5]}..., Scale: {scaler.scale_[:5]}..."
        )
        return scaler


def main():
    """Main function to demonstrate data loading."""
    # Initialize DataLoader
    loader = DataLoader()

    # Load and preprocess data
    df = loader.load_data()
    X, y = loader.preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test = loader.get_train_test_split(X, y)

    # Get scaler
    scaler = loader.get_scaler(X_train.values)

    logger.info("Data loading completed successfully!")

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    main()
