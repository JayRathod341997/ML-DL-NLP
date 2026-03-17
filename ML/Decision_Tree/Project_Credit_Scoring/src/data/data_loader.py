"""
Data Loading and Preprocessing Module
Credit Scoring Project - Decision Tree
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data loading and preprocessing for credit scoring model."""

    def __init__(self, config: dict):
        """
        Initialize DataLoader with configuration.

        Args:
            config: Dictionary containing data configuration
        """
        self.config = config
        self.target_column = config.get("data", {}).get("target_column", "default")
        self.test_size = config.get("data", {}).get("test_size", 0.2)
        self.random_state = config.get("data", {}).get("random_state", 42)
        self.handle_missing = config.get("data", {}).get("handle_missing", "mean")

    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """
        Load data from CSV file or URL.

        Args:
            filepath: Path to CSV file or URL

        Returns:
            DataFrame containing the data
        """
        logger.info(f"Loading data from {filepath}")

        try:
            if filepath.endswith(".csv"):
                df = pd.read_csv(filepath)
            elif filepath.endswith(".xlsx"):
                df = pd.read_excel(filepath)
            else:
                df = pd.read_csv(filepath)

            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with handled missing values
        """
        logger.info("Handling missing values")

        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            logger.warning(f"Missing values found:\n{missing[missing > 0]}")

            # Handle numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    if self.handle_missing == "mean":
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif self.handle_missing == "median":
                        df[col].fillna(df[col].median(), inplace=True)
                    logger.debug(f"Filled {col} with {self.handle_missing}")

            # Handle categorical columns
            cat_cols = df.select_dtypes(include=["object"]).columns
            for col in cat_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].mode()[0], inplace=True)
                    logger.debug(f"Filled {col} with mode")
        else:
            logger.info("No missing values found")

        return df

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with encoded categorical variables
        """
        logger.info("Encoding categorical variables")

        cat_cols = df.select_dtypes(include=["object"]).columns

        for col in cat_cols:
            if col != self.target_column:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                logger.debug(f"Encoded column: {col}")

        return df

    def split_data(self, df: pd.DataFrame) -> tuple:
        """
        Split data into features and target, then train/test.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Splitting data into train/test sets")

        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target distribution:\n{y.value_counts()}")

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        logger.info(f"Training set size: {X_train.shape[0]}")
        logger.info(f"Test set size: {X_test.shape[0]}")

        return X_train, X_test, y_train, y_test

    def preprocess(self, filepath: str = None) -> tuple:
        """
        Full preprocessing pipeline.

        Args:
            filepath: Path to data file

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Starting preprocessing pipeline")

        # Load data
        df = self.load_data(filepath)

        # Handle missing values
        df = self.handle_missing_values(df)

        # Encode categorical
        df = self.encode_categorical(df)

        # Split data
        X_train, X_test, y_train, y_test = self.split_data(df)

        logger.info("Preprocessing completed successfully")

        return X_train, X_test, y_train, y_test
