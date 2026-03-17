"""Data loader module for spam detection"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Handles loading and preprocessing of email data for spam detection"""

    def __init__(self, config: dict):
        self.config = config
        self.preprocessing_config = config.get("preprocessing", {})
        self.features_config = config.get("features", {})
        self.vectorizer = None

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load email data from CSV file"""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} emails")
        return df

    def preprocess_text(self, text: str) -> str:
        """Preprocess text: lowercase, remove stopwords, stem"""
        if not isinstance(text, str):
            return ""

        text = text.lower()

        if self.preprocessing_config.get("remove_stopwords", True):
            try:
                from nltk.corpus import stopwords

                stop_words = set(stopwords.words("english"))
                words = text.split()
                words = [w for w in words if w not in stop_words]
                text = " ".join(words)
            except:
                pass

        if self.preprocessing_config.get("stemming", True):
            try:
                from nltk.stem import PorterStemmer

                stemmer = PorterStemmer()
                words = text.split()
                words = [stemmer.stem(w) for w in words]
                text = " ".join(words)
            except:
                pass

        return text

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess emails and create features"""
        logger.info("Preprocessing emails...")

        # Apply text preprocessing
        if "text" in df.columns:
            df["processed_text"] = df["text"].apply(self.preprocess_text)
        elif "message" in df.columns:
            df["processed_text"] = df["message"].apply(self.preprocess_text)
        else:
            raise ValueError("Text column not found")

        # Create vectorizer
        vectorizer_type = self.features_config.get("vectorizer", "tfidf")
        max_features = self.features_config.get("max_features", 5000)
        ngram_range = tuple(self.features_config.get("ngram_range", [1, 2]))

        if vectorizer_type == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=max_features, ngram_range=ngram_range
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=max_features, ngram_range=ngram_range
            )

        X = self.vectorizer.fit_transform(df["processed_text"])
        y = df["label"].values if "label" in df.columns else df["spam"].values

        logger.info(f"Feature matrix shape: {X.shape}")

        return X, y

    def load_and_prepare_data(self, train_path: str, test_path: str) -> Tuple:
        """Load and prepare training and test data"""
        train_df = self.load_data(train_path)
        test_df = self.load_data(test_path)

        X_train, y_train = self.preprocess(train_df)
        X_test, y_test = self.preprocess(test_df)

        return X_train, X_test, y_train, y_test

    def save_vectorizer(self, filepath: str):
        """Save the vectorizer"""
        import joblib

        joblib.dump(self.vectorizer, filepath)

    def load_vectorizer(self, filepath: str):
        """Load the vectorizer"""
        import joblib

        self.vectorizer = joblib.load(filepath)
