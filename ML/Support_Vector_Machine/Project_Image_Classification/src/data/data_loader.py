"""Data loader module for image classification"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Handles loading and preprocessing of image data for SVM classification"""

    def __init__(self, config: dict):
        """
        Initialize the DataLoader with configuration

        Args:
            config: Configuration dictionary containing data paths and parameters
        """
        self.config = config
        self.image_size = tuple(config.get("features", {}).get("image_size", [64, 64]))
        self.color_space = config.get("features", {}).get("color_space", "RGB")
        self.histogram_bins = config.get("features", {}).get("histogram_bins", 256)
        self.label_encoder = LabelEncoder()
        self.scaler = None

    def load_images_from_directory(
        self, directory: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load images from a directory structure where each subdirectory represents a class

        Args:
            directory: Path to the root directory containing class subdirectories

        Returns:
            Tuple of (images, labels)
        """
        images = []
        labels = []

        directory_path = Path(directory)
        if not directory_path.exists():
            logger.warning(f"Directory {directory} does not exist")
            return np.array(images), np.array(labels)

        class_dirs = [d for d in directory_path.iterdir() if d.is_dir()]
        logger.info(f"Found {len(class_dirs)} classes in {directory}")

        for class_dir in class_dirs:
            class_name = class_dir.name
            logger.info(f"Processing class: {class_name}")

            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
                    try:
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            img = self._preprocess_image(img)
                            images.append(img)
                            labels.append(class_name)
                    except Exception as e:
                        logger.error(f"Error loading {img_file}: {e}")

        return np.array(images), np.array(labels)

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess a single image

        Args:
            image: Input image as numpy array

        Returns:
            Preprocessed image
        """
        # Resize to target size
        image = cv2.resize(image, self.image_size)

        # Convert color space if needed
        if self.color_space == "HSV":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.color_space == "Grayscale":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image

    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract features from images using histogram and HOG

        Args:
            images: Array of images

        Returns:
            Feature matrix
        """
        features = []

        for img in images:
            feature_vector = []

            # Handle grayscale images
            if len(img.shape) == 2:
                # Histogram features
                hist = cv2.calcHist([img], [0], None, [self.histogram_bins], [0, 256])
                hist = hist.flatten()
                feature_vector.extend(hist)
            else:
                # Color histogram for each channel
                for channel in range(img.shape[2]):
                    hist = cv2.calcHist(
                        [img], [channel], None, [self.histogram_bins], [0, 256]
                    )
                    hist = hist.flatten()
                    feature_vector.extend(hist)

            features.append(feature_vector)

        return np.array(features)

    def load_and_prepare_data(self, train_path: str, test_path: str) -> Tuple:
        """
        Load and prepare training and test data

        Args:
            train_path: Path to training data directory
            test_path: Path to test data directory

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, label_encoder)
        """
        logger.info("Loading training data...")
        train_images, train_labels = self.load_images_from_directory(train_path)

        logger.info("Loading test data...")
        test_images, test_labels = self.load_images_from_directory(test_path)

        if len(train_images) == 0 or len(test_images) == 0:
            logger.error("No data loaded. Please check your data paths.")
            raise ValueError("No data found in specified directories")

        # Extract features
        logger.info("Extracting features from training images...")
        X_train = self.extract_features(train_images)

        logger.info("Extracting features from test images...")
        X_test = self.extract_features(test_images)

        # Encode labels
        all_labels = np.concatenate([train_labels, test_labels])
        self.label_encoder.fit(all_labels)

        y_train = self.label_encoder.transform(train_labels)
        y_test = self.label_encoder.transform(test_labels)

        # Scale features
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        logger.info(f"Classes: {self.label_encoder.classes_}")

        return X_train, X_test, y_train, y_test, self.label_encoder

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and validation sets

        Args:
            X: Feature matrix
            y: Labels
            test_size: Proportion of data to use for testing
            random_state: Random seed

        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        logger.info(
            f"Train set size: {len(X_train)}, Validation set size: {len(X_val)}"
        )

        return X_train, X_val, y_train, y_val

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
