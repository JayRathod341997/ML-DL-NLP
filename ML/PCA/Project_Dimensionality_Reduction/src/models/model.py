"""PCA Model module for dimensionality reduction"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import explained_variance_ratio_
import joblib
import os


class PCAReducer:
    """Principal Component Analysis for dimensionality reduction"""

    def __init__(self, n_components=None, random_state=42):
        """Initialize PCA reducer

        Args:
            n_components: Number of components to keep
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state
        self.pca = None
        self.explained_variance_ratio_ = None
        self.components_ = None
        self.mean_ = None

    def fit(self, X):
        """Fit PCA to the data

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            self
        """
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca.fit(X)

        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.components_ = self.pca.components_
        self.mean_ = self.pca.mean_

        return self

    def transform(self, X):
        """Transform data using fitted PCA

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Transformed data (n_samples, n_components)
        """
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit() first.")
        return self.pca.transform(X)

    def fit_transform(self, X):
        """Fit PCA and transform data

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Transformed data (n_samples, n_components)
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        """Inverse transform reduced data back to original space

        Args:
            X_transformed: Transformed data (n_samples, n_components)

        Returns:
            Reconstructed data (n_samples, n_features)
        """
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit() first.")
        return self.pca.inverse_transform(X_transformed)

    def get_cumulative_variance(self, n_components=None):
        """Get cumulative explained variance

        Args:
            n_components: Number of components (None for all)

        Returns:
            Cumulative explained variance ratio
        """
        if self.explained_variance_ratio_ is None:
            raise ValueError("PCA not fitted. Call fit() first.")

        if n_components is None:
            n_components = len(self.explained_variance_ratio_)

        return np.sum(self.explained_variance_ratio_[:n_components])

    def get_optimal_components(self, variance_threshold=0.95):
        """Get number of components needed to reach variance threshold

        Args:
            variance_threshold: Minimum cumulative variance to retain

        Returns:
            Number of components needed
        """
        if self.explained_variance_ratio_ is None:
            raise ValueError("PCA not fitted. Call fit() first.")

        cumulative_variance = np.cumsum(self.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

        return n_components

    def get_feature_importance(self):
        """Get feature importance based on component loadings

        Returns:
            Dictionary with feature importance scores
        """
        if self.components_ is None:
            raise ValueError("PCA not fitted. Call fit() first.")

        importance = np.abs(self.components_).T @ self.explained_variance_ratio_
        return importance / importance.sum()

    def save(self, filepath):
        """Save PCA model to file

        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.pca, filepath)

    def load(self, filepath):
        """Load PCA model from file

        Args:
            filepath: Path to the saved model
        """
        self.pca = joblib.load(filepath)
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.components_ = self.pca.components_
        self.mean_ = self.pca.mean_

    def get_pca_info(self):
        """Get information about the fitted PCA

        Returns:
            Dictionary with PCA information
        """
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit() first.")

        return {
            "n_components": self.pca.n_components_,
            "n_features_in": self.pca.n_features_in_,
            "explained_variance": self.pca.explained_variance_,
            "explained_variance_ratio": self.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(self.explained_variance_ratio_),
            "singular_values": self.pca.singular_values_,
        }
