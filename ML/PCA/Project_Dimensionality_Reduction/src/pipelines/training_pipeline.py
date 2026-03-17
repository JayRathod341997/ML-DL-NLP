"""Training pipeline for PCA dimensionality reduction"""
import numpy as np
import os
from pathlib import Path

from ..data.data_loader import DataLoader
from ..models.model import PCAReducer
from ..utils.logger import get_logger
from ..utils.config import load_config


logger = get_logger(__name__)


def train_pca(config=None):
    """Train PCA model on the dataset
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Trained PCA model and transformed data
    """
    config = config or load_config()
    logger.info("Starting PCA training pipeline")
    
    # Initialize data loader
    data_loader = DataLoader(config)
    
    # Load dataset
    dataset_name = config.get("data", {}).get("dataset", "iris")
    logger.info(f"Loading dataset: {dataset_name}")
    X, y = data_loader.load_builtin_dataset(dataset_name)
    
    # Preprocess data
    logger.info("Preprocessing data with StandardScaler")
    X_scaled = data_loader.preprocess(apply_scaling=True)
    
    # Initialize PCA
    n_components = config.get("model", {}).get("n_components", None)
    random_state = config.get("model", {}).get("random_state", 42)
    
    pca_model = PCAReducer(
        n_components=n_components,
        random_state=random_state
    )
    
    # Fit and transform
    logger.info("Fitting PCA model")
    X_transformed = pca_model.fit_transform(X_scaled)
    
    # Get PCA info
    pca_info = pca_model.get_pca_info()
    logger.info(f"PCA fitted with {pca_info['n_components']} components")
    logger.info(f"Explained variance ratio: {pca_info['explained_variance_ratio']}")
    logger.info(f"Cumulative variance: {pca_info['cumulative_variance']}")
    
    # Find optimal components for 95% variance
    optimal_components = pca_model.get_optimal_components(variance_threshold=0.95)
    logger.info(f"Optimal components for 95% variance: {optimal_components}")
    
    # Save model
    model_path = config.get("model", {}).get("save_path", "models/pca_model.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    pca_model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    return pca_model, X_transformed, y


def evaluate_pca(pca_model, X, y=None):
    """Evaluate PCA model performance
    
    Args:
        pca_model: Trained PCA model
        X: Original data
        y: Target labels (optional)
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating PCA model")
    
    # Transform data
    X_transformed = pca_model.transform(X)
    
    # Get info
    pca_info = pca_model.get_pca_info()
    
    # Calculate reconstruction error
    X_reconstructed = pca_model.inverse_transform(X_transformed)
    reconstruction_error = np.mean((X - X_reconstructed) ** 2)
    
    # Get feature importance
    feature_importance = pca_model.get_feature_importance()
    
    results = {
        "n_components": pca_info["n_components"],
        "explained_variance_ratio": pca_info["explained_variance_ratio"].tolist(),
        "cumulative_variance": pca_info["cumulative_variance"].tolist(),
        "reconstruction_error": float(reconstruction_error),
        "feature_importance": feature_importance.tolist() if hasattr(feature_importance, 'tolist') else feature_importance,
        "dimensionality_reduction": f"{X.shape[1]} -> {X_transformed.shape[1]}"
    }
    
    logger.info(f"Dimensionality reduction: {results['dimensionality_reduction']}")
    logger.info(f"Reconstruction error: {results['reconstruction_error']:.6f}")
    
    return results


def visualize_results(pca_model, X, y=None, output_dir="outputs"):
    """Generate visualization plots for PCA results
    
    Args:
        pca_model: Trained PCA model
        X: Original data
        y: Target labels (optional)
        output_dir: Directory to save plots
    """
    logger.info("Generating visualizations")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Transform data
        X_transformed = pca_model.transform(X)
        
        # Plot 1: 2D projection
        plt.figure(figsize=(10, 8))
        if y is not None:
            scatter = plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter)
        else:
            plt.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.7)
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('PCA: 2D Projection of Data')
        plt.savefig(os.path.join(output_dir, 'pca_2d_projection.png'))
        plt.close()
        
        # Plot 2: Explained variance
        pca_info = pca_model.get_pca_info()
        plt.figure(figsize=(10, 6))
        components = range(1, len(pca_info['explained_variance_ratio']) + 1)
        plt.bar(components, pca_info['explained_variance_ratio'], alpha=0.7, label='Individual')
        plt.plot(components, pca_info['cumulative_variance'], 'r-o', label='Cumulative')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA: Explained Variance by Component')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'pca_explained_variance.png'))
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")
        
    except ImportError:
        logger.warning("matplotlib not available, skipping visualizations")
