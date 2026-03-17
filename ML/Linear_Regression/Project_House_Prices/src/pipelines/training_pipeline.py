"""
Training Pipeline Module
=========================
End-to-end training pipeline for Linear Regression model.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import DataLoader
from src.models.model import HousePriceModel, create_model
from src.utils.logger import get_logger, PipelineLogger
from src.utils.config import load_config

# Evaluation metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = get_logger(__name__)
pipeline_logger = PipelineLogger("training")


class TrainingPipeline:
    """
    Complete training pipeline for house price prediction.
    """

    def __init__(self, config_path: str = None):
        """Initialize training pipeline."""
        self.config = load_config(config_path)
        self.data_loader = DataLoader()
        self.model = None
        self.scaler = None
        self.metrics = {}

        logger.info("Training pipeline initialized")

    def run(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline.

        Returns:
            Dictionary with training results and metrics
        """
        pipeline_logger.log_pipeline_start(self.config)

        try:
            # Stage 1: Data Loading
            pipeline_logger.log_stage("Data Loading")
            df = self.data_loader.load_data()
            pipeline_logger.log_data_info("data_loading", df.shape, list(df.columns))

            # Stage 2: Data Preprocessing
            pipeline_logger.log_stage("Data Preprocessing")
            X, y = self.data_loader.preprocess_data(df, target_column="SalePrice")
            pipeline_logger.log_data_info("preprocessing", X.shape, list(X.columns))

            # Stage 3: Train-Test Split
            pipeline_logger.log_stage("Train-Test Split")
            data_config = self.config.get("data", {})
            X_train, X_test, y_train, y_test = self.data_loader.get_train_test_split(
                X,
                y,
                test_size=data_config.get("test_size", 0.2),
                random_state=data_config.get("random_state", 42),
            )
            pipeline_logger.log_data_info("train_set", X_train.shape)
            pipeline_logger.log_data_info("test_set", X_test.shape)

            # Stage 4: Feature Scaling
            pipeline_logger.log_stage("Feature Scaling")
            if self.config.get("preprocessing", {}).get("scale_features", True):
                self.scaler = self.data_loader.get_scaler(X_train.values)
                X_train_scaled = self.scaler.transform(X_train.values)
                X_test_scaled = self.scaler.transform(X_test.values)
            else:
                X_train_scaled = X_train.values
                X_test_scaled = X_test.values

            # Stage 5: Model Training
            pipeline_logger.log_stage("Model Training")
            self.model = create_model(self.config)
            pipeline_logger.log_model_info(
                self.model.model_type, self.model.model_params
            )

            self.model.fit(X_train_scaled, y_train, feature_names=list(X.columns))

            # Stage 6: Model Evaluation
            pipeline_logger.log_stage("Model Evaluation")
            y_train_pred = self.model.predict(X_train_scaled)
            y_test_pred = self.model.predict(X_test_scaled)

            # Calculate metrics
            train_metrics = self._calculate_metrics(y_train, y_train_pred, "Train")
            test_metrics = self._calculate_metrics(y_test, y_test_pred, "Test")

            self.metrics = {"train": train_metrics, "test": test_metrics}

            # Stage 7: Save Model
            pipeline_logger.log_stage("Model Saving")
            self._save_model()

            pipeline_logger.log_pipeline_end("SUCCESS")

            return {
                "status": "success",
                "metrics": self.metrics,
                "model_path": str(PROJECT_ROOT / "models" / "model.joblib"),
            }

        except Exception as e:
            pipeline_logger.log_error(e, "training_pipeline")
            pipeline_logger.log_pipeline_end("FAILED")
            raise

    def _calculate_metrics(self, y_true, y_pred, prefix: str) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metrics = {
            f"{prefix}_MSE": mse,
            f"{prefix}_RMSE": rmse,
            f"{prefix}_MAE": mae,
            f"{prefix}_R2": r2,
        }

        pipeline_logger.log_metrics(prefix, metrics)

        return metrics

    def _save_model(self):
        """Save trained model."""
        models_dir = PROJECT_ROOT / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        model_path = models_dir / "model.joblib"
        self.model.save(str(model_path))

        # Save scaler
        if self.scaler:
            scaler_path = models_dir / "scaler.joblib"
            import joblib

            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved to: {scaler_path}")


def main():
    """Main entry point."""
    logger.info("Starting training pipeline...")

    pipeline = TrainingPipeline()
    results = pipeline.run()

    logger.info(f"Training complete! Results: {results}")

    return results


if __name__ == "__main__":
    main()
