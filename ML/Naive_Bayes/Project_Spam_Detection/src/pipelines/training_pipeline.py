"""Training pipeline for spam detection"""

from ..data.data_loader import DataLoader
from ..models.model import NaiveBayesClassifier
from ..utils.logger import get_logger
from ..utils.config import load_config

logger = get_logger(__name__)


def train_model(config_path: str = "config/settings.yaml"):
    """Train the spam detection model"""
    logger.info("Starting Naive Bayes Training Pipeline")
    config = load_config(config_path)
    data_loader = DataLoader(config)

    train_path = config.get("data", {}).get("train_path", "data/raw/emails.csv")
    test_path = config.get("data", {}).get("test_path", "data/raw/test_emails.csv")

    X_train, X_test, y_train, y_test = data_loader.load_and_prepare_data(
        train_path, test_path
    )

    classifier = NaiveBayesClassifier(config)
    train_results = classifier.train(X_train, y_train)

    test_metrics = classifier.evaluate(X_test, y_test)
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")

    model_path = config.get("model_path", "models/naive_bayes_model.pkl")
    classifier.save(model_path)

    vectorizer_path = config.get("vectorizer_path", "models/vectorizer.pkl")
    data_loader.save_vectorizer(vectorizer_path)

    logger.info("Training completed!")
    return classifier
