import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


@pytest.fixture(scope="module")
def client():
    """TestClient with mocked model registry — no real model download needed."""
    mock_registry = MagicMock()
    mock_registry.is_loaded = True
    mock_registry.model_name = "mock-model"
    mock_registry.model_type = "text-classification"
    mock_registry.labels = ["NEGATIVE", "POSITIVE"]
    mock_registry.device = "cpu"
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = [{"label": "POSITIVE", "score": 0.99}]
    mock_registry.get_pipeline.return_value = mock_pipeline

    with patch("src.ml.model_loader.get_model_registry", return_value=mock_registry):
        with patch("src.ml.inference.get_model_registry", return_value=mock_registry):
            from src.app import create_app
            app = create_app()
            with TestClient(app) as c:
                yield c
