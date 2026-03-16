from __future__ import annotations

import os
from functools import lru_cache

from transformers import pipeline


class ModelRegistry:
    """Singleton model registry — loads model once at startup."""

    _instance: "ModelRegistry | None" = None
    _pipeline = None
    _model_name: str = ""
    _model_type: str = ""
    _labels: list[str] = []
    _device: str = "cpu"

    def __new__(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self) -> None:
        """Load model from environment config."""
        self._model_name = os.getenv(
            "MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english"
        )
        self._model_type = os.getenv("MODEL_TYPE", "text-classification")
        labels_env = os.getenv("LABELS", "NEGATIVE,POSITIVE")
        self._labels = [l.strip() for l in labels_env.split(",")]
        self._device = os.getenv("DEVICE", "cpu")
        device_id = 0 if self._device == "cuda" else -1

        self._pipeline = pipeline(
            self._model_type,
            model=self._model_name,
            device=device_id,
        )

    def get_pipeline(self):
        if self._pipeline is None:
            self.load()
        return self._pipeline

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def labels(self) -> list[str]:
        return self._labels

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None


@lru_cache(maxsize=1)
def get_model_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.load()
    return registry
