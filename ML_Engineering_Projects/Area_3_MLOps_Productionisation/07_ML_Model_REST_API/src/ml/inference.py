from __future__ import annotations

import time

from ..models.response_schemas import PredictResponse, BatchPredictResponse
from .model_loader import get_model_registry


def predict(text: str) -> PredictResponse:
    """Run inference on a single text.

    Args:
        text: Input text string.

    Returns:
        PredictResponse with label, score, and inference time.
    """
    registry = get_model_registry()
    pipe = registry.get_pipeline()

    start = time.perf_counter()
    result = pipe(text)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    return PredictResponse(
        label=result["label"],
        score=round(float(result["score"]), 4),
        inference_time_ms=round(elapsed_ms, 2),
    )


def batch_predict(texts: list[str]) -> BatchPredictResponse:
    """Run inference on a list of texts.

    Args:
        texts: List of input text strings.

    Returns:
        BatchPredictResponse with individual predictions and total time.
    """
    registry = get_model_registry()
    pipe = registry.get_pipeline()

    start = time.perf_counter()
    results = pipe(texts)
    total_ms = (time.perf_counter() - start) * 1000

    predictions = [
        PredictResponse(
            label=r["label"],
            score=round(float(r["score"]), 4),
            inference_time_ms=round(total_ms / len(texts), 2),
        )
        for r in results
    ]
    return BatchPredictResponse(
        predictions=predictions,
        total_inference_time_ms=round(total_ms, 2),
    )
