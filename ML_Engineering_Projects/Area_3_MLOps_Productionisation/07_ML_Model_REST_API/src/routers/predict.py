from fastapi import APIRouter, HTTPException

from ..models.request_schemas import PredictRequest, BatchPredictRequest
from ..models.response_schemas import PredictResponse, BatchPredictResponse
from ..ml.inference import predict, batch_predict

router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictResponse)
async def predict_endpoint(request: PredictRequest) -> PredictResponse:
    """Predict the label for a single text input."""
    try:
        return predict(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def batch_predict_endpoint(request: BatchPredictRequest) -> BatchPredictResponse:
    """Predict labels for a batch of texts (max 32)."""
    try:
        return batch_predict(request.texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {str(e)}")
