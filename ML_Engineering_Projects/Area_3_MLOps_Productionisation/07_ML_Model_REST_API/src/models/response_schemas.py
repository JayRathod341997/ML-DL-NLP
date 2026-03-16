from pydantic import BaseModel


class PredictResponse(BaseModel):
    label: str
    score: float
    inference_time_ms: float


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    total_inference_time_ms: float


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    labels: list[str]
    device: str
