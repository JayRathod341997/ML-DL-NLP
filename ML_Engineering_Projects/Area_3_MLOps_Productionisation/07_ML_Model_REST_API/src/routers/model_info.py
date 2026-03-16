from fastapi import APIRouter
from ..models.response_schemas import ModelInfoResponse
from ..ml.model_loader import get_model_registry

router = APIRouter(tags=["model"])


@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """Return metadata about the currently loaded model."""
    registry = get_model_registry()
    return ModelInfoResponse(
        model_name=registry.model_name,
        model_type=registry.model_type,
        labels=registry.labels,
        device=registry.device,
    )
