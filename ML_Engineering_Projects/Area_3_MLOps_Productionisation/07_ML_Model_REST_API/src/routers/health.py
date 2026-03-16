from fastapi import APIRouter, HTTPException
from ..models.response_schemas import HealthResponse
from ..ml.model_loader import get_model_registry

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness check — always returns ok if the server is running."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=HealthResponse)
async def ready() -> HealthResponse:
    """Readiness check — verifies the model is loaded and ready to serve."""
    registry = get_model_registry()
    if not registry.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return HealthResponse(status="ready")
