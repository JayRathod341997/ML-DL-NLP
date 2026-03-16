from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from .middleware.logging import RequestLoggingMiddleware
from .middleware.timing import TimingMiddleware
from .ml.model_loader import get_model_registry
from .routers import health, model_info, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    get_model_registry()  # warm up model
    yield
    # Nothing to clean up for this simple server


def create_app() -> FastAPI:
    app = FastAPI(
        title="ML Model API",
        description="Production inference server for HuggingFace text classification models",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Middleware (order matters: outermost runs first)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Prometheus metrics at /metrics
    Instrumentator().instrument(app).expose(app)

    # Routers
    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(model_info.router)

    return app


app = create_app()
