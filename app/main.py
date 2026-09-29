from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes.health import router as health_router
from app.api.routes.inference import router as inference_router
from app.api.routes.models import router as models_router
from app.core.config import get_settings
from app.core.exceptions import BBAPException
from app.core.logging import configure_logging, get_logger
from app.core.request_id import RequestIDMiddleware
from app.inference.engine import InferenceEngine
from app.models.loader import load_model_registry

settings = get_settings()

configure_logging(settings.log_level)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""

    logger.info(
        "Starting %s v%s",
        settings.app_name,
        settings.app_version,
    )

    registry = load_model_registry(
        config_path=settings.models_config_path,
        models_dir=settings.model_registry_path,
    )

    app.state.model_registry = registry
    inference_engine = InferenceEngine(
        registry=registry,
        max_batch_size=4,
        max_wait_ms=10,
    )
    await inference_engine.start()
    app.state.inference_engine = inference_engine

    logger.info(
        "Loaded models: %s",
        registry.list_models(),
    )

    inference_engine = InferenceEngine(
        registry=registry,
    )

    app.state.inference_engine = inference_engine

    await inference_engine.start()

    logger.info("Inference engine started")

    yield

    await inference_engine.stop()
    logger.info("Shutting down application")

    await inference_engine.stop()

    logger.info("Inference engine stopped")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=("Production-oriented real-time computer vision model serving platform."),
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    request_id = getattr(
        request.state,
        "request_id",
        "unknown",
    )

    logger.exception(
        "Unhandled exception request_id=%s",
        request_id,
    )

    return JSONResponse(
        status_code=500,
        content={
            "code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred.",
            "request_id": request_id,
        },
    )


@app.exception_handler(BBAPException)
async def bbap_exception_handler(
    request: Request,
    exc: BBAPException,
) -> JSONResponse:
    request_id = getattr(
        request.state,
        "request_id",
        "unknown",
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.code,
            "message": exc.message,
            "request_id": request_id,
        },
    )


app.add_middleware(RequestIDMiddleware)

app.include_router(health_router)
app.include_router(inference_router)
app.include_router(models_router)
