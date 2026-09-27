from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.api.routes.health import router as health_router
from app.api.routes.inference import router as inference_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.models.loader import load_model_registry
from app.api.routes.models import router as models_router
from app.core.request_id import RequestIDMiddleware


from app.api.schemas.errors import ErrorResponse
from app.core.exceptions import BBAPException

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

    logger.info(
        "Loaded models: %s",
        registry.list_models(),
    )

    yield

    logger.info("Shutting down application")


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
    #
    # return JSONResponse(
    #     status_code=400,
    #     content=error.model_dump(),
    # )


app.add_middleware(RequestIDMiddleware)
app.include_router(health_router)
app.include_router(inference_router)
app.include_router(models_router)