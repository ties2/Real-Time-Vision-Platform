from fastapi import APIRouter, Request

from app.api.schemas.models import ModelInfo, ModelListResponse

router = APIRouter(
    prefix="/api/v1",
    tags=["models"],
)


@router.get(
    "/models",
    response_model=ModelListResponse,
)
async def list_models(
    request: Request,
) -> ModelListResponse:
    """Return metadata for all loaded models."""

    registry = request.app.state.model_registry

    models = [ModelInfo(**metadata) for metadata in registry.list_metadata()]

    return ModelListResponse(models=models)
