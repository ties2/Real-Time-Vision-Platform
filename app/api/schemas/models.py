from pydantic import BaseModel


class ModelInfo(BaseModel):
    """Public model metadata."""

    name: str
    provider: str
    task: str
    version: str
    artifact: str
    device: str
    confidence_threshold: float
    loaded: bool


class ModelListResponse(BaseModel):
    """Response containing available models."""

    models: list[ModelInfo]