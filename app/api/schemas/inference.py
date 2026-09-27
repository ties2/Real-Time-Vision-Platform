from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Object bounding box."""

    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    """Single object detection."""

    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox


class InferenceResponse(BaseModel):
    """Structured inference response."""

    model: str
    model_version: str
    inference_time_ms: float
    image_width: int
    image_height: int
    detections: list[Detection]
