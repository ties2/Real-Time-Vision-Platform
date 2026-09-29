from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from app.api.dependencies import get_inference_engine
from app.api.schemas.inference import InferenceResponse
from app.core.exceptions import InvalidInputError
from app.inference.engine import InferenceEngine
from app.inference.postprocessing import postprocess_results
from app.inference.preprocessing import decode_image

router = APIRouter(
    prefix="/api/v1",
    tags=["inference"],
)


@router.post(
    "/inference",
    response_model=InferenceResponse,
)
async def inference(
        engine: Annotated[
            InferenceEngine,
            Depends(get_inference_engine),
        ],
        file: Annotated[UploadFile, File(...)],
        model: str = "yolo11",
) -> InferenceResponse:
    """Run object detection on an uploaded image."""

    if not file.content_type:
        raise InvalidInputError("Missing content type.")

    if not file.content_type.startswith("image/"):
        raise InvalidInputError("Only image files are supported.")

    # Validate model before processing the image.
    selected_model = engine.registry.get(model)

    data = await file.read()

    try:
        image = decode_image(data)
    except ValueError as exc:
        raise InvalidInputError(str(exc)) from exc

    result = await engine.submit(
        model_name=model,
        image=image,
    )

    metadata = selected_model.metadata()

    # result.output is ONE Ultralytics Results object (for this image only).
    detections = postprocess_results([result.output])

    image_height, image_width = image.shape[:2]

    return InferenceResponse(
        model=model,
        model_version=str(metadata["version"]),
        inference_time_ms=round(result.inference_time_ms, 2),
        batch_size=result.batch_size,
        image_width=image_width,
        image_height=image_height,
        detections=detections,
    )