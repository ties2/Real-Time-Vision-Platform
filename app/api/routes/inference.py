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
# async def inference(
#     file: UploadFile = File(...),
#     model: str = "yolo11",
#     engine: InferenceEngine = Depends(get_inference_engine),
# ) -> InferenceResponse:

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

    result = engine.predict(
        model_name=model,
        image=image,
    )

    metadata = selected_model.metadata()

    detections = postprocess_results(result.results)

    return InferenceResponse(
        model=model,
        model_version=str(metadata["version"]),
        inference_time_ms=result.inference_time_ms,
        image_width=result.image_width,
        image_height=result.image_height,
        detections=detections,
    )
