from typing import Any

from app.api.schemas.inference import (
    BoundingBox,
    Detection,
)


def postprocess_results(
    results: Any,
) -> list[Detection]:
    """Convert Ultralytics results into API detections."""

    detections: list[Detection] = []

    for result in results:
        if result.boxes is None:
            continue

        boxes = result.boxes

        for index in range(len(boxes)):
            class_id = int(boxes.cls[index].item())
            confidence = float(boxes.conf[index].item())

            coordinates = boxes.xyxy[index].tolist()

            x1, y1, x2, y2 = coordinates

            class_name = result.names[class_id]

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=BoundingBox(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                    ),
                )
            )

    return detections
