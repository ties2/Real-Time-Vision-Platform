import time
from dataclasses import dataclass

import numpy as np

from app.models.registry import ModelRegistry


@dataclass
class InferenceResult:
    """Raw inference result with timing metadata."""

    results: object
    inference_time_ms: float
    image_width: int
    image_height: int


class InferenceEngine:
    """Orchestrates model inference."""

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def predict(
        self,
        model_name: str,
        image: np.ndarray,
    ) -> InferenceResult:
        """Run inference using a registered model."""

        model = self.registry.get(model_name)

        image_height, image_width = image.shape[:2]

        start_time = time.perf_counter()

        results = model.predict(image)

        elapsed = time.perf_counter() - start_time

        return InferenceResult(
            results=results,
            inference_time_ms=elapsed * 1000,
            image_width=image_width,
            image_height=image_height,
        )
