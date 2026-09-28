import time
from dataclasses import dataclass

import numpy as np
from time import perf_counter
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
        model = self.registry.get(model_name)

        started_at = perf_counter()

        result = model.inference(image)

        elapsed_ms = (perf_counter() - started_at) * 1000

        return InferenceResult(
            results=result,
            inference_time_ms=elapsed_ms,
            image_width=image.shape[1],
            image_height=image.shape[0],
        )
