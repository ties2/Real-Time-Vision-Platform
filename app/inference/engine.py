from dataclasses import dataclass
from typing import Any

import numpy as np

from app.inference.batching import DynamicBatcher
from app.models.registry import ModelRegistry


@dataclass
class InferenceResult:
    """Raw inference result with timing metadata."""

    results: object
    inference_time_ms: float
    image_width: int
    image_height: int


class InferenceEngine:
    """Orchestrates model inference and dynamic batching."""

    def __init__(
        self,
        registry: ModelRegistry,
        max_batch_size: int = 4,
        max_wait_ms: int = 10,
    ) -> None:
        self.registry = registry

        self.batcher = DynamicBatcher(
            batch_handler=self._predict_batch,
            max_batch_size=max_batch_size,
            max_wait_ms=max_wait_ms,
        )

    async def start(self) -> None:
        """Start the inference batcher."""

        await self.batcher.start()

    async def stop(self) -> None:
        """Stop the inference batcher."""

        await self.batcher.stop()

    def predict(
        self,
        model_name: str,
        image: np.ndarray,
    ) -> Any:
        """Run single-image inference."""

        model = self.registry.get(model_name)

        return model.predict(image)

    async def predict_batch(
        self,
        model_name: str,
        inputs: list[Any],
    ) -> list[Any]:
        """Submit inputs for batched model inference."""

        return await self.batcher.submit(
            model_name=model_name,
            input_data=inputs,
        )

    async def _predict_batch(
        self,
        inputs: list[Any],
    ) -> list[Any]:
        """Execute one collected batch."""

        if not inputs:
            return []

        raise NotImplementedError("Model-aware batch execution is the next step.")
