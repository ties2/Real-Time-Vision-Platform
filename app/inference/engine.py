import asyncio
from typing import Any

import numpy as np

from app.inference.batching import DynamicBatcher
from app.models.registry import ModelRegistry


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
        """Submit inputs to the model-specific batcher."""

        results = await asyncio.gather(
            *[
                self.batcher.submit(
                    model_name,
                    input_data,
                )
                for input_data in inputs
            ]
        )

        return results

    async def _predict_batch(
        self,
        model_name: str,
        inputs: list[Any],
    ) -> list[Any]:
        """Execute a model-specific batch."""

        model = self.registry.get(model_name)

        return model.predict_batch(inputs)

    async def submit(
        self,
        model_name: str,
        image: Any,
    ) -> Any:
        """Submit one inference request to the dynamic batcher."""

        return await self.batcher.submit(
            model_name,
            image,
        )
