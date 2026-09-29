import asyncio
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from app.inference.batching import DynamicBatcher
from app.models.registry import ModelRegistry


@dataclass(frozen=True)
class InferenceOutput:
    """Result of one request after it went through a batch."""

    output: Any
    inference_time_ms: float
    batch_size: int


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
        """Run single-image inference (bypasses the batcher)."""

        model = self.registry.get(model_name)

        return model.predict(image)

    async def _predict_batch(
            self,
            model_name: str,
            inputs: list[Any],
    ) -> list[InferenceOutput]:
        """Execute one model-specific batch.

        The model call is blocking (CPU/GPU work), so it runs in a worker
        thread. Otherwise it would freeze the event loop and no new HTTP
        requests could be accepted while the model is running.
        """

        model = self.registry.get(model_name)

        started = perf_counter()

        outputs = await asyncio.to_thread(
            model.predict_batch,
            inputs,
        )

        elapsed_ms = (perf_counter() - started) * 1000

        return [
            InferenceOutput(
                output=output,
                inference_time_ms=elapsed_ms,
                batch_size=len(inputs),
            )
            for output in outputs
        ]

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