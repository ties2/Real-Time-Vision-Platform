import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

BatchHandler = Callable[
    [list[Any]],
    Awaitable[list[Any]],
]


@dataclass
class InferenceRequest:
    """Single inference request waiting for execution."""

    model_name: str
    input_data: Any
    future: asyncio.Future[Any]


class DynamicBatcher:
    """Collect inference requests into bounded batches."""

    def __init__(
        self,
        batch_handler: BatchHandler,
        max_batch_size: int = 4,
        max_wait_ms: int = 10,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be greater than zero.")

        if max_wait_ms < 0:
            raise ValueError("max_wait_ms cannot be negative.")

        self.batch_handler = batch_handler
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

        self._queue: asyncio.Queue[InferenceRequest | None] = asyncio.Queue()

        self._worker_task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        """Start the background batch worker."""

        if self._running:
            return

        self._running = True
        self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        """Stop the worker gracefully."""

        if not self._running:
            return

        self._running = False
        await self._queue.put(None)

        if self._worker_task is not None:
            await self._worker_task
            self._worker_task = None

    async def submit(
        self,
        model_name: str,
        input_data: Any,
    ) -> Any:
        """Submit one model inference request."""

        if not self._running:
            raise RuntimeError("DynamicBatcher is not running.")

        loop = asyncio.get_running_loop()

        future: asyncio.Future[Any] = loop.create_future()

        request = InferenceRequest(
            model_name=model_name,
            input_data=input_data,
            future=future,
        )

        await self._queue.put(request)

        return await future

    async def _worker(self) -> None:
        """Collect and process requests."""

        while True:
            request = await self._queue.get()

            if request is None:
                break

            batch = [request]

            deadline = perf_counter() + self.max_wait_ms / 1000

            while len(batch) < self.max_batch_size:
                remaining = deadline - perf_counter()

                if remaining <= 0:
                    break

                try:
                    next_request = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=remaining,
                    )
                except TimeoutError:
                    break

                if next_request is None:
                    break

                batch.append(next_request)

            await self._process_batch(batch)

    async def _process_batch(
        self,
        batch: list[InferenceRequest],
    ) -> None:
        """Process one batch."""

        inputs = [request.input_data for request in batch]

        try:
            results = await self.batch_handler(inputs)

            if len(results) != len(batch):
                raise RuntimeError("Batch handler returned an invalid number of results.")

        except Exception as exc:
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(exc)

            return

        for request, result in zip(
            batch,
            results,
            strict=True,
        ):
            if not request.future.done():
                request.future.set_result(result)
