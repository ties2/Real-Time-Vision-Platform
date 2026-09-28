import asyncio

import pytest

from app.inference.batching import DynamicBatcher


@pytest.mark.asyncio
async def test_single_request():
    batcher = DynamicBatcher(
        max_batch_size=4,
        max_wait_ms=10,
    )

    await batcher.start()

    try:
        result = await batcher.submit("image-1")

        assert result == "image-1"
    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_multiple_requests():
    batcher = DynamicBatcher(
        max_batch_size=4,
        max_wait_ms=10,
    )

    await batcher.start()

    try:
        results = await asyncio.gather(
            batcher.submit("image-1"),
            batcher.submit("image-2"),
            batcher.submit("image-3"),
            batcher.submit("image-4"),
        )

        assert results == [
            "image-1",
            "image-2",
            "image-3",
            "image-4",
        ]
    finally:
        await batcher.stop()
