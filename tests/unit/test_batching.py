import asyncio

import pytest

from app.inference.batching import DynamicBatcher


@pytest.mark.asyncio
async def test_single_request():
    async def handler(model_name, inputs):
        return inputs

    batcher = DynamicBatcher(
        batch_handler=handler,
        max_batch_size=4,
        max_wait_ms=10,
    )

    await batcher.start()

    try:
        result = await batcher.submit(
            "yolo11",
            "image-1",
        )

        assert result == "image-1"
    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_multiple_requests():
    async def handler(model_name, inputs):
        return inputs

    batcher = DynamicBatcher(
        batch_handler=handler,
        max_batch_size=4,
        max_wait_ms=10,
    )

    await batcher.start()

    try:
        results = await asyncio.gather(
            batcher.submit("yolo11", "image-1"),
            batcher.submit("yolo11", "image-2"),
            batcher.submit("yolo11", "image-3"),
            batcher.submit("yolo11", "image-4"),
        )

        assert results == [
            "image-1",
            "image-2",
            "image-3",
            "image-4",
        ]
    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_batch_handler_receives_multiple_inputs():
    received_batches: list[list[str]] = []

    async def handler(
        model_name,
        inputs: list[str],
    ) -> list[str]:
        received_batches.append(inputs)

        return [f"{item}-result" for item in inputs]

    batcher = DynamicBatcher(
        batch_handler=handler,
        max_batch_size=4,
        max_wait_ms=20,
    )

    await batcher.start()

    try:
        results = await asyncio.gather(
            batcher.submit("yolo11", "image-1"),
            batcher.submit("yolo11", "image-2"),
            batcher.submit("yolo11", "image-3"),
            batcher.submit("yolo11", "image-4"),
        )

        assert results == [
            "image-1-result",
            "image-2-result",
            "image-3-result",
            "image-4-result",
        ]

        assert received_batches == [
            [
                "image-1",
                "image-2",
                "image-3",
                "image-4",
            ]
        ]
    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_batch_handler_receives_one_batch():
    received_batches = []

    async def handler(model_name, inputs):
        received_batches.append(inputs)

        return [f"{item}-result" for item in inputs]

    batcher = DynamicBatcher(
        batch_handler=handler,
        max_batch_size=4,
        max_wait_ms=20,
    )

    await batcher.start()

    try:
        results = await asyncio.gather(
            batcher.submit("yolo11", "image-1"),
            batcher.submit("yolo11", "image-2"),
            batcher.submit("yolo11", "image-3"),
            batcher.submit("yolo11", "image-4"),
        )

        assert results == [
            "image-1-result",
            "image-2-result",
            "image-3-result",
            "image-4-result",
        ]

        assert received_batches == [
            [
                "image-1",
                "image-2",
                "image-3",
                "image-4",
            ]
        ]
    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_batches_are_separated_by_model():
    received_batches = []

    async def handler(model_name, inputs):
        received_batches.append((model_name, inputs))

        return [f"{model_name}:{item}" for item in inputs]

    batcher = DynamicBatcher(
        batch_handler=handler,
        max_batch_size=4,
        max_wait_ms=20,
    )

    await batcher.start()

    try:
        results = await asyncio.gather(
            batcher.submit("yolo11", "image-1"),
            batcher.submit("yolo11", "image-2"),
            batcher.submit("resnet", "image-3"),
            batcher.submit("resnet", "image-4"),
        )

        assert results == [
            "yolo11:image-1",
            "yolo11:image-2",
            "resnet:image-3",
            "resnet:image-4",
        ]

        assert received_batches == [
            (
                "yolo11",
                ["image-1", "image-2"],
            ),
            (
                "resnet",
                ["image-3", "image-4"],
            ),
        ]
    finally:
        await batcher.stop()
