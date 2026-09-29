import asyncio

import numpy as np
import pytest

from app.inference.engine import InferenceEngine


class FakeModel:
    def predict_batch(self, inputs):
        return [f"result-{index}" for index in range(len(inputs))]


class FakeRegistry:
    def get(self, model_name):
        assert model_name == "yolo11"
        return FakeModel()


@pytest.mark.asyncio
async def test_inference_engine_submit(monkeypatch):
    registry = FakeRegistry()

    engine = InferenceEngine(
        registry=registry,
        max_batch_size=4,
        max_wait_ms=20,
    )

    calls = []

    async def fake_predict_batch(
        model_name,
        inputs,
    ):
        calls.append((model_name, len(inputs)))

        return [f"result-{index}" for index in range(len(inputs))]

    monkeypatch.setattr(
        engine.batcher,
        "batch_handler",
        fake_predict_batch,
    )

    await engine.start()

    try:
        results = await asyncio.gather(
            engine.submit(
                "yolo11",
                np.zeros(
                    (10, 10, 3),
                    dtype=np.uint8,
                ),
            ),
            engine.submit(
                "yolo11",
                np.zeros(
                    (10, 10, 3),
                    dtype=np.uint8,
                ),
            ),
        )

        assert results == [
            "result-0",
            "result-1",
        ]

        assert calls == [
            ("yolo11", 2),
        ]

    finally:
        await engine.stop()
