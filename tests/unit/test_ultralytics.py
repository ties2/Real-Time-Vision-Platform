from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.models.registry import ModelConfig
from app.models.ultralytics import UltralyticsModel


def test_model_artifact_path():
    config = ModelConfig(
        provider="ultralytics",
        task="detect",
        version="1.0.0",
        artifact="yolo11n.pt",
    )

    model = UltralyticsModel(
        name="yolo11",
        config=config,
        models_dir=Path("/tmp/models"),
    )

    assert model.artifact_path == Path("/tmp/models/yolo11n.pt")


def test_missing_model_artifact():
    config = ModelConfig(
        provider="ultralytics",
        task="detect",
        version="1.0.0",
        artifact="missing.pt",
    )

    model = UltralyticsModel(
        name="yolo11",
        config=config,
        models_dir=Path("/tmp/models"),
    )

    with pytest.raises(FileNotFoundError):
        model.load()


def predict_batch(
    self,
    inputs: list[np.ndarray],
) -> list[Any]:
    """Run YOLO inference on a batch of images."""

    if not inputs:
        return []

    if not self.is_loaded():
        self.load()

    results = self._model.predict(
        source=inputs,
        verbose=False,
    )

    return list(results)
