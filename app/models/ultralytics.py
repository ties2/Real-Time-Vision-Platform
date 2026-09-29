from pathlib import Path
from typing import Any

from ultralytics import YOLO

from app.models.base import BaseModel
from app.models.registry import ModelConfig


class UltralyticsModel(BaseModel):
    """Ultralytics-backed object detection model."""

    def __init__(
            self,
            name: str,
            config: ModelConfig,
            models_dir: Path,
    ) -> None:
        self.name = name
        self.config = config
        self.models_dir = models_dir

        self._model: YOLO | None = None

        self.artifact_path = self.models_dir / self.config.artifact

    def load(self) -> None:
        """Load the Ultralytics model artifact."""

        if not self.artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self.artifact_path}")

        self._model = YOLO(str(self.artifact_path))

    def predict(
            self,
            input_data: Any,
    ) -> Any:
        """Run inference using the loaded model."""

        if self._model is None:
            raise RuntimeError(f"Model '{self.name}' has not been loaded.")

        target_device = None if self.config.device == "auto" else self.config.device
        return self._model.predict(
            source=input_data,
            device=target_device,
            conf=self.config.confidence_threshold,
            verbose=False,
        )

    def metadata(self) -> dict[str, Any]:
        """Return model metadata."""

        return {
            "name": self.name,
            "provider": self.config.provider,
            "task": self.config.task,
            "version": self.config.version,
            "artifact": self.config.artifact,
            "device": self.config.device,
            "confidence_threshold": (self.config.confidence_threshold),
            "loaded": self.is_loaded(),
        }

    def is_loaded(self) -> bool:
        """Return whether the model is loaded."""

        return self._model is not None

    def predict_batch(
            self,
            inputs: list[Any],
    ) -> list[Any]:
        """Run YOLO inference on a batch of images in ONE forward pass.

        Returns one Ultralytics ``Results`` object per input image,
        in the same order as ``inputs``.
        """

        if not inputs:
            return []

        if self._model is None:
            raise RuntimeError(f"Model '{self.name}' has not been loaded.")

        target_device = None if self.config.device == "auto" else self.config.device

        results = self._model.predict(
            source=inputs,
            device=target_device,
            conf=self.config.confidence_threshold,
            verbose=False,
        )

        return list(results)