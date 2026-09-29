from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    """Abstract interface for all inference models."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights/artifacts."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Run inference on input data."""
        raise NotImplementedError

    def predict_batch(
        self,
        inputs: list[Any],
    ) -> list[Any]:
        """Run inference on a batch of inputs."""
        return [self.predict(input_data) for input_data in inputs]

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return model metadata."""
        raise NotImplementedError

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return whether the model is loaded."""
        raise NotImplementedError

    def inference_batch(
        self,
        inputs: list[Any],
    ) -> list[Any]:
        """Run inference on a batch of inputs (Fallback loop)."""
        return [self.predict(item) for item in inputs]
