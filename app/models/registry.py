from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.models.base import BaseModel as ModelInterface


class ModelConfig(BaseModel):
    """Configuration describing a model artifact."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    task: str
    version: str
    artifact: str
    device: str = "auto"
    confidence_threshold: float = 0.5


class ModelRegistry:
    """Registry responsible for model configuration and instances."""

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self._configs: dict[str, ModelConfig] = {}
        self._models: dict[str, ModelInterface] = {}

    def register_config(
        self,
        name: str,
        config: ModelConfig,
    ) -> None:
        """Register model configuration."""

        if name in self._configs:
            raise ValueError(f"Model configuration '{name}' is already registered.")

        self._configs[name] = config

    def register(
        self,
        name: str,
        model: ModelInterface,
    ) -> None:
        """Register an initialized model instance."""

        if name in self._models:
            raise ValueError(f"Model '{name}' is already registered.")

        self._models[name] = model

    def get(self, name: str) -> ModelInterface:
        """Return a registered model instance."""

        try:
            return self._models[name]
        except KeyError as exc:
            raise KeyError(f"Model '{name}' is not loaded.") from exc

    def get_config(self, name: str) -> ModelConfig:
        """Return model configuration."""

        try:
            return self._configs[name]
        except KeyError as exc:
            raise KeyError(f"Model configuration '{name}' is not registered.") from exc

    def exists(self, name: str) -> bool:
        """Check whether a model is loaded."""

        return name in self._models

    def list_models(self) -> list[str]:
        """Return loaded model names."""

        return list(self._models.keys())

    def list_configured_models(self) -> list[str]:
        """Return configured model names."""

        return list(self._configs.keys())

    def metadata(self) -> dict[str, dict[str, Any]]:
        """Return metadata for loaded models."""

        return {name: model.metadata() for name, model in self._models.items()}
    def list_metadata(self) -> list[dict[str, Any]]:
        """Return metadata for all loaded models."""

        return [
            model.metadata()
            for model in self._models.values()
        ]
