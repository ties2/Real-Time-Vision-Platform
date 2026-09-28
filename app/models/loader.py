from pathlib import Path
from typing import Any

import yaml

from app.models.registry import ModelConfig, ModelRegistry
from app.models.ultralytics import UltralyticsModel


def load_model_registry(
    config_path: Path,
    models_dir: Path,
) -> ModelRegistry:
    """Load model configurations and initialize configured models."""

    if not config_path.exists():
        raise FileNotFoundError(f"Model configuration not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw_config: dict[str, Any] = yaml.safe_load(file) or {}

    registry = ModelRegistry(models_dir=models_dir)

    models_config = raw_config.get("models", {})

    for name, raw_model_config in models_config.items():
        config = ModelConfig(**raw_model_config)

        registry.register_config(name, config)

        if config.provider == "ultralytics":
            model = UltralyticsModel(
                name=name,
                config=config,
                models_dir=models_dir,
            )

            model.load()
            registry.register(name, model)

        else:
            raise ValueError(f"Unsupported model provider: {config.provider}")

    return registry
