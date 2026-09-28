from fastapi import Request

from app.inference.engine import InferenceEngine


def get_inference_engine(request: Request) -> InferenceEngine:
    """Return the application inference engine."""

    return InferenceEngine(
        registry=request.app.state.model_registry,
    )
