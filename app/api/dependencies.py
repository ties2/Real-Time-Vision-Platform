from fastapi import Request

from app.inference.engine import InferenceEngine


def get_inference_engine(
    request: Request,
) -> InferenceEngine:
    return request.app.state.inference_engine
