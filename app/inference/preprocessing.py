import cv2
import numpy as np


def decode_image(data: bytes) -> np.ndarray:
    """Decode raw image bytes into an OpenCV image."""

    buffer = np.frombuffer(data, dtype=np.uint8)

    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Unable to decode image.")

    return image
