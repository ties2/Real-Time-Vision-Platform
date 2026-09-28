import cv2
import numpy as np
import pytest

from app.inference.preprocessing import decode_image


def test_decode_image():
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    success, encoded = cv2.imencode(".jpg", image)

    assert success

    decoded = decode_image(encoded.tobytes())

    assert decoded.shape == (100, 200, 3)


def test_invalid_image():
    with pytest.raises(ValueError):
        decode_image(b"not-an-image")
