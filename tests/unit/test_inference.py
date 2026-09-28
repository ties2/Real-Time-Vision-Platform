from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def test_inference():
    client = TestClient(app)

    image_path = Path("streetAndpeople.jpg")

    if not image_path.exists():
        return

    with image_path.open("rb") as image:
        response = client.post(
            "/api/v1/inference",
            files={
                "file": (
                    "streetAndpeople.jpg",
                    image,
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "yolo11"
    assert data["model_version"] == "1.0.0"
    assert data["image_width"] > 0
    assert data["image_height"] > 0
    assert "detections" in data
