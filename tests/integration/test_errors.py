from fastapi.testclient import TestClient

from app.main import app


def test_unknown_model_returns_structured_error():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/inference?model=does-not-exist",
            files={
                "file": (
                    "test.jpg",
                    b"invalid-image",
                    "image/jpeg",
                )
            },
        )

    assert response.status_code in {400, 404}

    data = response.json()

    assert "code" in data
    assert "message" in data
    assert "request_id" in data


def test_invalid_image_returns_400():
    client = TestClient(app)

    response = client.post(
        "/api/v1/inference",
        files={
            "file": (
                "test.jpg",
                b"not-an-image",
                "image/jpeg",
            )
        },
        headers={
            "X-Request-ID": "test-invalid-image",
        },
    )

    assert response.status_code == 400

    data = response.json()

    assert data["code"] == "INVALID_INPUT"
    assert data["request_id"] == "test-invalid-image"
