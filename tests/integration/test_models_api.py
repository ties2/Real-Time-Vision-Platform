from fastapi.testclient import TestClient

from app.main import app


def test_list_models():
    client = TestClient(app)

    response = client.get("/api/v1/models")

    assert response.status_code == 200

    data = response.json()

    assert "models" in data
    assert len(data["models"]) >= 1

    model = data["models"][0]

    assert "name" in model
    assert "provider" in model
    assert "version" in model
    assert "loaded" in model
