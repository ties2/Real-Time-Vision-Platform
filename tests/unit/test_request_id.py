from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.request_id import RequestIDMiddleware


def create_test_app() -> FastAPI:
    app = FastAPI()

    app.add_middleware(RequestIDMiddleware)

    @app.get("/test")
    async def test_route():
        return {"status": "ok"}

    return app


def test_request_id_is_generated():
    client = TestClient(create_test_app())

    response = client.get("/test")

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]


def test_request_id_is_preserved():
    client = TestClient(create_test_app())

    response = client.get(
        "/test",
        headers={"X-Request-ID": "test-123"},
    )

    assert response.headers["X-Request-ID"] == "test-123"
