from fastapi import APIRouter

from app.core.config import get_settings

router = APIRouter(tags=["health"])

settings = get_settings()


@router.get("/health")
async def health() -> dict[str, str]:
    """Basic application health check."""

    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
    }


@router.get("/ready")
async def readiness() -> dict[str, str]:
    """Application readiness check."""

    return {
        "status": "ready",
    }
