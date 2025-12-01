"""Health and readiness checks."""

from fastapi import APIRouter

from app.config import get_settings
from app.schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/", response_model=HealthResponse, summary="Service health status")
async def health_check() -> HealthResponse:
    """Return a simple heartbeat payload."""
    settings = get_settings()
    return HealthResponse(status="ok", environment=settings.environment)

