from fastapi import (
    APIRouter,
    Response,
    status,
)
from sqlalchemy import text

from app.database import engine
from knowledge_base.vector_store import (
    get_client as get_chroma_client,
)


router = APIRouter(
    tags=["System Health"],
)

SERVICE_NAME = (
    "ai-reliability-platform"
)


def _check_database() -> None:
    """Raise if PostgreSQL is not reachable."""
    with engine.connect() as connection:
        connection.execute(
            text("SELECT 1")
        )


def _check_chroma() -> None:
    """Raise if the Chroma client is not healthy."""
    get_chroma_client().heartbeat()


@router.get("/health/live")
def liveness() -> dict:
    """Report whether the API process is alive."""
    return {
        "status": "alive",
        "service": SERVICE_NAME,
    }


@router.get("/health/ready")
def readiness(
    response: Response,
) -> dict:
    """Report whether required dependencies are ready."""
    checks = {
        "database": "ready",
        "chroma": "ready",
    }

    try:
        _check_database()
    except Exception:
        checks["database"] = (
            "unavailable"
        )

    try:
        _check_chroma()
    except Exception:
        checks["chroma"] = (
            "unavailable"
        )

    is_ready = all(
        value == "ready"
        for value in checks.values()
    )

    if not is_ready:
        response.status_code = (
            status.HTTP_503_SERVICE_UNAVAILABLE
        )

    return {
        "status": (
            "ready"
            if is_ready
            else "not_ready"
        ),
        "service": SERVICE_NAME,
        "checks": checks,
    }
