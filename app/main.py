from fastapi import (
    Depends,
    FastAPI,
)
from sqlalchemy.orm import Session

from app.database import get_db
from app.routers.history import (
    router as history_router,
)
from app.routers.knowledge_base import (
    router as knowledge_base_router,
)
from app.routers.system_health import (
    router as system_health_router,
)
from app.schemas.evaluation_result import (
    HealthCheckResponse,
)
from app.schemas.health_check import (
    HealthCheckRequest,
)
from app.services.health_check_service import (
    execute_health_check,
)


app = FastAPI(
    title="AI Reliability Platform",
    version="0.8.0",
)


app.include_router(
    history_router
)

app.include_router(
    knowledge_base_router
)

app.include_router(
    system_health_router
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": (
            "AI Reliability Platform "
            "is running"
        )
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": (
            "ai-reliability-platform"
        ),
    }


@app.post(
    "/api/v1/health-checks",
    response_model=HealthCheckResponse,
)
def create_health_check(
    payload: HealthCheckRequest,
    db: Session = Depends(get_db),
) -> HealthCheckResponse:
    return execute_health_check(
        payload=payload,
        db=db,
    )
