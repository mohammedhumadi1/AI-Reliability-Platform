from fastapi import Response

from app.main import app
from app.routers import system_health


class FakeConnection:
    def __init__(self):
        self.statements = []

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        return False

    def execute(self, statement):
        self.statements.append(
            str(statement)
        )


class FakeEngine:
    def __init__(
        self,
        fail=False,
    ):
        self.fail = fail
        self.connection = (
            FakeConnection()
        )

    def connect(self):
        if self.fail:
            raise RuntimeError(
                "database unavailable"
            )

        return self.connection


class FakeChromaClient:
    def __init__(
        self,
        fail=False,
    ):
        self.fail = fail
        self.heartbeat_calls = 0

    def heartbeat(self):
        self.heartbeat_calls += 1

        if self.fail:
            raise RuntimeError(
                "chroma unavailable"
            )

        return 1


def test_health_routes_are_registered():
    route_paths = set(
        app.openapi()[
            "paths"
        ].keys()
    )

    assert "/health/live" in route_paths
    assert "/health/ready" in route_paths


def test_liveness_reports_alive():
    response = (
        system_health.liveness()
    )

    assert response == {
        "status": "alive",
        "service": (
            "ai-reliability-platform"
        ),
    }


def test_readiness_reports_ready(
    monkeypatch,
):
    fake_engine = FakeEngine()
    fake_chroma = FakeChromaClient()

    monkeypatch.setattr(
        system_health,
        "engine",
        fake_engine,
    )

    monkeypatch.setattr(
        system_health,
        "get_chroma_client",
        lambda: fake_chroma,
    )

    http_response = Response()

    result = system_health.readiness(
        response=http_response,
    )

    assert http_response.status_code == 200
    assert result["status"] == "ready"
    assert result["checks"] == {
        "database": "ready",
        "chroma": "ready",
    }

    assert (
        fake_engine
        .connection
        .statements
        == ["SELECT 1"]
    )

    assert (
        fake_chroma.heartbeat_calls
        == 1
    )


def test_readiness_returns_503_when_database_fails(
    monkeypatch,
):
    fake_chroma = FakeChromaClient()

    monkeypatch.setattr(
        system_health,
        "engine",
        FakeEngine(
            fail=True
        ),
    )

    monkeypatch.setattr(
        system_health,
        "get_chroma_client",
        lambda: fake_chroma,
    )

    http_response = Response()

    result = system_health.readiness(
        response=http_response,
    )

    assert http_response.status_code == 503
    assert result["status"] == "not_ready"
    assert result["checks"] == {
        "database": "unavailable",
        "chroma": "ready",
    }


def test_readiness_returns_503_when_chroma_fails(
    monkeypatch,
):
    monkeypatch.setattr(
        system_health,
        "engine",
        FakeEngine(),
    )

    monkeypatch.setattr(
        system_health,
        "get_chroma_client",
        lambda: FakeChromaClient(
            fail=True
        ),
    )

    http_response = Response()

    result = system_health.readiness(
        response=http_response,
    )

    assert http_response.status_code == 503
    assert result["status"] == "not_ready"
    assert result["checks"] == {
        "database": "ready",
        "chroma": "unavailable",
    }


def test_readiness_returns_503_when_all_dependencies_fail(
    monkeypatch,
):
    monkeypatch.setattr(
        system_health,
        "engine",
        FakeEngine(
            fail=True
        ),
    )

    monkeypatch.setattr(
        system_health,
        "get_chroma_client",
        lambda: FakeChromaClient(
            fail=True
        ),
    )

    http_response = Response()

    result = system_health.readiness(
        response=http_response,
    )

    assert http_response.status_code == 503
    assert result["status"] == "not_ready"
    assert result["checks"] == {
        "database": "unavailable",
        "chroma": "unavailable",
    }
