#!/bin/sh
# Entrypoint for the "api" service.
# Runs pending Alembic migrations against the Postgres container
# before starting uvicorn, so the schema is always up to date.
set -eu

echo "[api-entrypoint] running: alembic upgrade head"
alembic upgrade head

echo "[api-entrypoint] starting FastAPI"
exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8002
