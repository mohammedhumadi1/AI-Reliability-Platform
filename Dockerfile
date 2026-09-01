# syntax=docker/dockerfile:1

# Single image used by both the "api" and "dashboard" services in
# compose.yaml. Which process runs is decided by the command each
# service overrides (see compose.yaml), not by this Dockerfile.

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# curl -> used by the container healthchecks in compose.yaml
# build-essential/libpq-dev -> safety net for any dependency that
# needs to compile from source on an architecture without a prebuilt wheel
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.6 /uv /uvx /usr/local/bin/

WORKDIR /app

# Install dependencies first so this layer is cached as long as
# pyproject.toml / uv.lock don't change (source code changes won't
# bust it and trigger a full reinstall).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Now bring in the actual source code.
COPY . .

ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8002 8501

# Overridden per-service in compose.yaml (api runs uvicorn behind an
# alembic-upgrade entrypoint, dashboard runs streamlit directly).
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
