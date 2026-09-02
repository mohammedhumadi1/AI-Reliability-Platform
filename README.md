# AI Reliability Platform

AI Reliability Platform evaluates RAG applications, diagnoses likely failure causes, tracks reliability over time, and independently verifies generated answers against company knowledge-base documents.

## Current capabilities

- FastAPI health-check API
- RAG evaluation metrics: correctness, faithfulness, context precision, context recall, answer relevancy, and hallucination risk
- Numeric contradiction detection
- Deterministic root-cause diagnosis
- Overall reliability health score
- Recommendation engine
- PostgreSQL persistence and Health Check History API
- Company Knowledge Base with PDF extraction, chunking, multilingual embeddings, and persistent Chroma storage
- Independent answer verification using Question + RAG Answer + RAG Context against company evidence
- Streamlit dashboard with English and Arabic interfaces

## Root-cause logic with company evidence

When indexed company documents are available, independent evidence is used before proxy-only rules:

- Company evidence is missing -> `KNOWLEDGE_BASE_FAILURE`
- Company evidence exists but RAG context missed it -> `RETRIEVAL_FAILURE`
- RAG context contains the evidence but the generated answer conflicts with it -> `GENERATION_FAILURE`

If no company Knowledge Base is available, the existing evaluation-based rules continue to work normally.

## Local setup

Requirements:

- Python 3.12+
- PostgreSQL
- `uv`

Install dependencies:

```powershell
uv sync
```

Create `.env` from `.env.example` and set your PostgreSQL password. Never commit the real `.env` file.

Apply database migrations:

```powershell
uv run alembic upgrade head
```

Start FastAPI:

```powershell
uv run python -m uvicorn app.main:app --host 127.0.0.1 --port 8002
```

Swagger UI is available at `http://127.0.0.1:8002/docs`.

Start Streamlit in another terminal:

```powershell
uv run streamlit run dashboard/app.py --server.port 8501
```

Dashboard: `http://localhost:8501`

## Docker setup

Requirements: Docker and Docker Compose.

Copy the example env file and set a real Postgres password:

```bash
cp .env.docker.example .env
# edit .env and replace CHANGE_ME
```

Build and start everything (Postgres, FastAPI, Streamlit):

```bash
docker compose build
docker compose up -d
docker compose ps
```

- The `api` container runs `alembic upgrade head` automatically before starting uvicorn, so the schema is always current.
- FastAPI: `http://localhost:8002` (Swagger UI at `/docs`)
- Streamlit dashboard: `http://localhost:8501`
- Postgres data persists in the `postgres_data` volume, Chroma vectors persist in `chroma_data` — both survive `docker compose down` (use `docker compose down -v` to wipe them).
- The dashboard container talks to the API via the `API_URL` environment variable (`http://api:8002` inside the Docker network); FastAPI talks to Postgres via `DATABASE_URL`, built from the Postgres credentials in your `.env`.

Stop everything:

```bash
docker compose down
```

View logs:

```bash
docker compose logs -f api
docker compose logs -f dashboard
```

## Main API endpoints

- `GET /health`
- `POST /api/v1/health-checks`
- `GET /api/v1/health-checks`
- `GET /api/v1/health-checks/{health_check_id}`
- `POST /api/v1/knowledge-base/upload`
- `POST /api/v1/knowledge-base/verify`

The Knowledge Base verification endpoint requires the generated RAG answer. An optional retrieved RAG context can also be supplied to help distinguish retrieval failures from generation failures.

## Storage

- PostgreSQL: health checks, evaluation metrics, diagnoses, recommendations, indexed-document metadata, and Knowledge Base verification results
- Chroma: document chunk vectors under `./chroma_db` by default

Both `.env` and `chroma_db/` are excluded from Git.

## Tests

```powershell
uv run pytest -q
uv run alembic check
```

## Notes

The semantic scores are reliability proxies rather than logical-entailment probabilities. Numeric duration contradictions are checked explicitly. Thresholds should be calibrated on representative production data before using them as enforcement criteria.
