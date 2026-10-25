# Installation and first run

This guide describes the containerized installation and the optional local
development workflow. The application uses central Python defaults, so
customers do not need to create or edit an `.env` file.

## Prerequisites

- Docker Desktop or Docker Engine with Docker Compose
- Internet access during dependency installation and the first embedding-model
  download
- An API key for at least one supported LLM provider

Python 3.12, [uv](https://docs.astral.sh/uv/), and Node.js 20 or newer are only
required for the optional non-containerized development workflow.

Supported providers are Google Gemini, OpenAI, Anthropic, and xAI Grok.

## Default local services

| Service | Address | Purpose |
| --- | --- | --- |
| Frontend | `http://localhost:5173` | Nginx-served React application |
| Backend | `http://localhost:8000` | FastAPI application |
| API documentation | `http://localhost:8000/docs` | Interactive OpenAPI interface |
| PostgreSQL/PGVector | `localhost:6024` | Vector and credential storage |
| Redis | `localhost:6379` | Ingestion state, locks, and evaluation jobs |

## Start the complete application

From the repository root:

```bash
docker compose up --build -d
docker compose ps
```

This builds and starts all five services:

- `frontend`: production React build served by Nginx
- `api`: FastAPI RAG and ingestion service
- `worker`: RQ evaluation worker
- `pgvector`: PostgreSQL with PGVector
- `redis`: queues, locks, status, and results

Wait until the services with health checks report healthy. Open
`http://localhost:5173`. PostgreSQL, Redis, the embedding-model cache, uploaded
files, processed files, evaluation datasets, and credential-encryption key all
survive normal container restarts.

Follow application logs with:

```bash
docker compose logs -f api worker frontend
```

## Optional local development

Install backend and frontend dependencies:

```bash
uv sync
npm --prefix frontend install
docker compose up -d pgvector redis
```

### Start the API

Open a terminal in the repository root:

```bash
uv run uvicorn app.api.main:app --reload
```

The ingestion pipeline uses absolute paths derived from the project root, but
running commands from the repository root remains the clearest convention.

### Start the evaluation worker

Open another terminal in the repository root:

```bash
uv run employee-rag-worker
```

This RQ worker is required for the asynchronous evaluation operations used by
the frontend. It is not used by normal answer generation. ZIP ingestion runs as
a guarded background task in the API process.

### Start the frontend

Open another terminal:

```bash
cd frontend
npm run dev
```

Vite serves the application at `http://localhost:5173` and proxies `/api`
requests to `http://localhost:8000`.

## First-run workflow

### 1. Save a provider credential

1. Open **Settings**.
2. Select a provider.
3. Enter the exact provider model ID.
4. Enter the API key.
5. Select an expiry period.
6. Save the encrypted credential.

Only masked metadata is returned to the browser after saving.

### 2. Upload data

1. Switch to **Developer** mode.
2. Open **Data**.
3. Select a ZIP containing supported files.
4. Select **Upload ZIP**.

The server validates the archive before storing it. Unsafe paths and archives
without any supported files are rejected. Unsupported members in a mixed
archive are ignored, while duplicate or existing filenames receive a numeric
suffix.

### 3. Build the knowledge index

1. Select **Start ingestion**.
2. Wait for `pending` to change to `running`.
3. Wait for **Knowledge base ready**.
4. Open **Ask**.

The first ingestion may take longer because Sentence Transformers downloads the
embedding model into the local cache. Chat remains unavailable until ingestion
finishes successfully.

## Central configuration

Runtime defaults are defined in `app/api/config.py`, including:

- PostgreSQL and Redis addresses
- PGVector table and schema names
- Embedding model, device, and batch size
- Data and evaluation paths
- Queue and result timeouts
- Upload limits
- Built-in model pricing data
- Credential encryption-key location

The application detects whether it is running in Docker and selects local host
ports or Docker service names accordingly. Nginx proxies browser `/api`
requests to the API container. Provider API keys and model names are not stored
in configuration source code.

## Validate the installation

Backend tests:

```bash
uv run pytest -q
```

Frontend lint and production build:

```bash
cd frontend
npm run lint
npm run build
```

## Stop the application

```bash
docker compose down
```

This preserves the named volumes. `docker compose down -v` also deletes the
PostgreSQL, Redis, and downloaded embedding-model cache volumes. Bind-mounted
source files, evaluation datasets, and the encryption key remain on the host.
