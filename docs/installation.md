# Installation and first run

This guide describes a local development installation. The application uses
central Python defaults, so customers do not need to create or edit an `.env`
file.

## Prerequisites

- Docker Desktop or Docker Engine with Docker Compose
- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- Node.js 20 or newer with npm
- Internet access during dependency installation and the first embedding-model
  download
- An API key for at least one supported LLM provider

Supported providers are Hugging Face, OpenAI, Anthropic, and xAI Grok.

## Default local services

| Service | Address | Purpose |
| --- | --- | --- |
| Frontend | `http://localhost:5173` | React development application |
| Backend | `http://localhost:8000` | FastAPI application |
| API documentation | `http://localhost:8000/docs` | Interactive OpenAPI interface |
| PostgreSQL/PGVector | `localhost:6024` | Vector and credential storage |
| Redis | `localhost:6379` | Ingestion state, locks, and evaluation jobs |

## Install dependencies

From the repository root:

```bash
uv sync
```

Install frontend dependencies:

```bash
cd frontend
npm install
cd ..
```

The lock files should be committed and used as-is for reproducible installs.

## Start the infrastructure

```bash
docker compose up -d
docker compose ps
```

Wait until both `employee-rag-pgvector` and `employee-rag-redis` report healthy.
Their data is stored in named Docker volumes, so a normal stop and restart does
not erase it.

## Start the API

Open a terminal in the repository root:

```bash
uv run uvicorn app.api.main:app --reload
```

The ingestion pipeline uses absolute paths derived from the project root, but
running commands from the repository root remains the clearest convention.

## Start the evaluation worker

Open another terminal in the repository root:

```bash
uv run employee-rag-worker
```

This RQ worker is required for the asynchronous evaluation operations used by
the frontend. It is not used by normal answer generation. ZIP ingestion runs as
a guarded background task in the API process.

## Start the frontend

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
ports or Docker service names accordingly. Provider API keys and model names are
not stored in configuration source code.

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

## Stop local services

```bash
docker compose down
```

This preserves the named volumes. `docker compose down -v` also deletes the
PostgreSQL and Redis volumes and should be used only when permanent local data
deletion is intended.
