# Troubleshooting

Start by checking infrastructure:

```bash
docker compose ps
```

PostgreSQL/PGVector and Redis should both report healthy. Confirm the API and,
for evaluation, the RQ worker are running from the repository root.

## The frontend cannot reach the API

Symptoms include a request error, unavailable ingestion status, or provider
settings failing to load.

1. Open `http://localhost:8000/docs`.
2. Confirm the API terminal has not exited.
3. Confirm Vite is running on port 5173.
4. Confirm `frontend/vite.config.ts` proxies `/api` to port 8000.
5. Check whether another process is already using either port.

Restart the API:

```bash
uv run uvicorn app.api.main:app --reload
```

## PostgreSQL is unavailable

```bash
docker compose up -d pgvector
docker compose ps
docker compose logs pgvector
```

The local application expects PostgreSQL on `localhost:6024`. Do not delete the
volume while diagnosing a connection problem.

## Redis is unavailable

Redis is required for ingestion status and locking, plus evaluation jobs.

```bash
docker compose up -d redis
docker compose logs redis
```

The local application expects Redis on `localhost:6379`.

## Chat shows the Data page

This is the expected readiness gate when ingestion is not `completed`.

1. Upload a supported ZIP.
2. Start ingestion.
3. Wait for **Knowledge base ready**.

A later upload deliberately resets readiness and requires another ingestion run.

## Ingestion reports `Data directory not found`

Current ingestion code resolves directories from `app/api/config.py` instead of
the shell's working directory. Restart the API so it loads the latest code,
confirm the ZIP upload succeeded, and retry ingestion.

If no upload has succeeded, upload the ZIP first. The upload service creates the
required source directories.

## Ingestion remains pending or running

- Check the API logs for extraction, embedding, PostgreSQL, or model-download
  errors.
- Confirm Redis is healthy because it stores the status and distributed lock.
- Allow extra time on the first run while the embedding model downloads.
- Confirm the API process was not restarted during its background task.

The ingestion lock has a safety timeout. Do not manually delete Redis keys while
a real ingestion process may still be running.

## Ingestion failed after a ZIP upload

The Data page displays a safe error. Common causes are:

- PostgreSQL is unavailable.
- The local embedding model cannot download or load.
- A document is corrupt or cannot be parsed.
- The process lacks permission to read or write data directories.
- The runtime has insufficient memory or disk space.

Correct the cause and select **Start ingestion** again. A successfully uploaded
ZIP does not normally need to be uploaded again.

## The ZIP is rejected

| Error | Meaning |
| --- | --- |
| Not a valid ZIP | Content is not a readable ZIP archive |
| No supported files | Every usable member has an unsupported extension |
| Unsafe path | An archive member uses an absolute path or path traversal |
| Too large | Compressed size, extracted size, or file count exceeds a limit |

Unsupported members in a mixed archive are ignored. Duplicate flattened names
and conflicts with stored files receive numeric suffixes, so uploads never
silently overwrite existing source files.

## No model provider is available

Open **Settings** and save an API key, exact model ID, and future expiry for at
least one provider. If a saved provider disappeared, its credential may have
expired and been deleted automatically.

## A provider request fails

Check:

- The credential has not expired.
- The exact model ID is valid for the account.
- The provider account has access and available quota.
- The key has permission to use that model.
- The machine can reach the provider.
- Provider status and rate limits.

The backend returns a safe provider error but does not return the stored key.

## Existing credentials cannot be decrypted

The database was opened with a different installation encryption key. Restore
the original `.secrets/credential_encryption_key` or mounted Docker secret. If
the original key is unavailable, delete and replace the provider credentials.

## Retrieval returns no relevant context

- Confirm ingestion completed after the latest upload.
- Ask a question covered by the uploaded content.
- Enable semantic and keyword search together.
- Inspect retrieved chunks in Developer mode.
- Re-ingest if the PGVector index is missing or out of date.

At least one retrieval strategy must be enabled.

## Evaluation stays queued

Start the worker in a separate terminal:

```bash
uv run employee-rag-worker
```

Also confirm Redis is healthy and that the worker terminal shows the evaluation
queue.

## Evaluation fails or reports stale records

- Verify answer and judge provider credentials.
- Review worker logs for provider or retrieval failures.
- Confirm the dataset still exists in `evaluation_data/`.
- Regenerate and review ground truth after changing the indexed corpus.
- Remember that asynchronous job results expire after the configured result TTL.

Stale records are not counted as passes or failures.

## Frontend dependencies or build fail

From `frontend/`:

```bash
npm install
npm run lint
npm run build
```

Use Node.js 20 or newer. Preserve `package-lock.json` rather than generating an
unrelated lock.

## Resetting local infrastructure

A normal restart preserves data:

```bash
docker compose down
docker compose up -d
```

`docker compose down -v` deletes PostgreSQL and Redis volumes. It is destructive
and should be used only when intentionally resetting local state. It does not
remove local source-data directories or the encryption key.

## Collecting a useful error report

Include:

- The operation and page where the failure occurred
- The HTTP status and safe response detail
- Relevant API or worker traceback with secrets removed
- `docker compose ps` output
- Application version or commit
- File type and approximate size, without confidential document content

Never include API keys, the credential encryption key, database passwords, or
confidential employee document contents.
