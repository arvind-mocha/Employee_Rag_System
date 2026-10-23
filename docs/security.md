# Security

This document describes the controls implemented by the application and the
additional controls required before public or multi-user deployment.

## Provider credential storage

Provider API keys are entered through the Settings page and sent only to the
backend. The backend:

1. Validates provider, model name, key, and expiry.
2. Generates a random 12-byte nonce.
3. Encrypts the key with AES-256-GCM.
4. Authenticates encryption with provider-specific associated data.
5. Stores ciphertext, nonce, provider, model, key hint, and timestamps in
   PostgreSQL.
6. Returns only masked metadata to the browser.

The API decrypts a key only when building a provider client for a request. Keys
are not cached in Redis, returned in API responses, or intentionally written to
logs.

Expired credentials are rejected before use. A background cleanup task checks
for expired records every 60 seconds, and users can delete a key immediately
from Settings.

## Encryption key

Local development creates a 32-byte installation key at:

```text
.secrets/credential_encryption_key
```

On POSIX systems, the application restricts the directory and file permissions.
The directory is ignored by Git.

Container deployments can provide a read-only key at:

```text
/run/secrets/credential_encryption_key
```

Operational requirements:

- Never commit the key.
- Back it up separately from the database.
- Restrict it to processes that must decrypt credentials.
- Restore the same key with a database backup.
- Rotate provider credentials if the key or database may be compromised.

Deleting or replacing the key makes existing ciphertext unreadable. Deleting a
credential from this app does not revoke it at the provider; revoke compromised
keys in the provider dashboard.

## ZIP upload controls

The upload service protects against common archive problems by:

- Rejecting non-ZIP content
- Rejecting absolute paths and `..` traversal
- Flattening archive folders to controlled destination directories
- Rejecting duplicate flattened filenames
- Rejecting unsupported file extensions
- Limiting compressed size, extracted size, and file count
- Testing archives for corrupt members before extraction
- Refusing to overwrite an existing filename
- Staging extraction before moving validated files into data directories

The system does not provide malware scanning or content moderation. Add an
antivirus or content-disarm stage when accepting files from untrusted users.

## Employee data handling

Uploaded documents and evaluation datasets can contain confidential or personal
information. They are stored locally and excluded from Git, but they are not
application-level encrypted by this repository.

Production deployments should add:

- Encrypted disks and encrypted database storage
- TLS for every network connection
- Access controls for source-data volumes
- Backup encryption and retention policies
- Data deletion and subject-request procedures
- Audit logs appropriate to the organization
- A documented data-classification and retention policy

The answer endpoint does not store conversation history, and chat messages are
not automatically added to the RAG corpus. Provider requests still send the
question and retrieved context to the chosen external provider, subject to that
provider's terms and data policies.

## Current trust boundary

The application currently has no user authentication, authorization, or tenant
isolation. Provider settings, documents, ingestion controls, and evaluation
operations belong to the installation rather than an individual user.

Therefore:

- Do not expose the current application directly to the public internet.
- Run it only in a trusted private environment.
- Add login, role-based access control, and per-tenant ownership before a
  multi-user deployment.
- Restrict Settings, Data, and Evaluation operations to administrators or
  developers.

The User/Developer switch is a UI feature, not a security boundary.

## Database and Redis

The included Compose file binds PostgreSQL and Redis to `127.0.0.1`, preventing
direct access from other network hosts under the standard local setup.
Production deployments should keep both services on private networks and use
unique managed credentials.

The database username and password in `app/api/config.py` are local development
defaults. They must not be reused for an internet-facing production database.

Redis contains job payloads, ingestion state, and evaluation results. Treat it
as sensitive infrastructure and protect it with network policy and, when
appropriate, authentication and transport encryption.

## Error and log handling

Safe API messages are returned for provider, database, Redis, ingestion, and
evaluation failures. Ingestion error handling removes the configured database
URL before storing an error message.

Application and infrastructure logs can still contain filenames, questions,
provider identifiers, exception details, or operational metadata. Limit access,
avoid shipping unnecessary content, and define log retention.

## Production hardening checklist

- [ ] Add authentication and role-based authorization.
- [ ] Add per-user or per-tenant document and credential ownership.
- [ ] Terminate HTTPS at a trusted reverse proxy or load balancer.
- [ ] Store production secrets in a managed secret service.
- [ ] Replace local database credentials and keep data services private.
- [ ] Add request size limits at the reverse proxy.
- [ ] Add rate limiting and abuse controls.
- [ ] Add malware scanning for untrusted uploads.
- [ ] Encrypt storage and backups.
- [ ] Add audit logging, metrics, alerts, and incident procedures.
- [ ] Review provider data-processing terms and employee privacy obligations.
- [ ] Back up and test restoration of PostgreSQL, source data, evaluation data,
      and the credential encryption key.

This guide is technical security information, not legal or compliance advice.
