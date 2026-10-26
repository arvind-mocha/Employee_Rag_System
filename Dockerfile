# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:0.12.13 AS uv
FROM python:3.12-slim-bookworm

COPY --from=uv /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# Install locked third-party dependencies in a cacheable layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Install the application as a non-editable package for runtime use.
COPY app ./app
RUN uv sync --frozen --no-dev --no-editable \
    && mkdir -p \
        /app/data \
        /app/structured_data \
        /app/temp_data \
        /app/evaluation_data \
        /app/.secrets \
        /root/.cache/huggingface

EXPOSE 8000

CMD ["employee-rag-system"]
