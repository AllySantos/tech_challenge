# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1 — builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=2.1.1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /build

RUN pip install "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock ./

RUN python -m venv /opt/venv

# AQUI ESTÁ A CORREÇÃO: Adicione o VIRTUAL_ENV para forçar o Poetry a usar este venv
ENV PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/venv"

RUN poetry install --only main --no-root

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

RUN groupadd --system app && useradd --system --gid app --create-home app

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY dvc.yaml pyproject.toml ./

RUN mkdir -p data models /mlflow && \
    chown -R app:app /app /opt/venv /mlflow

USER app

ENTRYPOINT []