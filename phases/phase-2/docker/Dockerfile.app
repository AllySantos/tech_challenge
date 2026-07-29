# syntax=docker/dockerfile:1

# Stage 1 — builder: resolve as dependências com Poetry e gera um venv  isolado.
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=2.1.1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true

WORKDIR /build

RUN pip install "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root \
    && mv /build/.venv /opt/venv

# Stage 2 — runtime: imagem final enxuta, sem Poetry/compiladores, rodando  como usuário não-root.

FROM python:3.11-slim AS runtime

RUN groupadd --system app && useradd --system --gid app --create-home app

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY dvc.yaml params.yaml pyproject.toml ./

# data/, models/ e mlruns/ são montados via volume no docker-compose —
# não fazem parte da imagem (mantém a imagem pequena e independente do
# dataset escolhido).
RUN mkdir -p data models && chown -R app:app /app

USER app

# Sem CMD fixo: o docker-compose.yml define o comando de cada serviço
# (treino roda `dvc repro`, MLflow roda `mlflow server`).
ENTRYPOINT []
