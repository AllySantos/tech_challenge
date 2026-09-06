"""API REST de triagem de urgência de laudos médicos."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.app import metrics
from src.app.schemas import (
    BatchTriageRequest,
    BatchTriageResponse,
    HealthResponse,
    TriageRequest,
    TriageResponse,
)
from src.configs.settings import settings
from src.inference.predictor import Predictor, load_predictor

logger = logging.getLogger(__name__)

predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Carrega o modelo uma única vez, na subida do processo.

    Uma falha no carregamento não derruba o serviço: ``/health`` e ``/metrics``
    continuam respondendo para que o orquestrador e o Prometheus enxerguem o
    estado degradado, enquanto ``/predict`` devolve 503.
    """
    global predictor

    try:
        predictor = load_predictor(settings.inference_backend)
        metrics.MODEL_LOADED.set(1)
        metrics.MODEL_INFO.labels(version=predictor.model_version, backend=predictor.backend).set(1)
        logger.info(
            "API pronta | versão=%s backend=%s classes=%s",
            predictor.model_version,
            predictor.backend,
            predictor.labels,
        )
    except Exception:  # noqa: BLE001 - qualquer falha de carga vira modo degradado
        # Deliberadamente amplo: artefato ausente, backend inválido, grafo
        # corrompido ou dependência de runtime faltando devem manter /health e
        # /metrics no ar para o operador diagnosticar, em vez de derrubar o
        # processo em um laço de restart.
        metrics.MODEL_LOADED.set(0)
        logger.exception("Falha ao carregar o modelo; a API sobe em estado degradado")

    yield


app = FastAPI(
    title=settings.api_title,
    description=(
        "Classifica laudos médicos em três níveis de urgência (normal, atenção, "
        "urgente) para priorização da fila de triagem hospitalar."
    ),
    version=settings.api_version,
    lifespan=lifespan,
)


@app.middleware("http")
async def track_requests(request: Request, call_next: Callable) -> Response:
    """Contabiliza volume e latência de cada requisição."""
    # A rota é resolvida via template (`/predict`, não `/predict/123`) para não
    # explodir a cardinalidade das séries no Prometheus.
    endpoint = request.scope.get("route").path if request.scope.get("route") else request.url.path

    started = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - started

    if endpoint != "/metrics":
        metrics.REQUEST_DURATION.labels(method=request.method, endpoint=endpoint).observe(elapsed)
        metrics.REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=str(response.status_code),
        ).inc()

    return response


def _require_predictor() -> Predictor:
    if predictor is None:
        metrics.ERRORS_TOTAL.labels(type="model_unavailable").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo não carregado. Execute o pipeline de treino e reinicie a API.",
        )
    return predictor


def _classify(texts: list[str]) -> list[TriageResponse]:
    """Executa a inferência e emite as métricas do modelo."""
    active = _require_predictor()

    started = time.perf_counter()
    predictions = active.predict(texts)
    elapsed = time.perf_counter() - started

    metrics.INFERENCE_DURATION.labels(backend=active.backend).observe(elapsed)

    # O tempo total é rateado entre os itens do lote para que a métrica por
    # laudo continue comparável entre chamadas unitárias e em lote.
    per_item_ms = round(elapsed * 1000 / len(predictions), 3)

    responses = []
    for prediction in predictions:
        metrics.record_prediction(prediction.urgency, prediction.confidence)
        responses.append(
            TriageResponse(
                urgency=prediction.urgency,
                confidence=prediction.confidence,
                probabilities=prediction.probabilities,
                model_version=active.model_version,
                backend=active.backend,
                inference_ms=per_item_ms,
            )
        )
    return responses


@app.get("/health", response_model=HealthResponse, tags=["operacional"])
def health() -> HealthResponse:
    """Estado do serviço e do modelo carregado."""
    if predictor is None:
        return HealthResponse(status="degraded", model_loaded=False)

    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=predictor.model_version,
        backend=predictor.backend,
        labels=predictor.labels,
    )


@app.get("/metrics", tags=["operacional"], include_in_schema=False)
def prometheus_metrics() -> Response:
    """Endpoint de scrape do Prometheus."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=TriageResponse, tags=["triagem"])
def predict(request: TriageRequest) -> TriageResponse:
    """Classifica a urgência de um laudo."""
    return _classify([request.text])[0]


@app.post("/predict/batch", response_model=BatchTriageResponse, tags=["triagem"])
def predict_batch(request: BatchTriageRequest) -> BatchTriageResponse:
    """Classifica um lote de laudos em uma única chamada."""
    results = _classify([item.text for item in request.items])
    return BatchTriageResponse(results=results, count=len(results))


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    """Contabiliza erros HTTP antes de devolvê-los."""
    if exc.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
        metrics.ERRORS_TOTAL.labels(type="server_error").inc()
    elif exc.status_code >= status.HTTP_400_BAD_REQUEST:
        metrics.ERRORS_TOTAL.labels(type="client_error").inc()

    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
