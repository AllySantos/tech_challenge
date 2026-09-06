"""Instrumentação Prometheus da API de triagem.

Os buckets de latência são declarados em milissegundos convertidos para
segundos e concentrados abaixo de 100 ms — os defaults do prometheus_client
começam em 5 ms e saltam para 10 ms, o que colapsaria toda a distribuição
deste serviço em um ou dois buckets e tornaria o p95 inútil.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

LATENCY_BUCKETS = (
    0.001,
    0.0025,
    0.005,
    0.0075,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.25,
    0.5,
    1.0,
)

REQUESTS_TOTAL = Counter(
    "triage_requests_total",
    "Total de requisições HTTP recebidas pela API de triagem.",
    ["method", "endpoint", "status_code"],
)

REQUEST_DURATION = Histogram(
    "triage_request_duration_seconds",
    "Duração ponta a ponta das requisições HTTP.",
    ["method", "endpoint"],
    buckets=LATENCY_BUCKETS,
)

INFERENCE_DURATION = Histogram(
    "triage_inference_duration_seconds",
    "Tempo gasto exclusivamente na inferência do modelo.",
    ["backend"],
    buckets=LATENCY_BUCKETS,
)

PREDICTIONS_TOTAL = Counter(
    "triage_predictions_total",
    "Laudos classificados, particionados por nível de urgência.",
    ["urgency"],
)

PREDICTION_CONFIDENCE = Histogram(
    "triage_prediction_confidence",
    "Distribuição da confiança da classe vencedora.",
    ["urgency"],
    buckets=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)

ERRORS_TOTAL = Counter(
    "triage_errors_total",
    "Erros tratados pela API, por tipo.",
    ["type"],
)

MODEL_INFO = Gauge(
    "triage_model_info",
    "Modelo atualmente carregado (sempre 1; a informação está nos rótulos).",
    ["version", "backend"],
)

MODEL_LOADED = Gauge(
    "triage_model_loaded",
    "1 quando há modelo carregado e a API pode servir predições, 0 caso contrário.",
)


def record_prediction(urgency: str, confidence: float) -> None:
    """Registra o resultado de uma classificação."""
    PREDICTIONS_TOTAL.labels(urgency=urgency).inc()
    PREDICTION_CONFIDENCE.labels(urgency=urgency).observe(confidence)
