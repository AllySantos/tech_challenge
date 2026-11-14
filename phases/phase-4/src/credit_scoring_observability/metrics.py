"""Métricas Prometheus do pipeline de monitoramento."""

import time

from prometheus_client import Counter, Gauge

# Counters (o sufixo _total é adicionado automaticamente pela biblioteca)
BATCHES = Counter(
    "credit_pipeline_batches", "Lotes processados, por status", ["status"]
)
ALERTS = Counter("credit_alerts", "Alertas disparados", ["severity", "rule"])

# Gauges (valor do último lote)
BASELINE_AUC = Gauge("credit_baseline_auc", "AUC do modelo baseline no treino")
CONTRACT_OK = Gauge(
    "credit_data_contract_ok",
    "1 se o último lote passou no contrato, 0 se foi bloqueado",
)
ROWS = Gauge("credit_last_batch_rows", "Linhas do último lote")
AUC = Gauge("credit_model_auc", "AUC do modelo no último lote")
RECALL = Gauge("credit_model_recall", "Recall de inadimplentes no último lote")
DEFAULT_RATE = Gauge("credit_model_default_rate", "Taxa real de default no último lote")
PSI = Gauge(
    "credit_drift_psi", "PSI por variável (referência x último lote)", ["feature"]
)
PSI_MAX = Gauge("credit_drift_psi_max", "Maior PSI entre as variáveis")
KS_SHARE = Gauge(
    "credit_drift_ks_share", "Fração de variáveis que o KS acusa como diferentes"
)
STAGE_SECONDS = Gauge(
    "credit_pipeline_stage_seconds", "Duração por etapa do pipeline", ["stage"]
)
LAST_BATCH_TS = Gauge(
    "credit_last_batch_timestamp_seconds", "Quando o último lote terminou"
)

STATUSES = ("ok", "warning", "critical", "blocked")


def init(baseline_auc: float) -> None:
    """Inicializa as métricas do Prometheus antes do processamento dos lotes."""
    BASELINE_AUC.set(baseline_auc)

    # Inicializa os contadores em zero para o Prometheus reconhecê-los.
    for status in STATUSES:
        BATCHES.labels(status=status).inc(0)

    # Evita valores artificiais de zero antes do primeiro lote válido.
    # NaN representa uma medição ainda não disponível.
    for gauge in (AUC, RECALL, DEFAULT_RATE, PSI_MAX, KS_SHARE):
        gauge.set(float("nan"))


def publish(result: dict) -> None:
    """Publica o resultado de um lote (o dict devolvido por process_batch)."""
    m = result["metrics"]

    BATCHES.labels(status=result["status"]).inc()

    for severity, rule, _ in result["alerts"]:
        ALERTS.labels(severity=severity, rule=rule).inc()

    ROWS.set(m["rows"])
    CONTRACT_OK.set(m["contract_ok"])
    STAGE_SECONDS.labels(stage="total").set(m["latency_total_s"])
    LAST_BATCH_TS.set(time.time())

    if m["contract_ok"]:  # Lote bloqueado não tem predição nem drift.
        AUC.set(m["auc"])
        RECALL.set(m["recall"])
        DEFAULT_RATE.set(m["default_rate"])
        PSI_MAX.set(m["psi_max"])
        KS_SHARE.set(m["ks_drifted_share"])

        STAGE_SECONDS.labels(stage="score").set(m["latency_score_s"])
        STAGE_SECONDS.labels(stage="drift").set(m["latency_drift_s"])

        for key, value in m.items():
            if key.startswith("psi/"):
                PSI.labels(feature=key[4:]).set(value)
