"""Pipeline de produção: valida, prediz, mede drift e dispara alertas por lote.

Cada lote gera logs estruturados (logs/pipeline.log) e métricas
centralizadas no MLflow.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import pandera.pandas as pa
from prometheus_client import start_http_server
from sklearn.metrics import recall_score, roc_auc_score

from credit_scoring_observability import metrics
from credit_scoring_observability.contracts import CONTRACT
from credit_scoring_observability.data import TARGET
from credit_scoring_observability.drift import KS_ALPHA, PSI_THRESHOLD, run_drift
from credit_scoring_observability.make_bad_batch import BAD_BATCH_CSV
from credit_scoring_observability.prep import REFERENCE_CSV
from credit_scoring_observability.simulate_drift import PERIODS, period_path
from credit_scoring_observability.train import METRICS_PATH, MODEL_PATH, feature_columns

LOG_PATH = Path("logs/pipeline.log")
MLFLOW_URI = "sqlite:///mlflow.db"
EXPERIMENT = "credit-scoring-monitoring"
METRICS_PORT = 8000

# Regras de alerta (documentadas no README)
PSI_ATTENTION = 0.1
KS_SHARE_WARN = 0.25
AUC_WARN_DROP = 0.03
AUC_CRIT_DROP = 0.05


# ---------- Logs estruturados ----------
class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "event": record.getMessage(),
            **getattr(record, "ctx", {}),
        }
        return json.dumps(payload, ensure_ascii=False, default=str)


def build_logger() -> logging.Logger:
    logger = logging.getLogger("pipeline")
    if not logger.handlers:
        LOG_PATH.parent.mkdir(exist_ok=True)
        for handler in (
            logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ):
            handler.setFormatter(JsonFormatter())
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


LOG = build_logger()


def event(level: int, name: str, **ctx) -> None:
    LOG.log(level, name, extra={"ctx": ctx})


# ---------- Regras de alerta ----------
def evaluate_alerts(
    psi: dict, ks_share: float, auc: float, baseline_auc: float
) -> list:
    alerts = []

    for col, value in psi.items():
        if value >= PSI_THRESHOLD:
            alerts.append(("critical", "psi_drift", f"{col} PSI={value:.3f}"))
        elif value >= PSI_ATTENTION:
            alerts.append(("warning", "psi_attention", f"{col} PSI={value:.3f}"))

    if ks_share >= KS_SHARE_WARN:
        alerts.append(("warning", "ks_drift", f"KS acusa {ks_share:.0%} das variáveis"))

    drop = baseline_auc - auc

    if drop >= AUC_CRIT_DROP:
        alerts.append(("critical", "auc_drop", f"AUC caiu {drop:.3f} vs baseline"))
    elif drop >= AUC_WARN_DROP:
        alerts.append(("warning", "auc_drop", f"AUC caiu {drop:.3f} vs baseline"))

    return alerts


def status_from(alerts: list) -> str:
    levels = {a[0] for a in alerts}
    return (
        "critical"
        if "critical" in levels
        else "warning" if "warning" in levels else "ok"
    )


# ---------- Processamento de um lote ----------
def process_batch(name, step, df, reference, model, features, baseline_auc) -> dict:
    t_start = time.perf_counter()
    event(logging.INFO, "batch_received", batch=name, rows=len(df))

    result = {
        "step": step,
        "batch": name,
        "metrics": {"rows": float(len(df))},
    }

    # 1) Contrato: dados inválidos são bloqueados.
    try:
        CONTRACT.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        grouped = err.failure_cases.groupby(["column", "check"]).size()
        violations = {f"{col}: {chk}": int(n) for (col, chk), n in grouped.items()}

        event(
            logging.ERROR,
            "contract_failed",
            batch=name,
            violations=violations,
        )

        alerts = [
            (
                "critical",
                "contract_failed",
                f"{sum(violations.values())} violações",
            )
        ]

        result.update(status="blocked", alerts=alerts)
        result["metrics"].update(
            contract_ok=0.0,
            alerts_critical=1.0,
            alerts_warning=0.0,
            latency_total_s=time.perf_counter() - t_start,
        )

        event(
            logging.CRITICAL,
            "alert",
            batch=name,
            severity="critical",
            rule="contract_failed",
            detail=alerts[0][2],
        )

        return result

    event(logging.INFO, "contract_passed", batch=name)

    # 2) Predição
    t = time.perf_counter()
    score = model.predict_proba(df[features])[:, 1]
    latency_score = time.perf_counter() - t

    event(
        logging.INFO,
        "batch_scored",
        batch=name,
        seconds=round(latency_score, 4),
    )

    # 3) Drift: PSI e KS contra a referência
    t = time.perf_counter()
    _, psi = run_drift(reference, df, features, "psi", PSI_THRESHOLD)
    _, ks = run_drift(reference, df, features, "ks", KS_ALPHA)

    latency_drift = time.perf_counter() - t
    ks_share = sum(p < KS_ALPHA for p in ks.values()) / len(ks)

    event(
        logging.INFO,
        "drift_computed",
        batch=name,
        psi_max=round(max(psi.values()), 4),
        ks_share=round(ks_share, 3),
        seconds=round(latency_drift, 2),
    )

    # 4) Desempenho: assumimos que o resultado real já chegou.
    auc = float(roc_auc_score(df[TARGET], score))
    recall = float(recall_score(df[TARGET], score >= 0.5))

    # 5) Alertas
    alerts = evaluate_alerts(psi, ks_share, auc, baseline_auc)

    for level, rule, detail in alerts:
        event(
            logging.CRITICAL if level == "critical" else logging.WARNING,
            "alert",
            batch=name,
            severity=level,
            rule=rule,
            detail=detail,
        )

    result.update(status=status_from(alerts), alerts=alerts)

    result["metrics"].update(
        contract_ok=1.0,
        auc=auc,
        recall=recall,
        default_rate=float(df[TARGET].mean()),
        psi_max=max(psi.values()),
        ks_drifted_share=ks_share,
        alerts_critical=float(sum(a[0] == "critical" for a in alerts)),
        alerts_warning=float(sum(a[0] == "warning" for a in alerts)),
        latency_score_s=latency_score,
        latency_drift_s=latency_drift,
        latency_total_s=time.perf_counter() - t_start,
        **{f"psi/{col}": value for col, value in psi.items()},
    )

    event(logging.INFO, "batch_done", batch=name, status=result["status"])
    return result


# ---------- MLflow ----------
def log_to_mlflow(results: list, baseline_auc: float) -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=time.strftime("monitoring-%Y%m%d-%H%M%S")):
        mlflow.log_params(
            {
                "model": "xgboost-baseline",
                "baseline_auc": baseline_auc,
                "psi_threshold": PSI_THRESHOLD,
                "psi_attention": PSI_ATTENTION,
                "ks_alpha": KS_ALPHA,
                "auc_warn_drop": AUC_WARN_DROP,
                "auc_crit_drop": AUC_CRIT_DROP,
            }
        )

        for r in results:
            for key, value in r["metrics"].items():
                mlflow.log_metric(key, value, step=r["step"])

        mlflow.log_artifact(str(LOG_PATH))
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
        mlflow.log_artifacts("reports", artifact_path="reports")


# ---------- Execução principal ----------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline de monitoramento do credit scoring"
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help=f"expõe métricas Prometheus em :{METRICS_PORT} e fica no ar no final",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="segundos de espera entre lotes (com --serve, use 15)",
    )
    args = parser.parse_args()

    reference = pd.read_csv(REFERENCE_CSV)
    features = feature_columns(reference)
    model = joblib.load(MODEL_PATH)
    baseline_auc = json.loads(METRICS_PATH.read_text())["roc_auc"]

    if args.serve:
        start_http_server(METRICS_PORT)
        metrics.init(baseline_auc)
        event(
            logging.INFO,
            "metrics_server_started",
            port=METRICS_PORT,
        )

    batches = [("bad_batch", BAD_BATCH_CSV)] + [
        (f"period_{k}", period_path(k)) for k in PERIODS
    ]

    event(
        logging.INFO,
        "pipeline_started",
        batches=len(batches),
        baseline_auc=baseline_auc,
    )

    results = []

    for step, (name, path) in enumerate(batches):
        result = process_batch(
            name,
            step,
            pd.read_csv(path),
            reference,
            model,
            features,
            baseline_auc,
        )
        results.append(result)

        if args.serve:
            metrics.publish(result)
            time.sleep(args.delay)

    event(
        logging.INFO,
        "pipeline_finished",
        batches=len(results),
    )

    log_to_mlflow(results, baseline_auc)

    table = pd.DataFrame(
        [
            {
                "lote": r["batch"],
                "status": r["status"],
                "auc": r["metrics"].get("auc"),
                "psi_max": r["metrics"].get("psi_max"),
                "crit": int(r["metrics"]["alerts_critical"]),
                "warn": int(r["metrics"]["alerts_warning"]),
            }
            for r in results
        ]
    )

    print("\n", table.round(3).to_string(index=False))
    print(
        "\nVeja o MLflow com: "
        "poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db"
    )

    if args.serve:
        print(
            f"\nMétricas no ar em "
            f"http://localhost:{METRICS_PORT}/metrics "
            "(Ctrl+C para encerrar)"
        )
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
