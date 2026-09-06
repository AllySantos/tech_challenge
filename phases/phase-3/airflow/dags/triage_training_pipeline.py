"""DAG de treino e retreino do classificador de triagem.

A DAG é uma casca fina: toda a lógica vive em ``src.pipeline`` e é testada
fora do Airflow. Cada task é idempotente e comunica apenas o identificador da
versão via XCom, de modo que os artefatos pesados fiquem no volume
compartilhado e não no banco de metadados do scheduler.

O grafo termina em um portão de qualidade — o modelo só é promovido para
serving se atender ao F1 macro mínimo e ao orçamento de latência p95.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task

# O projeto é montado em /opt/project no container do Airflow; incluí-lo no
# path permite reutilizar exatamente o mesmo código que roda no serving.
PROJECT_ROOT = Path("/opt/project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ARGS = {
    "owner": "grupo-102",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "depends_on_past": False,
}


@dag(
    dag_id="triage_training_pipeline",
    description="Ingestão, treino, otimização ONNX e promoção do classificador de urgência",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["fase-3", "nlp", "triagem"],
)
def triage_training_pipeline():
    """Pipeline semanal de retreino do modelo de triagem."""

    @task
    def ingest() -> str:
        """Baixa o corpus e deriva o rótulo de urgência."""
        from src.pipeline import run_ingest

        return run_ingest()

    @task
    def preprocess(_raw_path: str) -> dict:
        """Limpa os laudos e gera o split estratificado de treino/validação."""
        from src.pipeline import run_preprocess

        return run_preprocess()

    @task
    def train(_splits: dict) -> str:
        """Treina o pipeline TF-IDF + regressão logística em uma nova versão."""
        from src.pipeline import run_train

        return run_train()

    @task
    def export_optimized(version_dir: str) -> str:
        """Exporta as variantes ONNX e ONNX INT8 da versão treinada."""
        from src.pipeline import run_export

        run_export(version_dir)
        return version_dir

    @task
    def evaluate(version_dir: str) -> dict:
        """Mede accuracy e F1 por classe no conjunto de validação."""
        from src.pipeline import run_evaluate

        return run_evaluate(version_dir)

    @task
    def benchmark(version_dir: str) -> dict:
        """Compara a latência dos três backends de inferência."""
        from src.pipeline import run_benchmark

        return run_benchmark(version_dir)

    @task
    def quality_gate(version_dir: str, metrics: dict, latency: dict) -> dict:
        """Reprova a versão se o F1 macro ou o p95 de latência regredirem."""
        from src.pipeline import run_quality_gate

        return run_quality_gate(version_dir, metrics, latency)

    @task
    def promote(version_dir: str, metrics: dict, latency: dict, gate: dict) -> str:
        """Consolida o metadata e aponta o serving para a nova versão."""
        from src.pipeline import run_promote

        return run_promote(version_dir, metrics, latency, gate)

    raw_path = ingest()
    splits = preprocess(raw_path)
    version = export_optimized(train(splits))

    metrics = evaluate(version)
    latency = benchmark(version)
    gate = quality_gate(version, metrics, latency)

    promote(version, metrics, latency, gate)


triage_training_pipeline()
