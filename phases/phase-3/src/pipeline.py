"""Etapas do pipeline de treino, isoladas da ferramenta de orquestração.

A DAG do Airflow é uma casca fina sobre estas funções. A consequência prática
é que o pipeline inteiro roda localmente (``make train``) e é coberto por
testes sem que o Airflow precise estar instalado.

O fluxo é: ingestão → preparação → treino → exportação otimizada → avaliação
→ benchmark → portão de qualidade → promoção. Uma versão só passa a ser
servida se atravessar o portão.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from src.configs.settings import URGENCY_LABELS, settings

logger = logging.getLogger(__name__)


class QualityGateError(RuntimeError):
    """Levantada quando o modelo recém-treinado não atende aos mínimos."""


def run_ingest() -> str:
    """Etapa 1 — baixa o corpus e deriva o rótulo de urgência."""
    from src.data.ingest import ingest

    return str(ingest())


def run_preprocess() -> dict[str, str]:
    """Etapa 2 — limpa os laudos e gera o split estratificado."""
    from src.data.preprocess import preprocess

    train_path, validation_path = preprocess()
    return {"train": str(train_path), "validation": str(validation_path)}


def run_train() -> str:
    """Etapa 3 — treina o pipeline em uma nova versão e devolve o diretório.

    O ``metadata.json`` é escrito já aqui, com as classes que o modelo
    aprendeu, para que a versão seja auto-descritiva desde o primeiro momento —
    as etapas de benchmark e serving dependem desse arquivo para saber a que
    rótulo corresponde cada índice de saída.
    """
    from src.models.registry import new_version, write_metadata
    from src.models.train import train

    version_dir = new_version()
    pipeline, _ = train(output_dir=version_dir)

    write_metadata(
        version_dir,
        {
            "version": version_dir.name,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "training",
            "labels": [str(label) for label in pipeline.named_steps["classifier"].classes_],
            "urgency_levels": list(URGENCY_LABELS),
            "serving_backend": settings.inference_backend,
            "training_config": _training_config(),
        },
    )
    return str(version_dir)


def _training_config() -> dict:
    """Hiperparâmetros efetivos do treino, registrados junto ao artefato."""
    return {
        "tfidf_max_features": settings.tfidf_max_features,
        "tfidf_min_df": settings.tfidf_min_df,
        "tfidf_ngram_max": settings.tfidf_ngram_max,
        "logreg_c": settings.logreg_c,
        "logreg_max_iter": settings.logreg_max_iter,
        "validation_size": settings.validation_size,
        "random_seed": settings.random_seed,
    }


def run_export(version_dir: str) -> dict[str, str]:
    """Etapa 4 — gera as variantes otimizadas: ONNX, INT8 e ONNX com pruning."""
    import joblib
    import pandas as pd

    from src.models.export import export_all
    from src.models.registry import PIPELINE_ARTIFACT

    path = Path(version_dir)
    pipeline = joblib.load(path / PIPELINE_ARTIFACT)

    # O pruning reajusta o classificador sobre o vocabulário reduzido, então
    # precisa do mesmo conjunto de treino usado na etapa anterior.
    train_df = pd.read_csv(settings.processed_dir / "train.csv")

    artifacts = export_all(pipeline, path, texts=train_df["text"], labels=train_df["urgency"])
    return {backend: str(artifact) for backend, artifact in artifacts.items()}


def run_evaluate(version_dir: str) -> dict:
    """Etapa 5 — mede a qualidade no conjunto de validação.

    O modelo avaliado é o que efetivamente vai a serving. Como o backend
    padrão serve a variante com pruning, o F1 reportado é o dela — não o do
    modelo cheio, que seria otimista. As métricas do baseline entram junto,
    para deixar explícito quanto o pruning custou.
    """
    import joblib

    from src.evaluation.evaluate import evaluate_pipeline, save_metrics
    from src.models.registry import PIPELINE_ARTIFACT

    path = Path(version_dir)
    baseline = evaluate_pipeline(joblib.load(path / PIPELINE_ARTIFACT))

    served_path = path / _served_pipeline_artifact()
    if served_path.exists() and served_path.name != PIPELINE_ARTIFACT:
        served = evaluate_pipeline(joblib.load(served_path))
    else:
        served = baseline

    metrics = dict(served)
    metrics["served_artifact"] = served_path.name
    metrics["baseline"] = {
        "accuracy": baseline["accuracy"],
        "macro_f1": baseline["macro_f1"],
    }
    metrics["macro_f1_delta_pp"] = round((served["macro_f1"] - baseline["macro_f1"]) * 100, 2)

    logger.info(
        "Modelo servido (%s): macro_f1=%.4f | baseline=%.4f | delta=%+.2f p.p.",
        served_path.name,
        served["macro_f1"],
        baseline["macro_f1"],
        metrics["macro_f1_delta_pp"],
    )

    save_metrics(metrics, path / "evaluation.json")
    save_metrics(metrics, settings.metrics_root / "evaluation.json")
    return metrics


def _served_pipeline_artifact() -> str:
    """Artefato scikit-learn correspondente ao backend configurado para serving."""
    from src.models.registry import PIPELINE_ARTIFACT, PRUNED_PIPELINE_ARTIFACT

    return (
        PRUNED_PIPELINE_ARTIFACT
        if settings.inference_backend.endswith("pruned")
        else PIPELINE_ARTIFACT
    )


def run_benchmark(version_dir: str) -> dict:
    """Etapa 6 — compara a latência dos backends desta versão."""
    from src.evaluation.benchmark import run_benchmark as benchmark
    from src.evaluation.benchmark import save_report

    path = Path(version_dir)
    report = benchmark(version_dir=path)
    save_report(report, path / "latency_benchmark.json")
    save_report(report, settings.reports_root / "latency_benchmark.json")
    return report


def run_quality_gate(version_dir: str, metrics: dict, benchmark_report: dict) -> dict:
    """Etapa 7 — barra a promoção se qualidade ou latência regredirem.

    São dois critérios independentes: o F1 macro protege a qualidade da
    triagem, e o p95 do backend de serving protege o SLO de resposta. Reprovar
    em um deles já impede a promoção.
    """
    macro_f1 = metrics["macro_f1"]
    serving = benchmark_report["backends"].get(settings.inference_backend, {})
    p95_ms = serving.get("p95_ms")

    checks = {
        "macro_f1": {
            "value": macro_f1,
            "threshold": settings.min_macro_f1,
            "passed": macro_f1 >= settings.min_macro_f1,
        },
        "p95_latency_ms": {
            "value": p95_ms,
            "threshold": settings.max_p95_latency_ms,
            "passed": p95_ms is not None and p95_ms <= settings.max_p95_latency_ms,
        },
    }

    failures = [name for name, check in checks.items() if not check["passed"]]
    if failures:
        raise QualityGateError(
            f"Versão {Path(version_dir).name} reprovada no portão de qualidade: "
            + "; ".join(
                f"{name}={checks[name]['value']} (limite {checks[name]['threshold']})"
                for name in failures
            )
        )

    logger.info("Portão de qualidade aprovado: %s", json.dumps(checks))
    return checks


def run_promote(
    version_dir: str,
    metrics: dict,
    benchmark_report: dict,
    quality_gate: dict,
) -> str:
    """Etapa 8 — grava o metadata e aponta o serving para a nova versão."""
    from src.models.registry import promote, read_metadata, write_metadata

    path = Path(version_dir)

    metadata = read_metadata(path)
    metadata.update(
        {
            "status": "promoted",
            "promoted_at": datetime.now(UTC).isoformat(),
            "labels": metrics["labels"],
            "metrics": metrics,
            "latency": benchmark_report,
            "quality_gate": quality_gate,
        }
    )
    write_metadata(path, metadata)

    promote(path)
    return path.name


def run_full_pipeline() -> str:
    """Executa o pipeline inteiro em processo. Usado por ``make train``."""
    run_ingest()
    run_preprocess()
    version_dir = run_train()
    run_export(version_dir)

    metrics = run_evaluate(version_dir)
    benchmark_report = run_benchmark(version_dir)
    quality_gate = run_quality_gate(version_dir, metrics, benchmark_report)

    return run_promote(version_dir, metrics, benchmark_report, quality_gate)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    version = run_full_pipeline()
    logger.info("Pipeline concluído. Versão promovida: %s", version)


if __name__ == "__main__":
    main()
