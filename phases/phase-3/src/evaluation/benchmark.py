"""Benchmark de latência de inferência entre os backends disponíveis.

Mede o custo de classificar **um laudo por vez**, que é o padrão de uso da
triagem em tempo real. Latência média esconde o que importa em produção, então
o relatório é orientado a percentis — p95 é o número que sustenta o SLO.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import time
from pathlib import Path

from src.configs.settings import settings
from src.inference.predictor import BACKENDS, load_predictor
from src.models.export import size_mb
from src.models.registry import resolve_current

logger = logging.getLogger(__name__)


def _percentile(sorted_values: list[float], percentile: float) -> float:
    """Percentil pelo método nearest-rank, sobre a lista já ordenada.

    ``ceil`` em vez de ``round`` porque o arredondamento bancário do Python
    desloca meio índice em percentis como o p95 — justamente o que se quer
    medir com precisão.
    """
    if not sorted_values:
        raise ValueError("Nenhuma medição para calcular percentil")

    rank = math.ceil(percentile / 100 * len(sorted_values))
    index = min(max(rank - 1, 0), len(sorted_values) - 1)
    return sorted_values[index]


def measure(predictor, samples: list[str], runs: int, warmup: int) -> dict:
    """Cronometra ``runs`` inferências unitárias e devolve os percentis em ms."""
    for i in range(warmup):
        predictor.predict([samples[i % len(samples)]])

    latencies_ms: list[float] = []
    for i in range(runs):
        text = samples[i % len(samples)]
        started = time.perf_counter()
        predictor.predict([text])
        latencies_ms.append((time.perf_counter() - started) * 1000)

    latencies_ms.sort()
    return {
        "runs": runs,
        "mean_ms": round(statistics.fmean(latencies_ms), 3),
        "p50_ms": round(_percentile(latencies_ms, 50), 3),
        "p95_ms": round(_percentile(latencies_ms, 95), 3),
        "p99_ms": round(_percentile(latencies_ms, 99), 3),
        "throughput_rps": round(1000 / statistics.fmean(latencies_ms), 1),
    }


def load_samples(validation_path: Path | str | None = None, limit: int = 64) -> list[str]:
    """Carrega laudos reais da validação para servir de carga do benchmark."""
    import pandas as pd

    validation_path = Path(validation_path or settings.processed_dir / "validation.csv")
    return pd.read_csv(validation_path)["text"].head(limit).astype(str).tolist()


def run_benchmark(
    version_dir: Path | None = None,
    runs: int | None = None,
    warmup: int | None = None,
    samples: list[str] | None = None,
) -> dict:
    """Compara todos os backends com artefato disponível na versão dada."""
    version_dir = version_dir or resolve_current()
    if version_dir is None:
        raise FileNotFoundError("Nenhuma versão de modelo encontrada para benchmark")

    runs = runs or settings.benchmark_runs
    warmup = warmup or settings.benchmark_warmup
    samples = samples or load_samples()

    results: dict[str, dict] = {}
    for backend in BACKENDS:
        try:
            predictor = load_predictor(backend, version_dir)
        except FileNotFoundError as exc:
            logger.warning("Backend '%s' indisponível: %s", backend, exc)
            continue

        measurement = measure(predictor, samples, runs, warmup)
        artifact = version_dir / _artifact_name(backend)
        measurement["artifact_mb"] = round(size_mb(artifact), 3)
        results[backend] = measurement

        logger.info(
            "%-10s p50=%.2fms p95=%.2fms p99=%.2fms  %.1f req/s  %.2f MB",
            backend,
            measurement["p50_ms"],
            measurement["p95_ms"],
            measurement["p99_ms"],
            measurement["throughput_rps"],
            measurement["artifact_mb"],
        )

    return {
        "model_version": version_dir.name,
        "runs_per_backend": runs,
        "warmup": warmup,
        "backends": results,
        "speedup_vs_sklearn": _speedups(results),
    }


def _artifact_name(backend: str) -> str:
    from src.inference.predictor import _ARTIFACT_BY_BACKEND

    return _ARTIFACT_BY_BACKEND[backend]


def _speedups(results: dict[str, dict]) -> dict[str, float]:
    """Ganho de cada backend sobre a linha de base scikit-learn, no p95."""
    baseline = results.get("sklearn", {}).get("p95_ms")
    if not baseline:
        return {}
    return {
        backend: round(baseline / measurement["p95_ms"], 2)
        for backend, measurement in results.items()
        if measurement.get("p95_ms")
    }


def save_report(report: dict, output_path: Path | str | None = None) -> Path:
    """Persiste o relatório de latência em JSON."""
    output_path = Path(output_path or settings.reports_root / "latency_benchmark.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Relatório de latência gravado em %s", output_path)
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    save_report(run_benchmark())


if __name__ == "__main__":
    main()
