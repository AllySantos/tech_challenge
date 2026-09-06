"""Renderiza o relatório de latência em Markdown, pronto para o README."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BACKEND_LABELS = {
    "sklearn": "scikit-learn (baseline)",
    "onnx": "ONNX Runtime",
    "onnx-int8": "ONNX Runtime INT8",
    "onnx-pruned": "ONNX Runtime + pruning",
}


def render(report: dict) -> str:
    """Monta a tabela comparativa entre os backends medidos."""
    speedups = report.get("speedup_vs_sklearn", {})

    lines = [
        f"Versão do modelo: `{report['model_version']}` · "
        f"{report['runs_per_backend']} inferências unitárias por backend "
        f"(após {report['warmup']} de aquecimento).",
        "",
        "| Backend | p50 | p95 | p99 | Throughput | Artefato | Ganho no p95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for backend, measurement in report["backends"].items():
        speedup = speedups.get(backend)
        lines.append(
            f"| {BACKEND_LABELS.get(backend, backend)} "
            f"| {measurement['p50_ms']:.2f} ms "
            f"| {measurement['p95_ms']:.2f} ms "
            f"| {measurement['p99_ms']:.2f} ms "
            f"| {measurement['throughput_rps']:.0f} req/s "
            f"| {measurement['artifact_mb']:.2f} MB "
            f"| {f'{speedup:.2f}×' if speedup else '—'} |"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="reports/latency_benchmark.json")
    parser.add_argument("--output", default=None, help="Se omitido, imprime na saída padrão")
    args = parser.parse_args()

    markdown = render(json.loads(Path(args.report).read_text(encoding="utf-8")))

    if args.output:
        Path(args.output).write_text(markdown + "\n", encoding="utf-8")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
