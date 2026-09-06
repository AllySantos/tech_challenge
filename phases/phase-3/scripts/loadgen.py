"""Gerador de carga para alimentar os painéis do Grafana durante a demo.

Envia laudos reais do conjunto de validação em ritmo constante e injeta uma
fração de requisições inválidas, para que o painel de taxa de erro tenha o que
mostrar em vez de ficar zerado.
"""

from __future__ import annotations

import argparse
import logging
import random
import time

import httpx

logger = logging.getLogger("loadgen")

# Trechos representativos de cada nível de urgência, usados quando o CSV de
# validação não está disponível dentro do container.
FALLBACK_ABSTRACTS = [
    "acute myocardial infarction with st segment elevation and persistent chest pain "
    "radiating to the left arm, troponin markedly elevated on admission",
    "cerebral infarction following occlusion of the middle cerebral artery, sudden onset "
    "hemiparesis and aphasia documented within the therapeutic window",
    "adenocarcinoma of the colon staged after resection, lymph node involvement assessed "
    "for adjuvant chemotherapy planning",
    "chronic gastritis associated with helicobacter pylori colonization, endoscopic biopsy "
    "showing mild inflammatory infiltrate",
    "routine follow up of a stable patient with controlled metabolic parameters and no "
    "evidence of acute inflammatory process on laboratory screening",
]


def load_samples(path: str, limit: int) -> list[str]:
    """Carrega laudos do CSV de validação, caindo para os exemplos embutidos."""
    try:
        import pandas as pd

        texts = pd.read_csv(path)["text"].head(limit).astype(str).tolist()
        if texts:
            logger.info("Carregados %d laudos de %s", len(texts), path)
            return texts
    except Exception as exc:  # noqa: BLE001 - fallback é o comportamento esperado
        logger.info("Usando laudos embutidos (%s indisponível: %s)", path, exc)

    return FALLBACK_ABSTRACTS


def run(url: str, duration: int, rps: float, error_rate: float, samples: list[str]) -> None:
    """Envia requisições até esgotar ``duration`` segundos."""
    interval = 1 / rps if rps > 0 else 0
    deadline = time.monotonic() + duration
    sent = errors = 0

    with httpx.Client(base_url=url, timeout=10.0) as client:
        while time.monotonic() < deadline:
            started = time.monotonic()

            if random.random() < error_rate:
                # Corpo inválido de propósito: exercita o caminho de erro 422.
                payload = {"text": ""}
            else:
                payload = {"text": random.choice(samples)}

            try:
                response = client.post("/predict", json=payload)
                sent += 1
                if response.status_code >= 400:
                    errors += 1
            except httpx.HTTPError as exc:
                errors += 1
                logger.warning("Falha na requisição: %s", exc)

            if sent % 100 == 0 and sent:
                logger.info("%d requisições enviadas (%d com erro)", sent, errors)

            remaining = interval - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)

    logger.info("Concluído: %d requisições, %d respostas de erro", sent, errors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL da API")
    parser.add_argument("--duration", type=int, default=120, help="Duração em segundos")
    parser.add_argument("--rps", type=float, default=8.0, help="Requisições por segundo")
    parser.add_argument(
        "--error-rate", type=float, default=0.05, help="Fração de requisições inválidas"
    )
    parser.add_argument(
        "--samples", default="data/processed/validation.csv", help="CSV com a coluna 'text'"
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    run(args.url, args.duration, args.rps, args.error_rate, load_samples(args.samples, 200))


if __name__ == "__main__":
    main()
