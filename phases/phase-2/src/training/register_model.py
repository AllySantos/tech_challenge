"""Registra o melhor modelo no MLflow Model Registry e promove via alias.

Usa aliases (`staging`, `production`) em vez das antigas *stages* do
Model Registry — a partir do MLflow 2.9 as stages foram descontinuadas em
favor de aliases, e é isso que a versão do MLflow fixada no projeto
(`^3.11`, ver pyproject.toml) espera. O fluxo "Staging → Production" pedido
no enunciado é reproduzido com `--to-alias staging` e depois
`--to-alias production`, uma promoção manual de cada vez — promoção
automática para produção sem revisão humana não é uma boa prática.

Uso:
    # 1) Registra a melhor run (por padrão, maior ndcg_at_10) e promove a staging
    python -m src.training.register_model --to-alias staging

    # 2) Depois de validar manualmente, promove a mesma versão a produção
    python -m src.training.register_model --to-alias production --version 1
"""

from __future__ import annotations

import argparse
import os

import mlflow
from mlflow import MlflowClient


def find_best_run(experiment_name: str, metric: str, model_family: str) -> str:
    """Encontra o run com a melhor métrica entre os runs de um model_family.

    Args:
        experiment_name: Nome do experiment no MLflow.
        metric: Métrica usada para ranquear os runs (maior é melhor).
        model_family: Valor da tag `model_family` a filtrar (ex.: "embedding_mlp").

    Returns:
        O `run_id` do melhor run.

    Raises:
        ValueError: Se nenhum run correspondente for encontrado.
    """
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=f"tags.model_family = '{model_family}'",
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if runs.empty:
        raise ValueError(
            f"Nenhum run com tag model_family='{model_family}' encontrado "
            f"no experiment '{experiment_name}'. Rode `dvc repro` primeiro."
        )
    return runs.iloc[0]["run_id"]


def register_and_promote(run_id: str, model_name: str, alias: str) -> int:
    """Registra o modelo do run (se ainda não registrado) e aplica um alias.

    Args:
        run_id: Run que contém o artefato logado via `mlflow.pytorch.log_model`.
        model_name: Nome do modelo registrado no Model Registry.
        alias: Alias a aplicar (`staging`, `production`, etc.).

    Returns:
        O número da versão registrada/promovida.
    """
    client = MlflowClient()
    model_version = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=model_name)
    client.set_registered_model_alias(name=model_name, alias=alias, version=model_version.version)
    return int(model_version.version)


def main() -> None:
    """Ponto de entrada do CLI de registro/promoção."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="ecommerce-recsys-mlp")
    parser.add_argument("--metric", default="ndcg_at_10")
    parser.add_argument("--model-family", default="embedding_mlp")
    parser.add_argument("--to-alias", choices=["staging", "production"], required=True)
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Promove uma versão já registrada específica em vez de buscar o melhor run.",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "recsys-phase-2")

    if args.version is not None:
        MlflowClient().set_registered_model_alias(
            name=args.model_name, alias=args.to_alias, version=args.version
        )
        print(f"Versão {args.version} de '{args.model_name}' promovida a '@{args.to_alias}'.")
        return

    run_id = find_best_run(experiment_name, args.metric, args.model_family)
    version = register_and_promote(run_id, args.model_name, args.to_alias)
    print(
        f"Run {run_id} (melhor {args.metric}) registrado como "
        f"'{args.model_name}' versão {version}, promovido a '@{args.to_alias}'."
    )


if __name__ == "__main__":
    main()
