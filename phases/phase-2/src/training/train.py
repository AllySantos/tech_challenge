"""Treino do Modelo

Treina um modelo embedding-based (via `ModelFactory`) com negative sampling,
early stopping em validação, e loga params/métricas/artefatos no MLflow a
cada execução.

"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.model_factory import create_model


def set_seed(seed: int) -> None:
    """Fixa as sementes de aleatoriedade para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_mlflow() -> None:
    """Aponta o MLflow pro tracking server e experiment corretos."""
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "recsys-phase-2"))


def load_params(params_path: Path) -> dict:
    """Carrega a seção `train` do arquivo de parâmetros do DVC."""
    return yaml.safe_load(params_path.read_text())["train"]


def load_feature_meta(features_dir: Path) -> dict:
    """Carrega `num_users`/`num_items` gerados pelo stage `feature_eng`."""
    return json.loads((features_dir / "feature_meta.json").read_text())


def load_split(features_dir: Path, split_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Carrega os arrays codificados (`user_idx`, `item_idx`) de um split."""
    data = np.load(features_dir / f"{split_name}.npz")
    return data["user_idx"], data["item_idx"]


def build_training_batch(
    user_idx: np.ndarray, item_idx: np.ndarray, num_items: int, negative_ratio: int, seed: int
) -> TensorDataset:
    """Monta um dataset com exemplos positivos (label=1) e negativos (label=0).

    Args:
        user_idx: Índices de usuário das interações positivas.
        item_idx: Índices de item das interações positivas.
        num_items: Total de itens distintos (para sortear negativos).
        negative_ratio: Quantos negativos gerar por positivo.
        seed: Semente para o sorteio dos negativos.

    Returns:
        `TensorDataset` com (user_idx, item_idx, label).
    """
    rng = np.random.default_rng(seed)
    n = len(user_idx)
    neg_users = np.repeat(user_idx, negative_ratio)
    neg_items = rng.integers(0, num_items, size=n * negative_ratio)

    all_users = np.concatenate([user_idx, neg_users])
    all_items = np.concatenate([item_idx, neg_items])
    labels = np.concatenate([np.ones(n), np.zeros(n * negative_ratio)])

    return TensorDataset(
        torch.as_tensor(all_users, dtype=torch.long),
        torch.as_tensor(all_items, dtype=torch.long),
        torch.as_tensor(labels, dtype=torch.float32),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    """Roda uma época de treino (se `optimizer` for informado) ou avaliação.

    Args:
        model: Modelo a treinar/avaliar.
        loader: DataLoader da época.
        criterion: Função de perda (BCEWithLogitsLoss).
        optimizer: Otimizador; se `None`, roda em modo avaliação (sem grad).

    Returns:
        Perda média da época.
    """
    model.train(optimizer is not None)
    total_loss, total_examples = 0.0, 0
    for user_idx, item_idx, labels in loader:
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(user_idx, item_idx)
            loss = criterion(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(labels)
        total_examples += len(labels)
    return total_loss / total_examples


class EarlyStopper:
    """Rastreia o melhor `state_dict` visto e decide quando parar o treino.
    """

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_loss = float("inf")
        self.best_state: dict | None = None
        self._epochs_without_improvement = 0

    def update(self, val_loss: float, model: nn.Module) -> bool:
        """Registra a perda da época; retorna `True` se o treino deve parar."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_state = model.state_dict()
            self._epochs_without_improvement = 0
            return False
        self._epochs_without_improvement += 1
        return self._epochs_without_improvement >= self.patience


def build_model(params: dict, meta: dict) -> nn.Module:
    """Instancia o modelo via Factory, a partir dos params e do feature_meta."""
    return create_model(
        model_type="embedding_mlp",
        num_users=meta["num_users"],
        num_items=meta["num_items"],
        embedding_dim=params["embedding_dim"],
        hidden_dims=params["hidden_dims"],
        dropout=params["dropout"],
    )


def build_val_loader(
    val_users: np.ndarray, val_items: np.ndarray, meta: dict, params: dict
) -> DataLoader:
    """Monta o DataLoader de validação (negative sampling com seed fixa)."""
    dataset = build_training_batch(
        val_users, val_items, meta["num_items"], params["negative_ratio"], params["seed"]
    )
    return DataLoader(dataset, batch_size=params["batch_size"])


def build_train_loader(
    train_users: np.ndarray, train_items: np.ndarray, meta: dict, params: dict, epoch: int
) -> DataLoader:
    """Monta o DataLoader de treino da época (negative sampling varia por época)."""
    dataset = build_training_batch(
        train_users, train_items, meta["num_items"], params["negative_ratio"], params["seed"] + epoch
    )
    return DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)


def train_with_early_stopping(
    model: nn.Module,
    train_users: np.ndarray,
    train_items: np.ndarray,
    val_loader: DataLoader,
    meta: dict,
    params: dict,
) -> EarlyStopper:
    """Roda o laço de épocas com early stopping, logando cada época no MLflow.

    Returns:
        O `EarlyStopper` usado, com o melhor `state_dict` e a melhor perda.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()
    stopper = EarlyStopper(patience=params["early_stopping_patience"])

    for epoch in range(params["num_epochs"]):
        train_loader = build_train_loader(train_users, train_items, meta, params, epoch)
        train_loss = run_epoch(model, train_loader, criterion, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, None)
        mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if stopper.update(val_loss, model):
            print(f"Early stopping na época {epoch}.")
            break
    return stopper


def save_model_artifacts(
    model: nn.Module,
    stopper: EarlyStopper,
    model_dir: Path,
    run_id: str,
    meta: dict,
    params: dict,
) -> None:
    """Salva o `state_dict` local (dep. do stage `evaluate`) e loga no MLflow.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(stopper.best_state, model_dir / "model.pt")
    model.load_state_dict(stopper.best_state)

    mlflow.set_tag("model_family", "embedding_mlp")
    mlflow.pytorch.log_model(model, artifact_path="model")

    model_meta = {
        "model_type": "embedding_mlp",
        "num_users": meta["num_users"],
        "num_items": meta["num_items"],
        "embedding_dim": params["embedding_dim"],
        "hidden_dims": params["hidden_dims"],
        "dropout": params["dropout"],
        "best_val_loss": stopper.best_loss,
        "mlflow_run_id": run_id,
    }
    (model_dir / "model_meta.json").write_text(json.dumps(model_meta, indent=2))


def run_training(params: dict) -> None:
    """Executa o treino completo: carrega dados, treina, avalia e salva.

    Args:
        params: Seção `train` do `params.yaml` já carregada.
    """
    features_dir = Path(params["input_dir"])
    meta = load_feature_meta(features_dir)
    set_seed(params["seed"])
    configure_mlflow()

    train_users, train_items = load_split(features_dir, "train")
    val_users, val_items = load_split(features_dir, "val")
    model = build_model(params, meta)
    val_loader = build_val_loader(val_users, val_items, meta, params)

    with mlflow.start_run() as run:
        mlflow.log_params({**params, "num_users": meta["num_users"], "num_items": meta["num_items"]})
        stopper = train_with_early_stopping(model, train_users, train_items, val_loader, meta, params)
        save_model_artifacts(model, stopper, Path(params["model_dir"]), run.info.run_id, meta, params)

    print(f"train: melhor val_loss={stopper.best_loss:.4f} (mlflow_run_id={run.info.run_id})")


def main() -> None:
    """Ponto de entrada do stage `train`: parseia argumentos e delega a `run_training`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    args = parser.parse_args()
    run_training(load_params(args.params))


if __name__ == "__main__":
    main()