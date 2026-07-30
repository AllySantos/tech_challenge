"""Training script for MLP recommendation model."""

import json
import logging
import random
from pathlib import Path

import mlflow
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.configs.settings import settings

logger = logging.getLogger(__name__)


def set_seeds(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLPRecommender(nn.Module):
    """MLP-based recommendation model using user embeddings."""

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 32) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass combining user and item embeddings."""
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        x = torch.cat([user_emb, item_emb], dim=1)
        return self.mlp(x).squeeze()


def prepare_data(
    matrix: sp.csr_matrix,
    negative_ratio: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create positive and negative samples from interaction matrix."""
    pos_users, pos_items = matrix.nonzero()
    n_positive = len(pos_users)

    n_users, n_items = matrix.shape
    rng = np.random.default_rng(42)

    n_negative_needed = n_positive * negative_ratio
    u_neg = rng.integers(0, n_users, size=n_negative_needed * 3)
    i_neg = rng.integers(0, n_items, size=n_negative_needed * 3)

    mask = np.array(matrix[u_neg, i_neg]).flatten() == 0
    u_neg = u_neg[mask][:n_negative_needed]
    i_neg = i_neg[mask][:n_negative_needed]

    neg_users = u_neg.tolist()
    neg_items = i_neg.tolist()
    all_users = np.concatenate([pos_users, neg_users])
    all_items = np.concatenate([pos_items, neg_items])
    all_labels = np.concatenate(
        [
            np.ones(n_positive),
            np.zeros(n_positive * negative_ratio),
        ]
    ).astype(np.float32)

    return (
        torch.tensor(all_users, dtype=torch.long),
        torch.tensor(all_items, dtype=torch.long),
        torch.tensor(all_labels, dtype=torch.float32),
    )


def train(
    matrix_path: str,
    output_path: str,
    metrics_path: str,
) -> None:
    """Train MLP recommender and log to MLflow."""
    set_seeds(settings.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    import scipy.sparse as sp

    matrix = sp.load_npz(matrix_path)
    n_users, n_items = matrix.shape
    logger.info("Matrix shape: %s", matrix.shape)

    users, items, labels = prepare_data(matrix)
    dataset = TensorDataset(users, items, labels)
    loader = DataLoader(dataset, batch_size=settings.batch_size, shuffle=True)

    model = MLPRecommender(n_users, n_items).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    criterion = nn.BCELoss()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "n_users": n_users,
                "n_items": n_items,
                "batch_size": settings.batch_size,
                "learning_rate": settings.learning_rate,
                "num_epochs": settings.num_epochs,
                "random_seed": settings.random_seed,
            }
        )

        best_loss = float("inf")
        patience, patience_counter = 5, 0

        for epoch in range(settings.num_epochs):
            model.train()
            total_loss = 0.0
            for user_batch, item_batch, label_batch in loader:
                user_batch = user_batch.to(device)
                item_batch = item_batch.to(device)
                label_batch = label_batch.to(device)

                optimizer.zero_grad()
                preds = model(user_batch, item_batch)
                loss = criterion(preds, label_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            logger.info(
                "Epoch %d/%d - Loss: %.4f",
                epoch + 1,
                settings.num_epochs,
                avg_loss,
            )

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_path)

        # Logs the model as a first-class MLflow "logged model" entity
        # (not just a raw file artifact), so it can later be registered
        # with mlflow.register_model(model_info.model_uri, ...).
        # serialization_format="pickle" avoids MLflow's default 'pt2'
        # (torch.export) format, which requires tracing the model with a
        # concrete input_example — unnecessary overhead for this project.
        model_info = mlflow.pytorch.log_model(
            model, name="recommender", serialization_format="pickle"
        )
        mlflow.set_tag("model_uri", model_info.model_uri)
        mlflow.log_artifact(output_path)

        metrics = {"best_loss": best_loss, "epochs_trained": epoch + 1}
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
        mlflow.log_metrics(metrics)

        logger.info("Model saved to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train(
        matrix_path=f"{settings.data_processed_path}/interaction_matrix.npz",
        output_path="models/recommender.pt",
        metrics_path="metrics/train_metrics.json",
    )