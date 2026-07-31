import json
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from src.data.preprocess import load_events, preprocess
from src.evaluation.evaluator import (
    evaluate,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.features.feature_engineer import create_interaction_matrix, run_feature_engineering
from src.models.baseline import PopularityRecommender
from src.models.factory import ModelFactory
import src.models.train as train_module
from src.models.train import MLPRecommender, prepare_data, set_seeds


def test_load_events_reads_csv(tmp_path):
    input_path = tmp_path / "events.csv"
    input_path.write_text("visitorid,itemid,event\n1,10,view\n", encoding="utf-8")

    df = load_events(str(input_path))

    assert df.loc[0, "visitorid"] == 1
    assert df.loc[0, "itemid"] == 10
    assert df.loc[0, "event"] == "view"


def test_preprocess_writes_encoded_csv(tmp_path):
    input_path = tmp_path / "events.csv"
    output_path = tmp_path / "events_processed.csv"

    input_path.write_text(
        "visitorid,itemid,event\n"
        + "1,10,view\n" * 5
        + "1,11,view\n" * 5
        + "2,10,view\n" * 5
        + "2,11,view\n" * 5,
        encoding="utf-8",
    )

    preprocess(str(input_path), str(output_path))

    result = pd.read_csv(output_path)

    assert "user_idx" in result.columns
    assert "item_idx" in result.columns
    assert len(result) == 20
    assert set(result["user_idx"]) == {0, 1}
    assert set(result["item_idx"]) == {0, 1}


def test_create_interaction_matrix_unknown_events_default_weight():
    df = pd.DataFrame(
        {
            "user_idx": [0, 0],
            "item_idx": [0, 1],
            "event": ["view", "unknown_event"],
        }
    )

    matrix = create_interaction_matrix(df)

    assert matrix[0, 0] == 1
    assert matrix[0, 1] == 1


def test_run_feature_engineering_writes_files(tmp_path):
    input_path = tmp_path / "events.csv"
    output_dir = tmp_path / "output"

    input_path.write_text(
        "user_idx,item_idx,event\n"
        "0,0,view\n"
        "0,1,transaction\n"
        "1,0,addtocart\n",
        encoding="utf-8",
    )

    run_feature_engineering(str(input_path), str(output_dir))

    assert (output_dir / "user_features.csv").exists()
    assert (output_dir / "interaction_matrix.npz").exists()


def test_precision_recall_ndcg_hit_rate_metrics():
    relevant = {2, 4}
    recommended = [2, 1, 4, 3]

    assert precision_at_k(relevant, recommended, 3) == 2 / 3
    assert recall_at_k(relevant, recommended, 3) == 1
    assert ndcg_at_k(relevant, recommended, 3) > 0
    assert hit_rate_at_k(relevant, recommended, 3) == 1.0
    assert recall_at_k(set(), recommended, 3) == 0.0


def test_popularity_recommender_ranks_items_by_popularity():
    matrix = np.array([[1, 2, 0], [0, 1, 3]], dtype=np.float32)
    recommender = PopularityRecommender(top_k=2)
    recommender.fit(matrix)

    assert recommender.recommend(user_id=0, top_k=2) == [2, 1]


def test_model_factory_reports_unknown_model():
    available_models = ModelFactory.list_models()

    with pytest.raises(ValueError, match="Model 'unknown' not found"):
        ModelFactory.create("unknown")

    assert isinstance(available_models, list)


def test_set_seeds_reproducible():
    set_seeds(123)
    first_random = random.random()
    first_numpy = np.random.rand()
    first_torch = torch.rand(1).item()

    set_seeds(123)
    assert first_random == random.random()
    assert first_numpy == np.random.rand()
    assert first_torch == torch.rand(1).item()


def test_prepare_data_generates_positive_and_negative_samples():
    matrix = sp.csr_matrix(
        (
            [1.0, 1.0],
            ([0, 1], [0, 1]),
        ),
        shape=(2, 3),
        dtype=np.float32,
    )

    users, items, labels = prepare_data(matrix, negative_ratio=1)

    assert users.shape[0] == 4
    assert items.shape[0] == 4
    assert labels.shape[0] == 4
    assert int(labels.sum()) == 2


def test_train_saves_metrics_file(tmp_path, monkeypatch):
    matrix = sp.csr_matrix(
        (
            [1.0, 1.0],
            ([0, 1], [0, 1]),
        ),
        shape=(2, 2),
        dtype=np.float32,
    )
    matrix_path = tmp_path / "interaction_matrix.npz"
    sp.save_npz(matrix_path, matrix)

    output_model = tmp_path / "models" / "recommender.pt"
    metrics_path = tmp_path / "metrics" / "train_metrics.json"

    monkeypatch.setattr(train_module.settings, "num_epochs", 1)
    monkeypatch.setattr(train_module.settings, "batch_size", 2)
    monkeypatch.setattr(train_module.settings, "learning_rate", 0.01)
    monkeypatch.setattr(train_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train_module.mlflow, "set_tracking_uri", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_module.mlflow, "set_experiment", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_module.mlflow, "log_params", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_module.mlflow, "log_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_module.mlflow, "set_tag", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_module.mlflow, "log_artifact", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_module.mlflow, "log_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_module.torch, "save", lambda *args, **kwargs: None)

    class DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(train_module.mlflow, "start_run", lambda *args, **kwargs: DummyRun())
    monkeypatch.setattr(train_module.mlflow.pytorch, "log_model", lambda model, name, serialization_format: SimpleNamespace(model_uri="logged://model"))

    train_module.train(
        matrix_path=str(matrix_path),
        output_path=str(output_model),
        metrics_path=str(metrics_path),
    )

    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["epochs_trained"] == 1
    assert metrics["best_loss"] >= 0


def test_evaluate_writes_metrics_json(tmp_path, monkeypatch):
    events_path = tmp_path / "events_processed.csv"
    model_path = tmp_path / "recommender.pt"
    metrics_path = tmp_path / "metrics" / "eval_metrics.json"

    pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1],
            "item_idx": [0, 1, 0, 1],
            "event": ["view", "transaction", "view", "transaction"],
        }
    ).to_csv(events_path, index=False)

    model = MLPRecommender(n_users=2, n_items=2)
    torch.save(model.state_dict(), model_path)

    monkeypatch.setattr(train_module.torch.cuda, "is_available", lambda: False)

    def deterministic_choice(a, size, replace):
        return np.asarray(a)[:size]

    monkeypatch.setattr(np.random, "choice", deterministic_choice)

    evaluate(
        model_path=str(model_path),
        events_path=str(events_path),
        metrics_path=str(metrics_path),
        k=2,
        n_users=2,
    )

    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "precision@2" in metrics
    assert "recall@2" in metrics
    assert "ndcg@2" in metrics
    assert "hit_rate@2" in metrics
