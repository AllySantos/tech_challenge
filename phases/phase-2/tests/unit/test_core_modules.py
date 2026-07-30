import numpy as np
import pandas as pd
from scipy import sparse

from src.data.preprocess import encode_ids, filter_events, preprocess
from src.evaluation.evaluator import ndcg_at_k, precision_at_k, recall_at_k
from src.features.feature_engineer import create_interaction_matrix, create_user_features
from src.models.base import BaseRecommender
from src.models.factory import ModelFactory


def test_filter_events_keeps_users_and_items_with_enough_interactions():
    df = pd.DataFrame(
        {
            "visitorid": [1, 1, 1, 2, 2, 3, 3, 3],
            "itemid": [10, 11, 10, 20, 20, 30, 30, 31],
            "event": ["view", "view", "view", "view", "view", "view", "view", "view"],
        }
    )

    filtered = filter_events(df, min_interactions=2)

    assert set(filtered["visitorid"]) == {1, 2, 3}
    assert set(filtered["itemid"]) == {10, 20, 30}
    assert len(filtered) == 6


def test_encode_ids_creates_sequential_user_and_item_indexes():
    df = pd.DataFrame(
        {
            "visitorid": [100, 200, 100],
            "itemid": [10, 20, 10],
            "event": ["view", "view", "view"],
        }
    )

    encoded_df, user_encoder, item_encoder = encode_ids(df)

    assert user_encoder == {100: 0, 200: 1}
    assert item_encoder == {10: 0, 20: 1}
    assert encoded_df["user_idx"].tolist() == [0, 1, 0]
    assert encoded_df["item_idx"].tolist() == [0, 1, 0]


def test_preprocess_writes_encoded_csv(tmp_path):
    input_path = tmp_path / "events.csv"
    output_path = tmp_path / "events_processed.csv"

    input_path.write_text(
        "visitorid,itemid,event\n"
        "1,10,view\n"
        "1,11,view\n"
        "2,10,view\n"
        "2,10,view\n"
        "3,12,view\n"
        "3,12,view\n",
        encoding="utf-8",
    )

    preprocess(str(input_path), str(output_path))

    result = pd.read_csv(output_path)

    assert "user_idx" in result.columns
    assert "item_idx" in result.columns


def test_create_interaction_matrix_uses_event_weights():
    df = pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1],
            "item_idx": [0, 1, 0, 1],
            "event": ["view", "transaction", "addtocart", "view"],
        }
    )

    matrix = create_interaction_matrix(df)

    assert isinstance(matrix, sparse.csr_matrix)
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == 1
    assert matrix[0, 1] == 5
    assert matrix[1, 0] == 3
    assert matrix[1, 1] == 1


def test_create_user_features_aggregates_event_counts():
    df = pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1, 1],
            "item_idx": [0, 1, 0, 1, 2],
            "event": ["view", "transaction", "addtocart", "view", "transaction"],
        }
    )

    features = create_user_features(df)

    assert list(features.columns) == [
        "user_idx",
        "total_events",
        "unique_items",
        "transactions",
        "add_to_cart",
    ]
    assert features.loc[features["user_idx"] == 0, "total_events"].iloc[0] == 2
    assert features.loc[features["user_idx"] == 0, "transactions"].iloc[0] == 1
    assert features.loc[features["user_idx"] == 1, "add_to_cart"].iloc[0] == 1


def test_evaluator_metrics_return_expected_values():
    relevant = {2, 4}
    recommended = [2, 1, 4, 3]

    assert precision_at_k(relevant, recommended, 3) == 2 / 3
    assert recall_at_k(relevant, recommended, 3) == 1
    assert ndcg_at_k(relevant, recommended, 3) > 0


def test_model_factory_registers_and_instantiates_models():
    ModelFactory._registry.clear()

    @ModelFactory.register("dummy")
    class DummyRecommender(BaseRecommender):
        def __init__(self, multiplier: int = 1) -> None:
            self.multiplier = multiplier

        def fit(self, interaction_matrix: np.ndarray) -> None:
            self.fitted = True

        def recommend(self, user_id: int, top_k: int = 10) -> list[int]:
            return [user_id * self.multiplier]

    instance = ModelFactory.create("dummy", multiplier=2)

    assert "dummy" in ModelFactory.list_models()
    assert isinstance(instance, DummyRecommender)
    assert instance.recommend(3, top_k=1) == [6]
