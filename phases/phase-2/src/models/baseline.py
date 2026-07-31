import numpy as np

from src.models.base import BaseRecommender
from src.models.factory import ModelFactory


@ModelFactory.register("popularity")
class PopularityRecommender(BaseRecommender):
    """Baseline recommender based on item popularity.

    Recommends the most interacted items globally,
    regardless of user history.
    """

    def __init__(self, top_k: int = 10) -> None:
        self.top_k = top_k
        self._popular_items: list[int] = []

    def fit(self, interaction_matrix: np.ndarray) -> None:
        """Rank items by total interactions across all users."""
        item_counts = interaction_matrix.sum(axis=0)
        self._popular_items = np.argsort(item_counts)[::-1].tolist()

    def recommend(self, user_id: int, top_k: int = 10) -> list[int]:
        """Return the globally most popular items."""
        k = top_k or self.top_k
        return self._popular_items[:k]
