from abc import ABC, abstractmethod

import numpy as np


class BaseRecommender(ABC):
    """Abstract base class for all recommendation models."""

    @abstractmethod
    def fit(self, interaction_matrix: np.ndarray) -> None:
        """Train the model on interaction data."""
        ...

    @abstractmethod
    def recommend(self, user_id: int, top_k: int = 10) -> list[int]:
        """Return top-k item recommendations for a user."""
        ...
