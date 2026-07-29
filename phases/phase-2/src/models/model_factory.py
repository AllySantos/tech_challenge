from typing import Any

from recsys.models.base import BaseRecommender


class ModelFactory:
    """Factory for creating recommendation models."""

    _registry: dict[str, type[BaseRecommender]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a model class by name."""

        def decorator(model_class: type[BaseRecommender]):
            cls._registry[name] = model_class
            return model_class

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseRecommender:
        """Instantiate a registered model by name."""
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Model '{name}' not found. Available: {available}")
        return cls._registry[name](**kwargs)

    @classmethod
    def list_models(cls) -> list[str]:
        """Return all registered model names."""
        return list(cls._registry.keys())
