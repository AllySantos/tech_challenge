"""FastAPI inference API for recsys-retailrocket."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.configs.settings import settings
from src.models.train import MLPRecommender

logger = logging.getLogger(__name__)

model: MLPRecommender | None = None
n_users: int = 0
n_items: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """Load model on startup."""
    global model, n_users, n_items

    model_path = Path(settings.models_path) / "recommender.pt"
    if not model_path.exists():
        logger.warning("Model not found at %s", model_path)
    else:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        n_users = checkpoint["user_embedding.weight"].shape[0]
        n_items = checkpoint["item_embedding.weight"].shape[0]
        model = MLPRecommender(n_users, n_items)
        model.load_state_dict(checkpoint)
        model.eval()
        logger.info("Model loaded: %d users, %d items", n_users, n_items)

    yield


app = FastAPI(
    title="recsys-retailrocket",
    description="Product recommendation API using MLP embeddings",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""

    user_id: int
    recommendations: list[int]
    top_k: int


@app.get("/")
def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/recommend", response_model=RecommendationResponse)
def recommend(user_id: int, top_k: int = 10) -> RecommendationResponse:
    """Return top-k recommendations for a user."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if user_id < 0 or user_id >= n_users:
        raise HTTPException(
            status_code=404,
            detail=f"User {user_id} not found. Valid range: 0-{n_users - 1}",
        )

    with torch.no_grad():
        all_items = torch.arange(n_items, dtype=torch.long)
        user_tensor = torch.tensor([user_id] * n_items, dtype=torch.long)
        scores = model(user_tensor, all_items).numpy()

    recommended = np.argsort(scores)[::-1][:top_k].tolist()

    return RecommendationResponse(
        user_id=user_id,
        recommendations=recommended,
        top_k=top_k,
    )
