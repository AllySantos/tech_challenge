from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware

from app.api import health, predict
from app.middleware.logger import structlog_middleware
from app.utils.logger import setup_structlog

setup_structlog()

app = FastAPI(title="churn-prediction", version="0.1.0")

app.add_middleware(BaseHTTPMiddleware, dispatch=structlog_middleware)

app.include_router(health.router)
app.include_router(predict.router)
