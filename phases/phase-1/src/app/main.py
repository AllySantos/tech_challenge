from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware

from app.api import health, predict
from app.middleware.logger import structlog_middleware
from app.utils.logger import setup_structlog
from app.utils.machine_learning import load_machine_learning_model
from app.utils.project import get_project_info

setup_structlog()

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = load_machine_learning_model()

    yield {"model": model}

    logger.info("app_shutdown_clearing_machine_learning_model")
    del model


project = get_project_info()
app = FastAPI(title=project.name, version=project.version, lifespan=lifespan)

app.add_middleware(BaseHTTPMiddleware, dispatch=structlog_middleware)

app.include_router(health.router)
app.include_router(predict.router)
