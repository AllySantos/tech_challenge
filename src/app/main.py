from fastapi import FastAPI

from app.api import health, predict

app = FastAPI(title="churn-prediction", version="0.1.0")

app.include_router(health.router)
app.include_router(predict.router)
