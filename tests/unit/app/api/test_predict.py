from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.predict import router

app = FastAPI()

app.include_router(router)

client = TestClient(app)


def test_predict_endpoint_success():
    response = client.post("/predict")

    assert response.status_code == 200
    assert response.json() == {}
