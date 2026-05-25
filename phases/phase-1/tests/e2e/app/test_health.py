from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_e2e_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True}
