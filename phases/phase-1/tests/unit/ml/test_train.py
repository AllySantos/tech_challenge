import mlflow
import pytest

import ml.train as train

REAL_OPEN = open


@pytest.fixture(autouse=True)
def setup_train_env(tmp_path, monkeypatch):
    # 1. Reset MLflow
    mlflow.set_tracking_uri(None)
    local_db = f"sqlite:///{tmp_path}/train.db"

    # 2. Force local environment
    monkeypatch.setenv("MLFLOW_TRACKING_URI", local_db)
    for key in ["MLFLOW_TRACKING_ARN", "MLFLOW_TRACKING_SERVER_ARN"]:
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    # 3. Re-sync the global MLflow state
    mlflow.set_tracking_uri(local_db)

    # 4. Patch the global service instances inside train.py
    # This is vital because they were initialized during 'import ml.train'
    test_mlflow = train.MLFlowService(experiment_name="test-train", tracking_uri=local_db)
    monkeypatch.setattr(train, "mlflow_service", test_mlflow)
    monkeypatch.setattr(train, "ARTIFACTS_DIR", str(tmp_path))

    return local_db
