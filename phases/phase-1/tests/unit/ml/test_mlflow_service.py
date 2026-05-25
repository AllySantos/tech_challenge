import mlflow
import pytest


@pytest.fixture(autouse=True)
def clean_mlflow_env(tmp_path, monkeypatch):
    # 1. Clear MLflow global state
    mlflow.set_tracking_uri(None)
    if mlflow.active_run():
        mlflow.end_run()

    # 2. Hard-set local tracking to prevent SageMaker plugin from taking over
    local_db = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", local_db)

    # 3. Explicitly wipe all AWS/SageMaker tracking variables
    # Setting them to empty strings or deleting them forces the plugin to stay dormant
    keys = [
        "MLFLOW_TRACKING_ARN",
        "MLFLOW_TRACKING_SERVER_ARN",
        "MLFLOW_TRACKING_SERVER_NAME",
        "MLFLOW_TRACKING_SERVER_URL",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)

    # 4. Neutralize AWS Credentials
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    mlflow.set_tracking_uri(local_db)
    yield local_db
