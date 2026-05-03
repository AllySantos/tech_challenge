import argparse
import io
import json
import os
import tarfile
import time
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import boto3
import joblib
import numpy as np
import pandas as pd
import structlog
import torch
import torch.nn as nn
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from ml.enums.dataset_type import DatasetType
from ml.pipeline.builder import PipelineBuilder
from ml.services.dataframe_service import DataFrameService
from ml.services.mlflow_service import MLFlowService
from ml.services.preprocessing_service import PreprocessingService
from ml.utils.architecture import ChurnMLP, EarlyStopping
from ml.utils.feature_identifier import FeatureIdentifier
from ml.utils.loaders import make_loader

logger = structlog.get_logger()
load_dotenv()


SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
IN_SAGEMAKER_TRAINING = os.path.exists("/opt/ml/code")
SAGEMAKER_BUCKET = os.getenv("SAGEMAKER_BUCKET")
MODEL_S3_KEY = os.getenv("MODEL_S3_KEY", "models/model.pth")
MODEL_TAR_S3_KEY = os.getenv("MODEL_TAR_S3_KEY", "models/model.tar.gz")
SAGEMAKER_EXPERIMENT_NAME = os.getenv("SAGEMAKER_EXPERIMENT_NAME", "churn_prediction")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARTIFACTS_DIR = SM_MODEL_DIR if IN_SAGEMAKER_TRAINING else os.path.join(REPO_ROOT, "models")


def get_sagemaker_hyperparameters():
    hp_path = "/opt/ml/input/config/hyperparameters.json"
    if os.path.exists(hp_path):
        with open(hp_path, "r") as f:
            raw_hps = json.load(f)
            return {
                "epochs": int(raw_hps.get("epochs", os.getenv("EPOCHS", 100))),
                "batch_size": int(raw_hps.get("batch_size", os.getenv("BATCH_SIZE", 64))),
                "weight_decay": float(raw_hps.get("weight_decay", os.getenv("WEIGHT_DECAY", 1e-4))),
                "learning_rate": float(
                    raw_hps.get("learning_rate", os.getenv("LEARNING_RATE", 1e-3))
                ),
                "patience": int(raw_hps.get("patience", os.getenv("PATIENCE", 10))),
                "dropout": float(raw_hps.get("dropout", os.getenv("DROPOUT", 0.3))),
            }
    return {
        "epochs": int(os.getenv("EPOCHS", 100)),
        "batch_size": int(os.getenv("BATCH_SIZE", 64)),
        "weight_decay": float(os.getenv("WEIGHT_DECAY", 1e-4)),
        "learning_rate": float(os.getenv("LEARNING_RATE", 1e-3)),
        "patience": int(os.getenv("PATIENCE", 10)),
        "dropout": float(os.getenv("DROPOUT", 0.3)),
    }


HPARAMS = get_sagemaker_hyperparameters()

logger.info(
    "training_config",
    device=str(DEVICE),
    artifacts_dir=ARTIFACTS_DIR,
    hparams=HPARAMS,
)

df_service = DataFrameService()
pipeline_builder = PipelineBuilder()
feature_identifier = FeatureIdentifier()
preprocessing_service = PreprocessingService(
    pipeline_builder=pipeline_builder, feature_identifier=feature_identifier
)
mlflow_service = MLFlowService(experiment_name=SAGEMAKER_EXPERIMENT_NAME)


def get_sagemaker_bucket_name() -> str:
    if SAGEMAKER_BUCKET:
        return SAGEMAKER_BUCKET
    sts = boto3.client("sts")
    account_id = sts.get_caller_identity()["Account"]
    app_name = os.getenv("APP_NAME", "churn-prediction")
    return f"{app_name}-sagemaker-{account_id}"


def upload_to_s3(local_path: str, s3_key: str) -> None:
    bucket = get_sagemaker_bucket_name()
    s3_client = boto3.client("s3")
    s3_client.upload_file(local_path, bucket, s3_key)
    logger.info("model_uploaded_to_s3", bucket=bucket, key=s3_key, local_path=local_path)


def get_data() -> pd.DataFrame:
    bucket = get_sagemaker_bucket_name()
    s3_key = "data/raw/Telco-Customer-Churn.csv"
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    s3 = boto3.client("s3")

    try:
        obj = s3.get_object(Bucket=bucket, Key=s3_key)
        df = pd.read_csv(obj["Body"])
        logger.info("data_loaded_from_s3", bucket=bucket, key=s3_key)
    except ClientError:
        logger.info("data_not_in_s3_downloading", url=url)
        _default_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        local_data_path = os.getenv("PREPROCESSING_FILE_PATH", _default_path)

        if os.path.exists(local_data_path):
            df = df_service.load_dataframe(local_data_path)
            logger.info("data_loaded_from_local", path=local_data_path)
        else:
            df = pd.read_csv(url)
            logger.info("data_downloaded_from_url", url=url)

        try:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            s3.put_object(Bucket=bucket, Key=s3_key, Body=csv_buffer.getvalue())
        except Exception as e:
            logger.warning("data_upload_failed", error=str(e))

    return df


def preprocessing() -> tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
]:
    logger.info("preprocessing_start")

    df = get_data()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn", "customerID"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    X_train_proc = preprocessing_service.run_pipeline(X_train, type=DatasetType.TRAIN)
    X_val_proc = preprocessing_service.run_pipeline(X_val, type=DatasetType.VALIDATION)
    X_test_proc = preprocessing_service.run_pipeline(X_test, type=DatasetType.TEST)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    pipeline_path = os.path.join(ARTIFACTS_DIR, "pipeline.pkl")
    joblib.dump(preprocessing_service.pipeline, pipeline_path)
    logger.info("preprocessing_complete", pipeline_path=pipeline_path)

    return X_train_proc, y_train, X_val_proc, y_val, X_test_proc, y_test


def train_model(X_train, y_train, X_val, y_val):
    logger.info("model_training_start")

    num_cols = X_train.shape[1]
    model = ChurnMLP(input_dim=num_cols, dropout=HPARAMS["dropout"]).to(DEVICE)

    train_loader = make_loader(X_train, y_train, shuffle=True, batch_size=HPARAMS["batch_size"])
    validation_loader = make_loader(X_val, y_val, shuffle=False, batch_size=HPARAMS["batch_size"])

    positive_weight_val = (y_train == 0).sum() / (y_train == 1).sum()
    pos_weight = torch.tensor([positive_weight_val], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        model.parameters(), weight_decay=HPARAMS["weight_decay"], lr=HPARAMS["learning_rate"]
    )

    early_stopping = EarlyStopping(patience=HPARAMS["patience"])
    best_state = None

    mlflow_service.start_run(run_name="train_model")
    mlflow_service.log_params(HPARAMS)

    start_time = time.perf_counter()

    for epoch in range(1, HPARAMS["epochs"] + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()

            pred = model(xb).view(-1)
            yb = yb.view(-1).float()

            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)

        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, yb in validation_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)

                pred = model(xb).view(-1)
                yb = yb.view(-1).float()

                loss = criterion(pred, yb)
                val_losses.append(loss.item())

        val_loss = np.mean(val_losses)

        mlflow_service.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

        logger.info(
            "epoch_completed",
            epoch=epoch,
            train_loss=round(train_loss, 4),
            val_loss=round(val_loss, 4),
        )

        if early_stopping.step(val_loss, model):
            logger.info("early_stopping_triggered", epoch=epoch)
            if early_stopping.best_model is not None:
                best_state = {k: v.clone() for k, v in early_stopping.best_model.items()}
            break
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    mlflow_service.log_pytorch_model(model, name="final_model", export_model=False)

    duration = time.perf_counter() - start_time
    logger.info("model_training_complete", duration_s=round(duration, 2))

    return model


def save_model(model):
    logger.info("model_saving_start")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    model_path = os.path.join(ARTIFACTS_DIR, "model.pth")
    torch.save(model.state_dict(), model_path)
    logger.info("model_saved", path=model_path)

    model_tar_path = os.path.join(ARTIFACTS_DIR, "model.tar.gz")
    with tarfile.open(model_tar_path, "w:gz") as tar:
        tar.add(model_path, arcname="model.pth")
        pipeline_path = os.path.join(ARTIFACTS_DIR, "pipeline.pkl")
        if os.path.exists(pipeline_path):
            tar.add(pipeline_path, arcname="pipeline.pkl")

    try:
        upload_to_s3(model_path, MODEL_S3_KEY)
        upload_to_s3(model_tar_path, MODEL_TAR_S3_KEY)
    except Exception as e:
        logger.error("model_upload_failed", error=str(e))
        if IN_SAGEMAKER_TRAINING:
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", default="train")

    args, _ = parser.parse_known_args()

    if args.mode == "train":
        logger.info("training_mode_initiated")

        X_train, y_train, X_val, y_val, X_test, y_test = preprocessing()
        model = train_model(X_train, y_train, X_val, y_val)
        save_model(model)

        mlflow_service.end_run()
        logger.info("training_pipeline_complete")


if __name__ == "__main__":
    main()
