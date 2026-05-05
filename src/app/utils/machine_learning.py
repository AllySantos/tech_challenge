import io
import os
import time
from urllib.parse import urlparse

import boto3
import structlog
import torch

logger = structlog.get_logger()


def load_machine_learning_model():
    model_path = os.getenv("MODEL_S3_URI")

    if not model_path:
        logger.warning("machine_learning_load_skipped", reason="MODEL_S3_URI_not_set")
        return None

    logger.info("machine_learning_start_load", model_path=model_path)

    try:
        start_time = time.perf_counter()

        parsed_uri = urlparse(model_path)
        bucket_name = parsed_uri.netloc
        key = parsed_uri.path.lstrip("/")

        s3_client = boto3.client("s3")
        response = s3_client.get_object(Bucket=bucket_name, Key=key)

        buffer = io.BytesIO(response["Body"].read())

        model = torch.load(buffer, map_location="cpu")

        process_time = time.perf_counter() - start_time
        logger.info(
            "machine_learning_loaded",
            model_path=str(model_path),
            duration_ms=round(process_time * 1000, 2),
        )
        return model

    except Exception as e:
        logger.exception("machine_learning_error_load", model_path=model_path, error=str(e))
        raise
