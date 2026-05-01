import time
from pathlib import Path

import joblib
import structlog
import torch

logger = structlog.get_logger()


def load_machine_learning_model(model_path: str):
    repo_root = Path(__file__).resolve().parents[3]
    path = Path(model_path)
    if not path.is_absolute():
        path = repo_root / path

    if not path.exists() and path.suffix == ".pkl":
        alt_path = path.with_suffix(".pth")
        if alt_path.exists():
            path = alt_path

    logger.info("machine_learning_start_load", {"model_path": str(path)})
    try:
        start_time = time.perf_counter()
        if path.suffix in {".pth", ".pt"}:
            model = torch.load(path, map_location="cpu")
        else:
            model = joblib.load(path)
        process_time = time.perf_counter() - start_time

        logger.info(
            "machine_learning_loaded",
            model_path=str(path),
            duration_ms=round(process_time * 1000, 2),
        )
        return model
    except Exception:
        logger.exception("machine_learning_error_load", {"model_path": str(path)})
        raise
