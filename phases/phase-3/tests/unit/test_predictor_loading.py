import pytest

from src.inference.predictor import load_predictor
from src.models.registry import promote, write_metadata


def test_load_predictor_rejects_an_unknown_backend(tmp_path):
    with pytest.raises(ValueError, match="desconhecido"):
        load_predictor("tensorrt", tmp_path)


def test_load_predictor_reports_an_empty_registry(tmp_path, monkeypatch):
    from src.configs import settings as settings_module

    monkeypatch.setattr(settings_module.settings, "models_dir", str(tmp_path / "empty"))

    with pytest.raises(FileNotFoundError, match="Nenhuma versão de modelo"):
        load_predictor("onnx")


def test_load_predictor_reports_a_missing_artifact(tmp_path):
    root = tmp_path / "models"
    version_dir = root / "20260101T000000Z"
    version_dir.mkdir(parents=True)
    write_metadata(version_dir, {"version": version_dir.name, "labels": ["normal"]})
    promote(version_dir, root)

    with pytest.raises(FileNotFoundError, match="Artefato ausente"):
        load_predictor("onnx-pruned", version_dir)
