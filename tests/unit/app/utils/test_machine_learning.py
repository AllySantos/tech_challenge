import joblib
import pytest
import torch

from app.utils import machine_learning


def _patch_repo_root(monkeypatch, tmp_path):
    original_resolve = machine_learning.Path.resolve
    fake_machine_learning_file = tmp_path / "src" / "app" / "utils" / "machine_learning.py"

    def fake_resolve(self):
        if self.name == "machine_learning.py":
            return fake_machine_learning_file
        return original_resolve(self)

    monkeypatch.setattr(machine_learning.Path, "resolve", fake_resolve)
    return tmp_path


def test_load_machine_learning_model_pth_relative(monkeypatch, tmp_path):
    repo_root = _patch_repo_root(monkeypatch, tmp_path)
    model_dir = repo_root / "models"
    model_dir.mkdir(parents=True)

    data = {"test": "abc"}
    pth_path = model_dir / "test_model.pth"
    torch.save(data, pth_path)

    loaded = machine_learning.load_machine_learning_model("models/test_model.pth")

    assert loaded == data


def test_load_machine_learning_model_pkl_direct(monkeypatch, tmp_path):
    repo_root = _patch_repo_root(monkeypatch, tmp_path)
    model_dir = repo_root / "models"
    model_dir.mkdir(parents=True)

    data = {"test": "abc"}
    pkl_path = model_dir / "test_model.pkl"
    joblib.dump(data, pkl_path)

    loaded = machine_learning.load_machine_learning_model("models/test_model.pkl")

    assert loaded == data


def test_load_machine_learning_model_pkl_fallback_to_pth(monkeypatch, tmp_path):
    repo_root = _patch_repo_root(monkeypatch, tmp_path)
    model_dir = repo_root / "models"
    model_dir.mkdir(parents=True)

    data = {"test": True}
    pth_path = model_dir / "fallback_model.pth"
    torch.save(data, pth_path)

    loaded = machine_learning.load_machine_learning_model("models/fallback_model.pkl")

    assert loaded == data


def test_load_machine_learning_model_raises_when_missing(monkeypatch, tmp_path):
    _patch_repo_root(monkeypatch, tmp_path)

    with pytest.raises(FileNotFoundError):
        machine_learning.load_machine_learning_model("models/missing_model.pkl")
