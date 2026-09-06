from pathlib import Path

from src.configs.settings import DEFAULT_PROJECT_ROOT, settings


def test_project_root_points_at_the_directory_holding_src():
    assert (DEFAULT_PROJECT_ROOT / "src" / "configs" / "settings.py").exists()


def test_relative_paths_are_anchored_to_the_project_root(monkeypatch):
    monkeypatch.setattr(settings, "project_root", "/opt/project")
    monkeypatch.setattr(settings, "models_dir", "models")

    assert settings.models_root == Path("/opt/project/models")


def test_absolute_paths_are_left_untouched(monkeypatch):
    monkeypatch.setattr(settings, "project_root", "/opt/project")
    monkeypatch.setattr(settings, "models_dir", "/mnt/shared/models")

    assert settings.models_root == Path("/mnt/shared/models")


def test_every_directory_property_is_absolute():
    for path in (
        settings.raw_dir,
        settings.processed_dir,
        settings.models_root,
        settings.metrics_root,
        settings.reports_root,
    ):
        assert path.is_absolute()
