import json

import pytest

from src.models.registry import (
    METADATA_FILENAME,
    list_versions,
    new_version,
    promote,
    read_metadata,
    resolve_current,
    write_metadata,
)


@pytest.fixture()
def root(tmp_path):
    return tmp_path / "models"


def _make_version(root, name, metadata=None):
    version_dir = root / name
    version_dir.mkdir(parents=True)
    write_metadata(version_dir, metadata or {"version": name, "labels": ["normal"]})
    return version_dir


def test_new_version_creates_timestamped_directory(root):
    version_dir = new_version(root)

    assert version_dir.exists()
    assert version_dir.parent == root
    assert version_dir.name.endswith("Z")


def test_promote_writes_pointer_and_resolves_back(root):
    version_dir = _make_version(root, "20260101T000000Z")

    pointer = promote(version_dir, root)

    assert json.loads(pointer.read_text())["version"] == "20260101T000000Z"
    assert resolve_current(root) == version_dir


def test_resolve_current_returns_none_when_registry_is_empty(root):
    assert resolve_current(root) is None


def test_resolve_current_falls_back_to_latest_version_without_pointer(root):
    _make_version(root, "20260101T000000Z")
    latest = _make_version(root, "20260202T000000Z")

    assert resolve_current(root) == latest


def test_resolve_current_falls_back_when_pointer_is_stale(root):
    existing = _make_version(root, "20260101T000000Z")
    promote(root / "20269999T000000Z", root)

    assert resolve_current(root) == existing


def test_list_versions_ignores_directories_without_metadata(root):
    _make_version(root, "20260101T000000Z")
    (root / "scratch").mkdir()

    assert list_versions(root) == ["20260101T000000Z"]


def test_metadata_round_trips_unicode(root):
    version_dir = _make_version(root, "20260101T000000Z", {"nota": "atenção"})

    assert read_metadata(version_dir)["nota"] == "atenção"
    assert (version_dir / METADATA_FILENAME).exists()
