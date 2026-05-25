from unittest.mock import mock_open, patch

import pytest

from app.utils.project import get_project_info

MOCK_TOML_CONTENT = b"""
[project]
name = "sample-project"
version = "2.0.0"
"""


@patch("builtins.open", new_callable=mock_open, read_data=MOCK_TOML_CONTENT)
def test_get_project_info_success(mock_file):
    project = get_project_info()

    assert project.name == "sample-project"
    assert project.version == "2.0.0"

    mock_file.assert_called_once_with("pyproject.toml", "rb")


@patch("builtins.open", side_effect=FileNotFoundError)
def test_get_project_info_file_not_found(mock_file):
    with pytest.raises(FileNotFoundError):
        get_project_info()
