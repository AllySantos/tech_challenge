def test_folder_structure():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent.parent

    
    assert (root / "src").is_dir()
    assert (root / "tests").is_dir()
    assert (root / "pyproject.toml").is_file()
