def test_import_src_package():
    import src
    import src.data
    import src.evaluation
    import src.features
    import src.models
    import src.training

    assert src.__name__ == "src"
    assert src.data.__name__ == "src.data"
    assert src.evaluation.__name__ == "src.evaluation"
    assert src.features.__name__ == "src.features"
    assert src.models.__name__ == "src.models"
    assert src.training.__name__ == "src.training"
