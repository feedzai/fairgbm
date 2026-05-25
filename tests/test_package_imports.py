"""Unit tests for fairgbm package imports and re-exports."""


def test_lightgbm_reexports():
    """All core LightGBM classes are importable from fairgbm."""
    from fairgbm import Dataset, Booster, LGBMClassifier, LGBMRegressor, LGBMRanker, cv

    import lightgbm

    assert Dataset is lightgbm.Dataset
    assert Booster is lightgbm.Booster
    assert LGBMClassifier is lightgbm.LGBMClassifier
    assert LGBMRegressor is lightgbm.LGBMRegressor
    assert LGBMRanker is lightgbm.LGBMRanker
    assert cv is lightgbm.cv


def test_fairgbm_classifier_importable():
    """FairGBMClassifier is importable from top-level."""
    from fairgbm import FairGBMClassifier
    from fairgbm.sklearn import FairGBMClassifier as DirectImport

    assert FairGBMClassifier is DirectImport


def test_train_importable():
    """train function is importable from top-level."""
    from fairgbm import train
    from fairgbm.training import train as DirectTrain

    assert train is DirectTrain


def test_version_set():
    """__version__ is a non-empty string."""
    import fairgbm

    assert hasattr(fairgbm, "__version__")
    assert isinstance(fairgbm.__version__, str)
    assert len(fairgbm.__version__) > 0


def test_native_extension_flag():
    """NATIVE_EXTENSION_AVAILABLE flag exists and is boolean."""
    import fairgbm

    assert hasattr(fairgbm, "NATIVE_EXTENSION_AVAILABLE")
    assert isinstance(fairgbm.NATIVE_EXTENSION_AVAILABLE, bool)


def test_all_defined():
    """__all__ is defined and contains expected names."""
    import fairgbm

    assert hasattr(fairgbm, "__all__")
    expected = {
        "Dataset",
        "Booster",
        "LGBMClassifier",
        "LGBMRegressor",
        "LGBMRanker",
        "cv",
        "FairGBMClassifier",
        "train",
        "__version__",
        "NATIVE_EXTENSION_AVAILABLE",
    }
    assert expected.issubset(set(fairgbm.__all__))
