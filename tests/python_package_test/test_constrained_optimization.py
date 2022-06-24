# coding: utf-8
import pytest

import lightgbm as lgb

from .utils import load_compas, load_california_housing_classification


@pytest.fixture
def lightgbm_hyperparams() -> dict:
    return {
        # TODO
    }

@pytest.fixture
def fairgbm_hyperparams(lightgbm_hyperparams) -> dict:
    hyperparams = lightgbm_hyperparams.copy()
    hyperparams.update({
        # TODO
    })

    return hyperparams

def assert_fairness_fairgbm_vs_lightgbm(
        fairgbm_params,
        lightgbm_params,
        X_train, y_train, s_train,
        X_test, y_test, s_test
    ):
    # TODO:
    # 1. create lgb objects
    # 2. train fairgbm and lightgbm
    # 3. compute predictions on test data
    # 4. compute fairness and accuracy
    # 5. assert FairGBM is strictly better than LightGBM in fairness
    assert True


def test_fairgbm_vs_lightgbm_on_compas():
    data = load_compas()

    def load_lgb_data(data):
        X, Y, S = data
        return lgb.Dataset(X, label=Y, constraint_group=S)

    lgb_train = load_lgb_data(data["train"])
    lgb_test = load_lgb_data(data["test"])


def test_fairgbm_vs_lightgbm_on_california_housing():
    X, Y, S = load_california_housing_classification()

    # TODO: train test split and train fairgbm etc.
    assert True
