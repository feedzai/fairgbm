# coding: utf-8
import pytest

import lightgbm as lgb

from .utils import load_baf_base, binarize_predictions, evaluate_recall, evaluate_fairness


@pytest.fixture
def lightgbm_params() -> dict:
    return {
        "objective": "cross_entropy",
        "num_iterations": 1000,
    }


@pytest.fixture
def fairgbm_params(lightgbm_params) -> dict:
    params = lightgbm_params.copy()
    params.update({
        "objective": "constrained_cross_entropy",
        "multiplier_learning_rate": 10_000,
        "constraint_type": "FPR",
        "global_constraint_type": "FPR,FNR",
        "global_target_fpr": 0.05,
        "global_target_fnr": 0.70,
    })

    return params


@pytest.mark.parametrize("fairgbm_target_fpr", [
    0.03, 0.05, 0.075, 0.10, 0.2, 0.5,
])
def test_fairgbm_fairness(fairgbm_params, fairgbm_target_fpr):
    # Load BAF-base dataset
    data = load_baf_base()
    X_train, Y_train, S_train = data["train"]
    X_test, Y_test, S_test = data["test"]

    train_data = lgb.Dataset(X_train, label=Y_train, constraint_group=S_train)

    # Set a specific value of target global FPR
    fairgbm_params["global_target_fpr"] = fairgbm_target_fpr

    # Train FairGBM model
    clf = lgb.train(params=fairgbm_params, train_set=train_data)

    # Compute test predictions
    Y_test_pred = clf.predict(X_test)    # NOTE: we don't need sensitive attribute data to compute predictions!

    # Binarize predictions at target FPR
    Y_test_pred_binarized = binarize_predictions(Y_test, Y_test_pred, fpr=fairgbm_target_fpr)

    # Compute Recall and Fairness at target FPR
    recall = evaluate_recall(Y_test, Y_test_pred_binarized)
    fpr_ratio = evaluate_fairness(Y_test, Y_test_pred_binarized, S_test, metric_col="fpr")

    # Assert Recall makes sense, i.e., is somewhat better than random :)
    smallest_acceptable_recall = fairgbm_target_fpr + 0.20      # this is a loose educated guess
    assert recall > smallest_acceptable_recall, (
        f"FairGBM achieved recall {recall:.1%} at {fairgbm_target_fpr:.1%} FPR, expected at least "
        f"{smallest_acceptable_recall:.1%} recall."
    )

    # Assert Fairness makes sense; it should be better than 80% for most cases
    assert (fpr_ratio > 0.70) if (fairgbm_target_fpr > 0.05) else (fpr_ratio > 0.60), (
        f"FairGBM achieved fairness (FPR ratio) of {fpr_ratio:.1%} at {fairgbm_target_fpr:.1%} global FPR, "
        f"this seems too low..."
    )


def test_fairgbm_vs_lightgbm(lightgbm_params, fairgbm_params):
    # TODO:
    # 1. create lgb objects
    # 2. train fairgbm and lightgbm
    # 3. compute predictions on test data
    # 4. compute fairness and accuracy
    # 5. assert FairGBM is strictly better than LightGBM in fairness
    assert True
