# coding: utf-8
import logging

import pytest

import lightgbm as lgb

from .utils import load_baf_base, binarize_predictions, evaluate_recall, evaluate_fairness


@pytest.fixture
def lightgbm_params() -> dict:
    return {
        "objective": "cross_entropy",
        "seed": 3402,
        "num_iterations": 1000,
    }


@pytest.fixture(params=[0.05, 0.075, 0.10, 0.20, 0.50])
def target_fpr(request):
    return request.param


@pytest.fixture
def target_fnr(target_fpr):
    return 1 - target_fpr - 0.10    # loose target for FNR


@pytest.fixture(params=["FPR"])
def target_fairness_metric(request):
    return request.param


@pytest.fixture
def fairgbm_params(lightgbm_params, target_fpr, target_fnr, target_fairness_metric) -> dict:
    params = lightgbm_params.copy()
    params.update({
        "objective": "constrained_cross_entropy",
        "multiplier_learning_rate": 10_000,
        "constraint_type": target_fairness_metric,
        "global_constraint_type": "FPR,FNR",
        "global_target_fpr": target_fpr,
        "global_target_fnr": target_fnr,
    })

    return params


def test_fairgbm_fairness(fairgbm_params, target_fpr, target_fnr, target_fairness_metric):
    # Load BAF-base dataset
    data = load_baf_base()
    X_train, Y_train, S_train = data["train"]
    X_test, Y_test, S_test = data["test"]

    train_data = lgb.Dataset(X_train, label=Y_train, constraint_group=S_train)

    # Train FairGBM model
    clf = lgb.train(params=fairgbm_params, train_set=train_data)

    # Compute test predictions
    Y_test_pred = clf.predict(X_test)    # NOTE: we don't need sensitive attribute data to compute predictions!

    # Binarize predictions at target FPR
    Y_test_pred_binarized = binarize_predictions(Y_test, Y_test_pred, fpr=target_fpr)

    # Compute Recall and Fairness at target FPR
    recall = evaluate_recall(Y_test, Y_test_pred_binarized)
    fairness = evaluate_fairness(Y_test, Y_test_pred_binarized, S_test, metric_col=target_fairness_metric)

    # Assert Recall makes sense, i.e., is somewhat better than random :)
    target_recall = 1 - target_fnr
    assert recall > target_recall, (
        f"FairGBM achieved recall {recall:.1%} at {target_fpr:.1%} FPR, expected at least "
        f"{target_recall:.1%} recall."
    )

    # Assert Fairness makes sense
    target_metric_val = target_fpr if target_fairness_metric == "FPR" else target_fnr
    assert (fairness > 0.70) if (target_metric_val >= 0.10) else (fairness > 0.60), (
        f"FairGBM achieved fairness ({target_fairness_metric} ratio) of {fairness:.1%} at {target_fpr:.1%} global FPR, "
        f"this seems too low..."
    )


def test_fairgbm_vs_lightgbm(lightgbm_params, fairgbm_params, target_fpr, target_fairness_metric):
    # Load BAF-base dataset
    data = load_baf_base()
    X_train, Y_train, S_train = data["train"]
    X_test, Y_test, S_test = data["test"]

    train_data_without_sensitive = lgb.Dataset(X_train, label=Y_train)
    train_data_with_sensitive = lgb.Dataset(X_train, label=Y_train, constraint_group=S_train)

    # Train FairGBM model
    fair_clf = lgb.train(params=fairgbm_params, train_set=train_data_with_sensitive)
    vanilla_clf = lgb.train(params=lightgbm_params, train_set=train_data_without_sensitive)

    # Compute Recall and Fairness at target FPR
    def evaluate_perf_fair(clf):
        Y_pred = clf.predict(X_test)
        Y_pred_bin = binarize_predictions(Y_test, Y_pred, fpr=target_fpr)

        recall = evaluate_recall(Y_test, Y_pred_bin)
        fpr_ratio = evaluate_fairness(Y_test, Y_pred_bin, S_test, metric_col=target_fairness_metric)
        return recall, fpr_ratio

    fairgbm_recall, fairgbm_fairness = evaluate_perf_fair(fair_clf)
    lightgbm_recall, lightgbm_fairness = evaluate_perf_fair(vanilla_clf)

    logging.info(
        f"Target FPR={target_fpr:.1%}; \n"
        f"FairGBM perf/fair: {fairgbm_recall:.1%} / {fairgbm_fairness:.1%}; \n"
        f"LightGBM perf/fair: {lightgbm_recall:.1%} / {lightgbm_fairness:.1%}; \n"
    )

    # Assert FairGBM's performance is within some margin of LightGBM's performance
    assert lightgbm_recall > fairgbm_recall > lightgbm_recall * 0.8, (
        f"FairGBM achieved recall={fairgbm_recall:.1%}; LightGBM achieved recall={lightgbm_recall:.1%};"
    )

    # Assert FairGBM's fairness is strictly better then LightGBM's fairness
    # (very low target FPR values may lead to high stochasticity and in some cases FairGBM may be worse than LightGBM)
    assert fairgbm_fairness > lightgbm_fairness, (
        f"FairGBM achieved fairness ({target_fairness_metric} ratio) of {fairgbm_fairness:.1%} at {target_fpr:.1%}"
        f" global FPR, while LightGBM achieved fairness {lightgbm_fairness:.1%};"
    )
