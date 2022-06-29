# coding: utf-8
from functools import lru_cache
from pathlib import Path
from typing import Tuple
import logging

import pytest
import numpy as np
import sklearn.datasets
from sklearn.utils import check_random_state
from sklearn.metrics import roc_curve, confusion_matrix


@lru_cache(maxsize=None)
def load_baf_base():
    pd = pytest.importorskip(
        "pandas",
        reason="The `pandas` package must be installed to import BAF-base and run fairness tests.")

    local_root_path = Path(__file__).parent.parent.parent / "examples" / "FairGBM"
    label_col = "fraud_bool"
    sensitive_col = "customer_age"
    # month_col = "month"
    categorical_cols = ["payment_type", "employment_status", "housing_status", "source", "device_os"]
    age_category_threshold = 50

    def split_X_Y_S(path):
        # Read data from path
        data = pd.read_csv(path, header=0, sep="\t", index_col=None)

        # Set categorical columns as such (needed for LightGBM to be able to identify which cols are categorical)
        data = data.astype({col: "category" for col in categorical_cols})

        # Split X, Y, S
        X = data[[col for col in data.columns if col != label_col]]
        Y = data[label_col]
        S = (data[sensitive_col] >= age_category_threshold).astype(int)
        return X, Y, S

    data_paths = {
        "train": local_root_path / "BAF-base.train",
        "test": local_root_path / "BAF-base.test",
    }

    return {key: split_X_Y_S(path) for key, path in data_paths.items()}


@lru_cache(maxsize=None)
def load_compas():
    pd = pytest.importorskip(
        "pandas",
        reason="The `pandas` package must be installed to import COMPAS and run fairness tests.")

    local_root_path = Path(__file__).parent.parent.parent / "examples" / "FairGBM-other"
    target_col_name = "two_year_recid"
    sensitive_col_name = "race_Caucasian"

    def read_data(path) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        data = pd.read_csv(path, header=0, sep="\t", index_col=None)
        Y = data[target_col_name]
        S = data[sensitive_col_name]
        X = data[[col for col in data.columns if col not in {target_col_name, sensitive_col_name}]]
        return X, Y, S

    data_paths = {
        "train": local_root_path / "COMPAS.train",
        "test": local_root_path / "COMPAS.test",
    }

    return {key: read_data(path) for key, path in data_paths.items()}


@lru_cache(maxsize=None)
def load_boston(**kwargs):
    return sklearn.datasets.load_boston(**kwargs)


@lru_cache(maxsize=None)
def load_breast_cancer(**kwargs):
    return sklearn.datasets.load_breast_cancer(**kwargs)


@lru_cache(maxsize=None)
def load_digits(**kwargs):
    return sklearn.datasets.load_digits(**kwargs)


@lru_cache(maxsize=None)
def load_iris(**kwargs):
    return sklearn.datasets.load_iris(**kwargs)


@lru_cache(maxsize=None)
def load_linnerud(**kwargs):
    return sklearn.datasets.load_linnerud(**kwargs)


def make_ranking(n_samples=100, n_features=20, n_informative=5, gmax=2,
                 group=None, random_gs=False, avg_gs=10, random_state=0):
    """Generate a learning-to-rank dataset - feature vectors grouped together with
    integer-valued graded relevance scores. Replace this with a sklearn.datasets function
    if ranking objective becomes supported in sklearn.datasets module.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        Total number of documents (records) in the dataset.
    n_features : int, optional (default=20)
        Total number of features in the dataset.
    n_informative : int, optional (default=5)
        Number of features that are "informative" for ranking, as they are bias + beta * y
        where bias and beta are standard normal variates. If this is greater than n_features, the dataset will have
        n_features features, all will be informative.
    gmax : int, optional (default=2)
        Maximum graded relevance value for creating relevance/target vector. If you set this to 2, for example, all
        documents in a group will have relevance scores of either 0, 1, or 2.
    group : array-like, optional (default=None)
        1-d array or list of group sizes. When `group` is specified, this overrides n_samples, random_gs, and
        avg_gs by simply creating groups with sizes group[0], ..., group[-1].
    random_gs : bool, optional (default=False)
        True will make group sizes ~ Poisson(avg_gs), False will make group sizes == avg_gs.
    avg_gs : int, optional (default=10)
        Average number of documents (records) in each group.
    random_state : int, optional (default=0)
        Random seed.

    Returns
    -------
    X : 2-d np.ndarray of shape = [n_samples (or np.sum(group)), n_features]
        Input feature matrix for ranking objective.
    y : 1-d np.array of shape = [n_samples (or np.sum(group))]
        Integer-graded relevance scores.
    group_ids : 1-d np.array of shape = [n_samples (or np.sum(group))]
        Array of group ids, each value indicates to which group each record belongs.
    """
    rnd_generator = check_random_state(random_state)

    y_vec, group_id_vec = np.empty((0,), dtype=int), np.empty((0,), dtype=int)
    gid = 0

    # build target, group ID vectors.
    relvalues = range(gmax + 1)

    # build y/target and group-id vectors with user-specified group sizes.
    if group is not None and hasattr(group, '__len__'):
        n_samples = np.sum(group)

        for i, gsize in enumerate(group):
            y_vec = np.concatenate((y_vec, rnd_generator.choice(relvalues, size=gsize, replace=True)))
            group_id_vec = np.concatenate((group_id_vec, [i] * gsize))

    # build y/target and group-id vectors according to n_samples, avg_gs, and random_gs.
    else:
        while len(y_vec) < n_samples:
            gsize = avg_gs if not random_gs else rnd_generator.poisson(avg_gs)

            # groups should contain > 1 element for pairwise learning objective.
            if gsize < 1:
                continue

            y_vec = np.append(y_vec, rnd_generator.choice(relvalues, size=gsize, replace=True))
            group_id_vec = np.append(group_id_vec, [gid] * gsize)
            gid += 1

        y_vec, group_id_vec = y_vec[:n_samples], group_id_vec[:n_samples]

    # build feature data, X. Transform first few into informative features.
    n_informative = max(min(n_features, n_informative), 0)
    X = rnd_generator.uniform(size=(n_samples, n_features))

    for j in range(n_informative):
        bias, coef = rnd_generator.normal(size=2)
        X[:, j] = bias + coef * y_vec

    return X, y_vec, group_id_vec


def threshold_at_target(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        tpr: float = None,
        fpr: float = None,
    ) -> float:
    """Computes the threshold at the given target.
    Does not untie rows, may miss target in the presence of ties.
    Uses scikit-learn to compute ROC curve.
    """
    fpr_vals, tpr_vals, thresholds = roc_curve(y_true, y_pred, pos_label=1)

    # Find threshold that hits **at least** the target TPR
    if tpr:
        threshold_idx = np.argmax(np.argwhere(tpr_vals < tpr)) + 1

    # Find threshold that hits **at most** the target FPR
    elif fpr:
        threshold_idx = np.argmax(np.argwhere(fpr_vals <= fpr))

    threshold = thresholds[threshold_idx]

    # Sanity check!
    y_pred_binarized = (y_pred >= threshold)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binarized).ravel()
    actual_tpr = tp / (tp + fn)
    actual_fpr = fp / (fp + tn)
    if (tpr and actual_tpr < tpr) or (fpr and actual_fpr > fpr):
        logging.error(f"Missed target metric: TPR={actual_tpr:.1%}, FPR={actual_fpr:.1%};")

    return threshold


def binarize_predictions(y_true, y_pred, tpr: float = None, fpr: float = None):
    threshold = threshold_at_target(y_true, y_pred, tpr=tpr, fpr=fpr)
    return (y_pred >= threshold).astype(int)


def evaluate_recall(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall = tp / (tp + fn)
    return recall


def evaluate_fairness(y_true, y_pred, sensitive_col, metric_col="fpr"):
    pd = pytest.importorskip("pandas")
    aequitas_group = pytest.importorskip("aequitas.group")

    g = aequitas_group.Group()
    aequitas_df = pd.DataFrame({
        "label": y_true,
        "prediction": y_pred,
        "sensitive": sensitive_col.astype(str)
    })

    aequitas_results, _ = g.get_crosstabs(
        aequitas_df,
        label_col="label",
        score_col="prediction",
        attr_cols=["sensitive"])

    perf_metrics = aequitas_results[metric_col]
    return perf_metrics.min() / perf_metrics.max()
