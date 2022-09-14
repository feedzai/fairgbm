"""
Utility functions to keep the example notebooks uncluttered with boilerplate.
"""
import re
from collections import OrderedDict
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


DATA_DIR = Path(__file__).parent / "data"
UCI_ADULT_TARGET_COL = "target"
UCI_ADULT_SENSITIVE_COL = "sex"


def load_uci_adult() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Downloads and pre-processes the UCI Adult dataset.
    
    Returns
    -------
    train_set, test_set : tuple[pd.DataFrame, pd.DataFrame]
        The pre-processed train and test datasets.
    """
    try:
        import wget
    except ModuleNotFoundError as err:
        print(f"Downloading this dataset requires the `wget` python package; got \"{err}\"")

    # URLs for downloading dataset
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
    train_url = base_url + "adult.data"
    test_url = base_url + "adult.test"
    names_url = base_url + "adult.names"
    
    # Make local data directory
    DATA_DIR.mkdir(exist_ok=True)

    # Download data
    train_path = wget.download(train_url, str(DATA_DIR))
    test_path = wget.download(test_url, str(DATA_DIR))
    names_path = wget.download(names_url, str(DATA_DIR))
    
    return (
        _preprocess_uci_adult(train_path, names_path),
        _preprocess_uci_adult(test_path, names_path, skiprows=1),
    )


def _preprocess_uci_adult(data_path, names_path, **read_kwargs) -> pd.DataFrame:

    # Load column names
    column_map = OrderedDict()
    line_regexp = re.compile(r"^([-\w]+): (.*)[.]$")

    with open(names_path, "r") as f_in:
        lines = f_in.readlines()
        for l in lines:
            match = line_regexp.match(l)
            if not match: continue

            col_name = match.group(1)
            col_values = match.group(2).split(", ")
            if len(col_values) == 1:
                col_values = col_values[0]

            column_map[col_name] = col_values

    # Last column is the target
    column_map[UCI_ADULT_TARGET_COL] = ["<=50K", ">50K"]

    # Load data
    data = pd.read_csv(
        data_path,
        header=None,
        names=list(column_map.keys()),
        index_col=None,
        **read_kwargs)

    # Set correct dtypes
    data = data.astype({
        col_name: (
            float if col_value == "continuous" else "category"
        ) for col_name, col_value in column_map.items()
    })
    
    # Strip whitespace from categorical values
    for col in data.columns:
        if data[col].dtype == "category":
            data[col] = data[col].map(lambda val: val.strip())            

    # Convert label to numeric
    data[UCI_ADULT_TARGET_COL] = pd.Series(
        data=[
            0 if "<=50K" in val.strip() else 1 
            for val in data[UCI_ADULT_TARGET_COL]
        ],
        dtype=float)
    return data


def split_X_Y_S_uci_adult(data) -> Tuple[pd.DataFrame, pd.Series]:
    """Splits the given UCI Adult data into features and target.
    """
    ignored_cols = [UCI_ADULT_TARGET_COL, UCI_ADULT_SENSITIVE_COL, "fnlwgt"]
    feature_cols = [col for col in data.columns if col not in ignored_cols]

    X = data[feature_cols]
    Y = data[UCI_ADULT_TARGET_COL].astype(float)
    S = pd.Series(
        data=[1. if val == "Male" else 0. for val in data[UCI_ADULT_SENSITIVE_COL]],
        dtype=float,)
    return X, Y, S


def compute_recall_at_target(y_true, y_pred, fpr=None, fnr=None) -> float:
    pass # TODO


def compute_fairness_ratio(y_true: np.ndarray, y_pred: np.ndarray, s_true, metric: str) -> float:
    """Compute fairness metric as the disparity (group-wise ratio)
    of a given performance metric.

    Parameters
    ----------
    y_true : np.ndarray
        The true labels.
    y_pred : np.ndarray
        The binarized predictions.
    s_true : np.ndarray
        The sensitive attribute column.
    metric : str
        The performance metric used to compute disparity.

    Returns
    -------
    value : float
        The fairness metric value (between 0 and 1).
    """
    metric = metric.lower()
    valid_perf_metrics = ("fpr", "fnr", "tpr", "tnr")
    
    def compute_metric(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        if metric == "fpr":
            return fp / (fp + tn)
        elif metric == "tnr":
            return tn / (fp + tn)
        elif metric == "fnr":
            return fn / (fn + tp)
        elif metric == "tpr":
            return tp / (fn + tp)
        else:
            raise ValueError(f"Invalid metric chosen; must be one of {valid_perf_metrics}; got '{metric}'")

    groupwise_metrics = []
    for group in pd.Series(s_true).unique():
        group_filter = (s_true == group)
        
        groupwise_metrics.append(compute_metric(
            y_true[group_filter],
            y_pred[group_filter],
        ))

    return min(groupwise_metrics) / max(groupwise_metrics)
        
        
        