"""
Utility functions to keep the example notebooks uncluttered with boilerplate.
"""

import numpy as np
import pandas as pd


def compute_recall_at_target(y_true, y_pred, fpr=None, fnr=None) -> float:
    pass # TODO


def compute_disparity(y_true, y_pred, metric: str) -> float:
    """Compute fairness metric as the disparity (group-wise ratio)
    of a given performance metric.

    Parameters
    ----------
    y_true
    y_pred
    metric

    Returns
    -------
    value : float
        The fairness metric value (between 0 and 1).
    """
    assert metric.lower() in ("fpr", "fnr", "tpr", "tnr")
    pass # TODO
