"""Functions for finding the right threshold for a given target metric.
"""
import logging
import math
from typing import Tuple, Union, Callable

import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix


UNTIE_PRECISION = 9
PREDICT_PRECISION = 12


def compute_threshold(
        y_true: np.array, y_pred: np.array, tie_breaker=True, **kwargs,
    ):
    """Computes the threshold at the given target.

    Returns
    -------
    A tuple containing (y_true, y_pred, threshold).
    """
    if tie_breaker:
        y_true, y_pred, threshold = custom_threshold(y_true, y_pred, **kwargs)
    else:
        y_true, y_pred, threshold = threshold_at_target_sklearn(y_true, y_pred, **kwargs)

    assert isinstance(threshold, (float, int)), \
        f'Invalid threshold value: {threshold} (of type {type(threshold)})'
    return y_true, y_pred, threshold


def threshold_at_target_sklearn(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        tpr: float = None,
        fpr: float = None,
        pp: int = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Computes the threshold at the given target.
    Does not untie rows, may miss target in the presence of ties.
    Uses scikit-learn to compute ROC curve.
    """
    if kwargs:
        logging.warning(f'Not using the following kwargs: {kwargs}.')

    if fpr or tpr:
        fpr_vals, tpr_vals, thresholds = roc_curve(y_true, y_pred, pos_label=1)

        # # Interpolating precise threshold as ROC curve is discrete
        # interpolation = interp1d(tpr if tpr else fpr, thresholds, kind='linear')
        # threshold = interpolation(tpr if tpr else fpr).item()
        ordered_vals = tpr_vals if tpr else fpr_vals
        target_val: float = tpr or fpr

        if tpr:
            assert fpr is None, "Please provide only one target."
            threshold_idx, = np.argwhere(tpr_vals >= tpr)[0]
            assert tpr_vals[threshold_idx] >= tpr
        elif fpr:
            threshold_idx, = np.argwhere(fpr_vals <= fpr)[-1]
            assert fpr_vals[threshold_idx] <= fpr

        threshold = thresholds[threshold_idx]

        # Sanity check!
        y_pred_binary = (y_pred >= threshold)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        actual_tpr = tp / (tp + fn)
        actual_tnr = tn / (tn + fp)
        actual_fpr = 1 - actual_tnr
        if (tpr and not math.isclose(actual_tpr, tpr, rel_tol=1e-3)) or (fpr and not math.isclose(actual_fpr, fpr, rel_tol=1e-3)):
            logging.error(f'Missed target metric: TPR={actual_tpr}, FPR={actual_fpr}')

    elif pp:
        indices = np.argsort(y_pred)[::-1]
        threshold = y_pred[indices][pp - 1]

    else:
        raise RuntimeError(f'Error when setting the threshold at target metric!')

    return y_true, y_pred, float(threshold)


def custom_threshold(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        tpr: float = None,
        fpr: float = None,
        pp: int = None,
        jitter: float = 10 ** -UNTIE_PRECISION,
        pos_label: Union[int, str] = 1,
        round_fn: Callable = math.floor,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Method to compute a custom threshold that hits the target metric as close
    as possible.
    If any prediction ties imposibilitate hitting the target metric, will
    randomly untie these rows.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted scores.
    tpr : float
        Target tpr.
        Only one of (tpr, fpr, pp) can be given.
    fpr : float
        Target FPR
    pp : int
        Target number of positive predictions.
    jitter : float
        Jitter to add when untying results.
    pos_label : int
        Positive label.
    round_fn : Callable
        Rounding function.

    Returns
    -------
    Returns a tuple (<1>, <2>, <3>):
    1. y_true: untouched labels;
    2. y_pred: predictions with possible changes for untying rows;
    3. threshold: the threshold that hits the given target.
    """
    assert len(y_true) == len(y_pred), 'Dimension mismatch'
    assert (10 ** -PREDICT_PRECISION) < jitter * 1e-2, \
        'Jitter must be a larger value in order to correctly serialize threshold.'
    # (times 1e2 for good measure)

    if kwargs:
        logging.warning(f'Not using the following kwargs: {kwargs}.')

    classes = set(y_true)
    n_classes = len(classes)
    if n_classes > 2:
        raise NotImplementedError(
            f"Non-binary setting detected: {n_classes}"
        )

    # Order preds by score descending
    y_pred_sort = (-y_pred).argsort()

    y_true, y_pred = y_true.copy(), y_pred.copy()
    # Ids of the predicted scores
    y_ids = np.arange(len(y_true))

    # Ids of the predicted scores sorted by descending score
    y_ids_sorted = y_ids[y_pred_sort]

    # True labels sorted by descending score
    y_true_sorted = y_true[y_pred_sort]

    # Determine the necessary number of FPs or TPs required to satisfy target
    target_val, target_mask = None, None
    if fpr:
        # Count label negatives
        neg_label = list(classes - {pos_label})[0]
        ln = (y_true == neg_label).sum()
        # Compute target threshold masks
        target_val = round_fn(fpr * ln)  # number of FP required
        target_mask = y_true_sorted == 0        # to satisfy the threshold
    elif tpr:
        # Count label positives
        lp = (y_true == pos_label).sum()
        # Compute target threshold masks
        target_val = round_fn(tpr * lp)  # number of TP required
        target_mask = y_true_sorted == 1        # to satisfy the threshold
    elif pp:
        target_val = pp
        target_mask = np.ones_like(y_ids_sorted).astype(bool)
    else:
        raise RuntimeError(f'Error when setting the threshold at target metric!')

    # The id of the threshold
    y_pred_id = y_ids_sorted[target_mask][target_val-1]
    # The threshold score
    threshold = y_pred[y_pred_id]
    # ^NOTE:
    # ----------------------------------------------------------------------
    # A tie happens when determining the threshold for a given target metric
    # if we have multiple scores with the same value and the threshold lies
    # in between these values:
    # 0.9567
    # 0.9555
    # 0.9555 <-- if this is the threshold there's no way to distinguish
    # 0.9555     instances above the threshold from the ones below.
    # 0.9544
    #  ...
    # To accommodate these changes, we add a jitter, that is, we add
    # a fraction small enough to maintain the same records but not
    # so small that the tie remains.

    # The index of the id in the y_pred_sorted (IDs are unique, access 1st result)
    id_idx = np.argwhere(y_ids_sorted == y_pred_id)[0, 0]

    # The index of the id of the prediction right below y_pred_id
    y_pred_below_thresh_id_idx = id_idx + 1

    # The original id of the prediction
    y_pred_below_thresh_id_original = y_ids_sorted[y_pred_below_thresh_id_idx]
    y_pred_below_thresh_val = y_pred[y_pred_below_thresh_id_original]

    # If chosen threshold leads to ties
    if np.isclose(threshold, y_pred_below_thresh_val, rtol=jitter, atol=jitter):
        # Add jitter to instances below threshold if it's 1
        if threshold + jitter < 1:
            ids_preds_ge_threshold = y_ids_sorted[:y_pred_below_thresh_id_idx]
            add_jitter = y_pred[ids_preds_ge_threshold] + jitter
            add_jitter = np.clip(add_jitter, 0, 1)
            y_pred[ids_preds_ge_threshold] = add_jitter
            threshold += jitter * 0.5

        # Threshold is already to high, subtract jitter to instances below
        else:
            # Subtract jitter to ``preds_ids```
            ids_preds_below_threshold = y_ids_sorted[y_pred_below_thresh_id_idx:]
            sub_jitter = y_pred[ids_preds_below_threshold] - jitter
            sub_jitter = np.clip(sub_jitter, 0, 1)
            y_pred[ids_preds_below_threshold] = sub_jitter

    actual_pred_count = (y_pred[y_ids_sorted[target_mask]] >= threshold).sum()
    assert actual_pred_count == target_val, \
        f"Target metric missed (PP/FP/TP)! Target was {target_val}, got {actual_pred_count}."

    return y_true, y_pred, float(threshold)
