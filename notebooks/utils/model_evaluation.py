"""This model relates to evaluating (performance of) already trained models.

1. Gathers model metadata and saved predictions from tuner_output
saved in DB.
2. Evaluates models/predictions on a number of metrics.
3. Inserts corresponding metadata in DB.

"""
import time
import logging
from typing import Callable
from inspect import signature

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
import lightgbm as lgb

from .thresholding import compute_threshold, PREDICT_PRECISION
from .randomized_classifier import randomized_classifier_predictions
from .fairautoml_utils_types import require_type
from .fairautoml_utils_classpath import import_object


def try_hyperparams(hyperparams, data: dict, eval_on_train: bool = False, **eval_kwargs) -> dict:
    def instantiate_model(classpath: str, **kwargs) -> object:
        constructor = require_type(import_object(classpath), Callable)
        return constructor(**kwargs)
    
    # Instantiate model
    start_time = time.process_time()
    model = instantiate_model(**hyperparams)

    # Fit model
    model.fit(data["X_train"], data["y_train"])
    seconds = time.process_time() - start_time

    # Evaluate on train data as well?
    data_for_evaluation = data.copy()
    if not eval_on_train:
        data_for_evaluation.pop("train")

    # Evaluate model
    eval_results = evaluate_model(model, data_for_evaluation, **eval_kwargs)
    eval_results["time-taken"] = seconds
    return eval_results


def evaluate_model(
        model,
        data: dict,
        target_metric,
        target_value,
        set_test_threshold_on_validation: bool = False,
        randomized_classifier: bool = False,
        min_iter: int = None,
        n_threads: int = 5,
    ) -> dict:

    results = {}
    # Evaluate predictions for each data set
    ordered_sets = ["train", "validation", "test"]
    for elem in ordered_sets:
        if elem not in data:
            #print(f"{elem} set not found. Skipping...")
            continue

        df, y_true, s_true = data[elem], data[f"y_{elem}"], data[f"s_{elem}"]
        if randomized_classifier:
            assert min_iter is not None, "Please provide min_iter kwarg to use randomized classifier"
            y_scores = randomized_classifier_predictions(model, df, min_iter=min_iter)
        else:
            # if "num_threads" in signature(model.predict).parameters.keys():
            if isinstance(model, (lgb.Booster, lgb.LGBMClassifier, lgb.LGBMRegressor)):
                y_scores = model.predict(df, num_threads=n_threads)
            else:
                X = data[f"X_{elem}"]
                y_scores = model.predict(X)

        if len(y_scores.shape) > 1:
            y_scores = y_scores[:, -1]

        if elem == "test" and set_test_threshold_on_validation:
            eval_kwargs = {"threshold": results["validation"]["threshold"]}
        else:
            eval_kwargs = {target_metric: target_value}

        results[elem] = evaluate_predictions(
            y_true=y_true, y_pred=y_scores, s_true=s_true,
            tie_breaker=True, **eval_kwargs,
        )

    return results


def evaluate_predictions(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        s_true: np.ndarray = None,
        fpr: float = None,
        tpr: float = None,
        pp: int = None,
        threshold: float = None,
        tie_breaker: bool = True,
    ) -> dict:
    """Evaluates the predictions represented by the given metadata object,
    and returns the Evaluation metadata.

    If provided a `threshold` must also provide the uuid of the matrix
    where it was set.

    Parameters
    ----------
    y_true : np.ndarray
        The true target labels.
    y_pred : np.ndarray
        The predicted target scores.
    s_true : np.ndarray, optional
        The sensitive attribute column.
    fpr: float
        Prediction threshold will be set to match this value of FPR.
        Exactly one threshold target must be given.
    tpr: float
        Prediction threshold will be set to match this value of TPR (or Recall).
    pp: int
        Prediction threshold will be set to match this value of positive
        predictions (PP).
    threshold : float
        The exact threshold to be used for classification.
        `positive` if score >= threshold else `negative`.
    tie_breaker : bool
        Whether to use the tie braking mechanism or interpret predictions as is.

    Returns
    -------
    The list of data objects corresponding to the performed evaluations.
    """
    assert sum(1 for val in (tpr, fpr, pp, threshold) if val) == 1, \
        "Got conflicting threshold targets. Must provide only one."

    # Round predictions to same precision as threshold
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64).round(decimals=PREDICT_PRECISION)

    # If no threshold is provided, compute threshold at target metric
    if not threshold:
        y_true, y_pred, threshold = compute_threshold(
            y_true, y_pred,
            fpr=fpr,
            tpr=tpr,
            pp=pp,
            tie_breaker=tie_breaker,
        )
    logging.info(f"Using decision threshold == {threshold}.")
    assert threshold is not None, f"Invalid threshold value: {threshold}"

    # Use threshold with the same precision as is stored on the DB
    threshold = round(threshold, ndigits=PREDICT_PRECISION)

    metrics_results = evaluate_preds_with_treshold(
        y_true, y_pred, threshold=threshold, s_true=s_true,
    )

    return metrics_results


def evaluate_preds_with_treshold(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float,
        s_true: np.ndarray = None,  # If provided, will compute group-wise metrics
    ) -> dict:
    """Computes global metrics and, optionally, group-wise metrics.

    Returns
    -------
    A dictionary of metric_name -> metric_value.
    """

    # Compute metrics
    results = compute_metrics(y_true, y_pred, threshold)
    
    # Compute group-wise metrics
    if s_true is not None:

        group_values = set(s_true)

        for val in group_values:
            group_filter = s_true == val

            group_y_true = y_true[group_filter]
            group_y_pred = y_pred[group_filter]

            group_metrics = compute_metrics(group_y_true, group_y_pred, threshold)
            results.update({
                f"{metric_name}_group-{val}": metric_val
                for metric_name, metric_val in group_metrics.items()
            })

        # Fairness metrics
        groupwise_fpr = [results[f"fpr_group-{val}"] for val in group_values]
        groupwise_fnr = [results[f"fnr_group-{val}"] for val in group_values]

        results["fpr_ratio"] = (min(groupwise_fpr) / max(groupwise_fpr)) if max(groupwise_fpr) > 0 else 0
        results["fpr_diff"] = max(groupwise_fpr) - min(groupwise_fpr)

        results["fnr_ratio"] = (min(groupwise_fnr) / max(groupwise_fnr)) if max(groupwise_fnr) > 0 else 0
        results["fnr_diff"] = max(groupwise_fnr) - min(groupwise_fnr)

    return results


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: np.ndarray) -> dict:

    # Label is predicted positive if score >= threshold
    y_pred_binary = (y_pred >= threshold)

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    label_positives = tp + fn
    label_negatives = tn + fp

    # Build results dict
    results = dict()
    results["threshold"] = threshold
    results["pp"] = np.sum(y_pred_binary)
    results["pn"] = np.sum(~y_pred_binary)
    assert results["pn"] + results["pp"] == len(y_true)

    # Prevalence and predicted prevalence
    results["prevalence"] = np.mean(y_true)
    results["pred-prevalence"] = np.mean(y_pred_binary)

    # FPR at this threshold (false positives over label negatives)
    results["fpr"] = fp / label_negatives if label_negatives > 0 else 0
    results["tnr"] = tn / label_negatives if label_negatives > 0 else 0

    # TPR (Recall) at this threshold (true positives over label positives)
    results["tpr"] = tp / label_positives if label_positives > 0 else 0
    results["fnr"] = fn / label_positives if label_positives > 0 else 0

    # Accuracy at this threshold (note: accuracy value is misleading for imbalanced data)
    results["accuracy"] = (tp + tn) / (label_positives + label_negatives) if len(y_true) > 0 else 0

    # Precision at this threshold
    results["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Area under the ROC curve
    results["roc_auc"] = roc_auc_score(y_true, y_pred)

    # F1-score
    results["f1"] = (
            2 * results["tpr"] * results["precision"] /
            (results["tpr"] + results["precision"])
    ) if (results["tpr"] + results["precision"]) > 1e-3 else 0

    return results
