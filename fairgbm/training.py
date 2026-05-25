"""Lagrangian training loop for FairGBM.

Orchestrates constrained gradient boosting by combining LightGBM's
standard training with Lagrangian multiplier updates for fairness constraints.
"""

import numpy as np

import lightgbm

from fairgbm.constrained import ConstrainedObjective
from fairgbm.params import split_params, _is_constrained, _validate_fairgbm_params


def train(
    params,
    train_set,
    num_boost_round=100,
    valid_sets=None,
    valid_names=None,
    feval=None,
    init_model=None,
    keep_training_booster=False,
    callbacks=None,
    constraint_group=None,
):
    """Train a FairGBM model with optional fairness constraints.

    When fairness constraints are configured (via ``constraint_type`` or
    ``global_constraint_type``), this function runs a Lagrangian training
    loop that alternates between LightGBM boosting iterations and dual
    variable (multiplier) updates. Otherwise it delegates to
    ``lightgbm.train()`` directly.

    Parameters
    ----------
    params : dict
        Combined parameter dictionary containing both LightGBM and
        FairGBM-specific parameters.
    train_set : lightgbm.Dataset
        Training data.
    num_boost_round : int, default=100
        Number of boosting iterations.
    valid_sets : list of lightgbm.Dataset or None
        Validation datasets.
    valid_names : list of str or None
        Names for validation datasets.
    feval : callable or None
        Custom evaluation function.
    init_model : str, pathlib.Path, Booster, or None
        Model to continue training from.
    keep_training_booster : bool, default=False
        Whether to return a booster that can be used for further training.
    callbacks : list of callable or None
        Additional callbacks for LightGBM training.
    constraint_group : numpy.ndarray or None
        Integer array of constraint group membership per instance.
        If None, all instances are treated as group 0.

    Returns
    -------
    lightgbm.Booster
        Trained model.
    """
    fairgbm_params, lgbm_params = split_params(params)

    if not _is_constrained(fairgbm_params):
        # Non-constrained: delegate entirely to LightGBM
        return lightgbm.train(
            lgbm_params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            feval=feval,
            init_model=init_model,
            keep_training_booster=keep_training_booster,
            callbacks=callbacks,
        )

    # --- Constrained training path ---

    # Validate FairGBM params before proceeding
    _validate_fairgbm_params(fairgbm_params)

    # Determine objective type
    objective_type = lgbm_params.pop("objective", "constrained_cross_entropy")
    if objective_type in ("none", "custom", ""):
        objective_type = "constrained_cross_entropy"

    # Create constrained objective
    obj = ConstrainedObjective(objective_type, fairgbm_params)

    # Ensure Dataset is constructed so we can access label/weight
    train_set.construct()

    # Extract training metadata
    labels = train_set.get_label()
    weights = train_set.get_weight()
    num_data = len(labels)

    # Resolve constraint groups
    if constraint_group is None:
        constraint_group = np.zeros(num_data, dtype=np.int32)
    else:
        constraint_group = np.ascontiguousarray(constraint_group, dtype=np.int32)

    # Initialize objective with data
    obj.init(labels, constraint_group, weights)

    # Initialize Lagrangian multipliers
    num_constraints = obj.num_constraints
    init_multipliers = fairgbm_params.get("init_lagrangian_multipliers", [])
    if init_multipliers:
        if len(init_multipliers) != num_constraints:
            raise ValueError(
                f"init_lagrangian_multipliers length ({len(init_multipliers)}) "
                f"does not match number of constraints ({num_constraints})"
            )
        multipliers = np.array(init_multipliers, dtype=np.float64)
    else:
        multipliers = np.zeros(num_constraints, dtype=np.float64)

    lr = float(fairgbm_params.get("multiplier_learning_rate", 0.1))

    # Store latest predictions from fobj for use in the multiplier update callback
    _latest_scores = {}

    # Custom objective function for LightGBM
    def fobj(preds, dataset):
        _latest_scores["preds"] = preds.copy()
        grad, hess = obj.get_gradients(preds)
        obj.get_constraint_gradients(multipliers, preds, grad, hess)
        return grad, hess

    # Callback for Lagrangian multiplier update after each iteration
    def _lagrangian_update(env):
        nonlocal multipliers
        # Use scores from the most recent fobj call (same iteration)
        scores = _latest_scores.get("preds")
        if scores is None:
            return
        # Compute constraint violations
        updates = obj.get_multiplier_updates(scores)
        # Gradient ascent on dual variables, clamped to non-negative
        multipliers = np.maximum(0.0, multipliers + lr * updates)

    # Assemble callbacks
    all_callbacks = [_lagrangian_update]
    if callbacks:
        all_callbacks.extend(callbacks)

    # Pass custom objective as callable in params (LightGBM 4.x API)
    lgbm_params["objective"] = fobj

    booster = lightgbm.train(
        lgbm_params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        feval=feval,
        init_model=init_model,
        keep_training_booster=keep_training_booster,
        callbacks=all_callbacks,
    )

    return booster
