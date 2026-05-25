"""Parameter splitting and configuration handling for FairGBM."""

# All FairGBM-specific parameter names
FAIRGBM_PARAMS = {
    "multiplier_learning_rate",
    "constraint_type",
    "constraint_stepwise_proxy",
    "objective_stepwise_proxy",
    "stepwise_proxy_margin",
    "constraint_fpr_tolerance",
    "constraint_fnr_tolerance",
    "score_threshold",
    "global_constraint_type",
    "global_target_fpr",
    "global_target_fnr",
    "global_score_threshold",
    "init_lagrangian_multipliers",
    "debugging_output_dir",
    "constraint_group_column",
}

# Default values for FairGBM-specific parameters
_DEFAULTS = {
    "multiplier_learning_rate": 0.1,
    "constraint_type": "",
    "constraint_stepwise_proxy": "cross_entropy",
    "objective_stepwise_proxy": "",
    "stepwise_proxy_margin": 1.0,
    "constraint_fpr_tolerance": 0.0,
    "constraint_fnr_tolerance": 0.0,
    "score_threshold": 0.5,
    "global_constraint_type": "",
    "global_target_fpr": 0.0,
    "global_target_fnr": 0.0,
    "global_score_threshold": 0.5,
    "init_lagrangian_multipliers": [],
    "debugging_output_dir": ".",
    "constraint_group_column": "",
}


VALID_CONSTRAINT_TYPES = {"FPR", "FNR", "FPR,FNR", ""}
VALID_PROXY_TYPES = {"cross_entropy", "hinge", "quadratic", ""}


def _validate_fairgbm_params(fairgbm_params):
    """Validate FairGBM-specific parameter values.

    Only validates string-typed values. Non-string values are passed
    through without validation (they will fail at the C API level if used).

    Parameters
    ----------
    fairgbm_params : dict
        FairGBM-specific parameters.

    Raises
    ------
    ValueError
        If any parameter has an invalid string value.
    """
    constraint_type = fairgbm_params.get("constraint_type", "")
    if (
        isinstance(constraint_type, str)
        and constraint_type
        and constraint_type not in VALID_CONSTRAINT_TYPES
    ):
        raise ValueError(
            f"Invalid constraint_type: '{constraint_type}'. "
            f"Valid options: 'FPR', 'FNR', 'FPR,FNR'"
        )

    global_constraint_type = fairgbm_params.get("global_constraint_type", "")
    if (
        isinstance(global_constraint_type, str)
        and global_constraint_type
        and global_constraint_type not in VALID_CONSTRAINT_TYPES
    ):
        raise ValueError(
            f"Invalid global_constraint_type: '{global_constraint_type}'. "
            f"Valid options: 'FPR', 'FNR', 'FPR,FNR'"
        )

    constraint_proxy = fairgbm_params.get("constraint_stepwise_proxy", "")
    if (
        isinstance(constraint_proxy, str)
        and constraint_proxy
        and constraint_proxy not in VALID_PROXY_TYPES
    ):
        raise ValueError(
            f"Invalid constraint_stepwise_proxy: '{constraint_proxy}'. "
            f"Valid options: 'cross_entropy', 'hinge', 'quadratic'"
        )

    objective_proxy = fairgbm_params.get("objective_stepwise_proxy", "")
    if (
        isinstance(objective_proxy, str)
        and objective_proxy
        and objective_proxy not in VALID_PROXY_TYPES
    ):
        raise ValueError(
            f"Invalid objective_stepwise_proxy: '{objective_proxy}'. "
            f"Valid options: 'cross_entropy', 'hinge', 'quadratic'"
        )


def split_params(params):
    """Split a parameter dict into FairGBM-specific and LightGBM params.

    Parameters
    ----------
    params : dict
        Combined parameter dictionary.

    Returns
    -------
    tuple of (dict, dict)
        (fairgbm_params, lgbm_params) where fairgbm_params contains all
        FairGBM-specific keys with defaults applied, and lgbm_params
        contains the remaining keys for LightGBM.
    """
    fairgbm_params = dict(_DEFAULTS)
    lgbm_params = {}

    for key, value in params.items():
        if key in FAIRGBM_PARAMS:
            fairgbm_params[key] = value
        else:
            lgbm_params[key] = value

    return fairgbm_params, lgbm_params


def _is_constrained(fairgbm_params):
    """Check if configuration uses fairness constraints.

    Parameters
    ----------
    fairgbm_params : dict
        FairGBM-specific parameters (output of split_params).

    Returns
    -------
    bool
        True if constraint_type or global_constraint_type is set.
    """
    return bool(fairgbm_params.get("constraint_type")) or bool(
        fairgbm_params.get("global_constraint_type")
    )
