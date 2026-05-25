# coding: utf-8
"""
Property-based tests for parameter splitting.

Feature: fairgbm-dependency-refactor
Property 13: FairGBM parameters are separated from LightGBM parameters

For any mixed parameter dictionary, split_params produces two dicts where
the FairGBM dict contains all and only FairGBM-specific keys, and the
LightGBM dict contains no FairGBM-specific keys.

Validates: Requirements 9.1, 9.3
"""

import sys
import os

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

# Ensure fairgbm package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fairgbm.params import split_params, FAIRGBM_PARAMS, _DEFAULTS, _is_constrained


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Arbitrary values that could appear in a param dict
param_values = st.one_of(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.integers(min_value=-1000, max_value=1000),
    st.text(min_size=0, max_size=20),
    st.lists(
        st.floats(min_value=0, max_value=10, allow_nan=False, allow_infinity=False),
        max_size=5,
    ),
    st.booleans(),
)

# LightGBM parameter names (representative sample)
LGBM_PARAM_NAMES = [
    "num_leaves",
    "max_depth",
    "learning_rate",
    "n_estimators",
    "objective",
    "metric",
    "boosting_type",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "min_child_samples",
    "min_child_weight",
    "num_threads",
    "verbose",
    "seed",
    "bagging_fraction",
    "feature_fraction",
    "early_stopping_rounds",
    "categorical_feature",
]

fairgbm_param_names = st.sampled_from(sorted(FAIRGBM_PARAMS))
lgbm_param_names = st.sampled_from(LGBM_PARAM_NAMES)


@st.composite
def mixed_param_dict(draw):
    """Generate a dict with a mix of FairGBM and LightGBM params."""
    # Draw some FairGBM params
    n_fairgbm = draw(st.integers(min_value=0, max_value=len(FAIRGBM_PARAMS)))
    fairgbm_keys = draw(
        st.lists(
            fairgbm_param_names, min_size=n_fairgbm, max_size=n_fairgbm, unique=True
        )
    )

    # Draw some LightGBM params
    n_lgbm = draw(st.integers(min_value=0, max_value=len(LGBM_PARAM_NAMES)))
    lgbm_keys = draw(
        st.lists(lgbm_param_names, min_size=n_lgbm, max_size=n_lgbm, unique=True)
    )

    params = {}
    for k in fairgbm_keys:
        params[k] = draw(param_values)
    for k in lgbm_keys:
        params[k] = draw(param_values)

    return params


@st.composite
def only_fairgbm_params(draw):
    """Generate a dict with only FairGBM-specific params."""
    n = draw(st.integers(min_value=1, max_value=len(FAIRGBM_PARAMS)))
    keys = draw(st.lists(fairgbm_param_names, min_size=n, max_size=n, unique=True))
    return {k: draw(param_values) for k in keys}


@st.composite
def only_lgbm_params(draw):
    """Generate a dict with only LightGBM params."""
    n = draw(st.integers(min_value=1, max_value=len(LGBM_PARAM_NAMES)))
    keys = draw(st.lists(lgbm_param_names, min_size=n, max_size=n, unique=True))
    return {k: draw(param_values) for k in keys}


# ---------------------------------------------------------------------------
# Property 13: FairGBM parameters are separated from LightGBM parameters
# ---------------------------------------------------------------------------


class TestParameterSplitting:
    """
    Feature: fairgbm-dependency-refactor
    Property 13: FairGBM parameters are separated from LightGBM parameters
    """

    @given(params=mixed_param_dict())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_fairgbm_dict_contains_only_fairgbm_keys(self, params):
        """FairGBM output dict keys are subset of FAIRGBM_PARAMS."""
        fairgbm_params, _ = split_params(params)
        assert set(fairgbm_params.keys()).issubset(FAIRGBM_PARAMS)

    @given(params=mixed_param_dict())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_lgbm_dict_contains_no_fairgbm_keys(self, params):
        """LightGBM output dict has no FairGBM-specific keys."""
        _, lgbm_params = split_params(params)
        assert set(lgbm_params.keys()).isdisjoint(FAIRGBM_PARAMS)

    @given(params=mixed_param_dict())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_all_input_keys_accounted_for(self, params):
        """Every input key appears in exactly one output dict."""
        fairgbm_params, lgbm_params = split_params(params)
        for key in params:
            if key in FAIRGBM_PARAMS:
                assert key in fairgbm_params
                assert key not in lgbm_params
            else:
                assert key in lgbm_params
                assert key not in fairgbm_params

    @given(params=mixed_param_dict())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_fairgbm_values_match_input(self, params):
        """FairGBM params in output have same values as input."""
        fairgbm_params, _ = split_params(params)
        for key in params:
            if key in FAIRGBM_PARAMS:
                assert fairgbm_params[key] == params[key]

    @given(params=mixed_param_dict())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_lgbm_values_match_input(self, params):
        """LightGBM params in output have same values as input."""
        _, lgbm_params = split_params(params)
        for key in params:
            if key not in FAIRGBM_PARAMS:
                assert lgbm_params[key] == params[key]

    @given(params=mixed_param_dict())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_defaults_applied_for_missing_fairgbm_keys(self, params):
        """FairGBM keys not in input get default values."""
        fairgbm_params, _ = split_params(params)
        for key in FAIRGBM_PARAMS:
            if key not in params:
                assert fairgbm_params[key] == _DEFAULTS[key]

    @given(params=only_lgbm_params())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_no_fairgbm_input_yields_all_defaults(self, params):
        """When no FairGBM keys in input, FairGBM dict equals defaults."""
        fairgbm_params, _ = split_params(params)
        assert fairgbm_params == _DEFAULTS

    @given(params=only_fairgbm_params())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_no_lgbm_input_yields_empty_lgbm_dict(self, params):
        """When no LightGBM keys in input, LightGBM dict is empty."""
        _, lgbm_params = split_params(params)
        assert lgbm_params == {}

    @given(params=st.just({}))
    def test_empty_input(self, params):
        """Empty input → defaults for FairGBM, empty for LightGBM."""
        fairgbm_params, lgbm_params = split_params(params)
        assert fairgbm_params == _DEFAULTS
        assert lgbm_params == {}
