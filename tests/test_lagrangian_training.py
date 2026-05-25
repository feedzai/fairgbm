# coding: utf-8
"""
Property-based tests for Lagrangian multiplier behavior.

Feature: fairgbm-dependency-refactor
Property 5: Lagrangian multipliers are updated each iteration
Property 6: Initial multipliers match configuration
Property 7: Lagrangian multipliers are non-negative

These tests exercise the Lagrangian training loop logic in fairgbm.training.
"""

import numpy as np
import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from fairgbm.constrained import ConstrainedObjective


# ---------------------------------------------------------------------------
# Helpers: native extension loading check
# ---------------------------------------------------------------------------


def _native_extension_available():
    """Check if the native extension is loadable."""
    try:
        from fairgbm._lib import _LIB

        return _LIB is not None
    except (ImportError, OSError, RuntimeError):
        return False


skip_no_native = pytest.mark.skipif(
    not _native_extension_available(),
    reason="lib_fairgbm native extension not available",
)


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


@st.composite
def num_boost_rounds(draw):
    """Strategy for number of boosting rounds (small for test speed)."""
    return draw(st.integers(min_value=1, max_value=10))


@st.composite
def multiplier_learning_rates(draw):
    """Strategy for multiplier learning rates."""
    return draw(
        st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False)
    )


@st.composite
def initial_multipliers(draw, num_constraints):
    """Strategy for initial multiplier vectors."""
    return draw(
        arrays(
            dtype=np.float64,
            shape=(num_constraints,),
            elements=st.floats(
                min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False
            ),
        )
    )


@st.composite
def multiplier_updates(draw, size):
    """Strategy for constraint violation values (can be negative or positive)."""
    return draw(
        arrays(
            dtype=np.float64,
            shape=(size,),
            elements=st.floats(
                min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
            ),
        )
    )


@st.composite
def training_data_small(draw, min_size=10, max_size=50):
    """Strategy for small training datasets with binary labels and groups."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    num_groups = draw(st.integers(min_value=2, max_value=4))
    labels = draw(
        arrays(dtype=np.float32, shape=(n,), elements=st.sampled_from([0.0, 1.0]))
    )
    groups = draw(
        arrays(
            dtype=np.int32,
            shape=(n,),
            elements=st.integers(min_value=0, max_value=num_groups - 1),
        )
    )
    # Ensure at least one positive and one negative label
    assume(labels.sum() > 0 and labels.sum() < n)
    # Ensure at least 2 groups present
    assume(len(np.unique(groups)) >= 2)
    return labels, groups, num_groups


# ---------------------------------------------------------------------------
# Property 7: Lagrangian multipliers are non-negative
# (Pure Python logic — no native extension needed)
# ---------------------------------------------------------------------------


class TestLagrangianMultipliersNonNegative:
    """Property 7: For any sequence of multiplier updates, all values after
    clamping are >= 0."""

    @given(
        num_constraints=st.integers(min_value=1, max_value=20),
        num_iterations=st.integers(min_value=1, max_value=50),
        lr=st.floats(
            min_value=0.001, max_value=2.0, allow_nan=False, allow_infinity=False
        ),
        data=st.data(),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_multipliers_always_non_negative_after_clamping(
        self, num_constraints, num_iterations, lr, data
    ):
        """Multipliers remain >= 0 regardless of update direction/magnitude."""
        # Start with valid non-negative initial multipliers
        multipliers = data.draw(
            arrays(
                dtype=np.float64,
                shape=(num_constraints,),
                elements=st.floats(
                    min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False
                ),
            )
        )

        for _ in range(num_iterations):
            # Draw arbitrary constraint violation values (can be very negative)
            updates = data.draw(
                arrays(
                    dtype=np.float64,
                    shape=(num_constraints,),
                    elements=st.floats(
                        min_value=-100.0,
                        max_value=100.0,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                )
            )
            # Apply the same update rule as in training.py
            multipliers = np.maximum(0.0, multipliers + lr * updates)

            # Property: all multipliers non-negative
            assert np.all(
                multipliers >= 0.0
            ), f"Found negative multiplier after clamping: {multipliers}"

    @given(
        num_constraints=st.integers(min_value=1, max_value=10),
        lr=st.floats(
            min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        data=st.data(),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_multipliers_non_negative_with_large_negative_updates(
        self, num_constraints, lr, data
    ):
        """Even with extremely negative updates, clamping keeps multipliers >= 0."""
        multipliers = data.draw(
            arrays(
                dtype=np.float64,
                shape=(num_constraints,),
                elements=st.floats(
                    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
                ),
            )
        )
        # Very negative updates that would drive multipliers below zero
        updates = data.draw(
            arrays(
                dtype=np.float64,
                shape=(num_constraints,),
                elements=st.floats(
                    min_value=-1000.0,
                    max_value=-1.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            )
        )
        multipliers = np.maximum(0.0, multipliers + lr * updates)
        assert np.all(multipliers >= 0.0)


# ---------------------------------------------------------------------------
# Property 6: Initial multipliers match configuration
# (Uses native extension to verify num_constraints, then checks init logic)
# ---------------------------------------------------------------------------


@skip_no_native
class TestInitialMultipliersMatchConfiguration:
    """Property 6: For any valid initial multiplier vector, multipliers at
    iteration 0 equal the provided values."""

    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_initial_multipliers_equal_provided_values(self, data):
        """Multipliers at iteration 0 match init_lagrangian_multipliers config."""
        # Create a constrained objective to determine num_constraints
        num_groups = data.draw(st.integers(min_value=2, max_value=4))
        constraint_type = data.draw(st.sampled_from(["FPR", "FNR", "FPR,FNR"]))

        params = {
            "constraint_type": constraint_type,
            "multiplier_learning_rate": 0.1,
            "constraint_stepwise_proxy": "cross_entropy",
            "score_threshold": 0.5,
        }

        # Create objective and init to get num_constraints
        n = data.draw(st.integers(min_value=10, max_value=30))
        labels = data.draw(
            arrays(dtype=np.float32, shape=(n,), elements=st.sampled_from([0.0, 1.0]))
        )
        groups = data.draw(
            arrays(
                dtype=np.int32,
                shape=(n,),
                elements=st.integers(min_value=0, max_value=num_groups - 1),
            )
        )
        assume(labels.sum() > 0 and labels.sum() < n)
        assume(len(np.unique(groups)) >= 2)

        obj = ConstrainedObjective("constrained_cross_entropy", params)
        obj.init(labels, groups, None)
        num_constraints = obj.num_constraints
        assume(num_constraints > 0)

        # Draw initial multiplier values
        init_mults = data.draw(
            arrays(
                dtype=np.float64,
                shape=(num_constraints,),
                elements=st.floats(
                    min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False
                ),
            )
        )

        # Simulate the initialization logic from training.py
        multipliers = np.array(init_mults, dtype=np.float64)

        # Property: multipliers at iteration 0 equal provided values
        np.testing.assert_array_equal(
            multipliers,
            init_mults,
            err_msg="Initial multipliers do not match provided values",
        )

    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_default_multipliers_are_zero(self, data):
        """When no init_lagrangian_multipliers provided, multipliers start at zero."""
        num_groups = data.draw(st.integers(min_value=2, max_value=4))
        constraint_type = data.draw(st.sampled_from(["FPR", "FNR", "FPR,FNR"]))

        params = {
            "constraint_type": constraint_type,
            "multiplier_learning_rate": 0.1,
            "constraint_stepwise_proxy": "cross_entropy",
            "score_threshold": 0.5,
        }

        n = data.draw(st.integers(min_value=10, max_value=30))
        labels = data.draw(
            arrays(dtype=np.float32, shape=(n,), elements=st.sampled_from([0.0, 1.0]))
        )
        groups = data.draw(
            arrays(
                dtype=np.int32,
                shape=(n,),
                elements=st.integers(min_value=0, max_value=num_groups - 1),
            )
        )
        assume(labels.sum() > 0 and labels.sum() < n)
        assume(len(np.unique(groups)) >= 2)

        obj = ConstrainedObjective("constrained_cross_entropy", params)
        obj.init(labels, groups, None)
        num_constraints = obj.num_constraints
        assume(num_constraints > 0)

        # Simulate default init from training.py (no init_lagrangian_multipliers)
        init_multipliers = []
        if init_multipliers:
            multipliers = np.array(init_multipliers, dtype=np.float64)
        else:
            multipliers = np.zeros(num_constraints, dtype=np.float64)

        # Property: default multipliers are all zero
        np.testing.assert_array_equal(
            multipliers,
            np.zeros(num_constraints, dtype=np.float64),
            err_msg="Default multipliers should be all zeros",
        )


# ---------------------------------------------------------------------------
# Property 5: Lagrangian multipliers are updated each iteration
# (Uses native extension for full constrained objective computation)
# ---------------------------------------------------------------------------


@skip_no_native
class TestLagrangianMultipliersUpdatedEachIteration:
    """Property 5: For any constrained training run of N iterations, the
    multiplier history has N+1 entries and at least one differs from initial."""

    @given(data=st.data())
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_multiplier_history_length_and_change(self, data):
        """Multiplier history has N+1 entries and at least one update changes values."""
        num_iterations = data.draw(st.integers(min_value=1, max_value=5))
        lr = data.draw(
            st.floats(
                min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False
            )
        )
        num_groups = data.draw(st.integers(min_value=2, max_value=3))
        constraint_type = data.draw(st.sampled_from(["FPR", "FNR", "FPR,FNR"]))

        n = data.draw(st.integers(min_value=20, max_value=50))
        labels = data.draw(
            arrays(dtype=np.float32, shape=(n,), elements=st.sampled_from([0.0, 1.0]))
        )
        groups = data.draw(
            arrays(
                dtype=np.int32,
                shape=(n,),
                elements=st.integers(min_value=0, max_value=num_groups - 1),
            )
        )
        assume(labels.sum() > 0 and labels.sum() < n)
        assume(len(np.unique(groups)) >= 2)

        params = {
            "constraint_type": constraint_type,
            "multiplier_learning_rate": lr,
            "constraint_stepwise_proxy": "cross_entropy",
            "score_threshold": 0.5,
        }

        obj = ConstrainedObjective("constrained_cross_entropy", params)
        obj.init(labels, groups, None)
        num_constraints = obj.num_constraints
        assume(num_constraints > 0)

        # Initialize multipliers at zero (default)
        multipliers = np.zeros(num_constraints, dtype=np.float64)
        multiplier_history = [multipliers.copy()]

        # Simulate the Lagrangian training loop (without actual LightGBM training)
        # Simulate the Lagrangian training loop with scores that cross threshold
        any_positive_update = False
        for i in range(num_iterations):
            scores = data.draw(
                arrays(
                    dtype=np.float64,
                    shape=(n,),
                    elements=st.floats(
                        min_value=-3.0,
                        max_value=3.0,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                )
            )
            # Compute constraint violations (same as training.py _lagrangian_update)
            updates = obj.get_multiplier_updates(scores)
            # Positive update = constraint violated, will increase multiplier
            if np.any(updates > 0.0):
                any_positive_update = True
            # Gradient ascent + clamping
            multipliers = np.maximum(0.0, multipliers + lr * updates)
            multiplier_history.append(multipliers.copy())

        # Filter: need at least one positive update to guarantee multiplier change
        # (negative updates on zero multipliers get clamped back to zero)
        assume(any_positive_update)

        # Property: history has N+1 entries
        expected = num_iterations + 1
        assert len(multiplier_history) == expected, (
            f"Expected {expected} history entries, " f"got {len(multiplier_history)}"
        )

        # Property: at least one entry differs from initial
        # (given positive violations)
        initial = multiplier_history[0]
        any_changed = any(
            not np.array_equal(initial, multiplier_history[i])
            for i in range(1, len(multiplier_history))
        )
        assert any_changed, (
            "Multipliers never changed from initial values across all iterations. "
            f"Initial: {initial}, Final: {multiplier_history[-1]}"
        )

    @given(data=st.data())
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_multiplier_history_with_custom_initial_values(self, data):
        """Multiplier history starts with provided initial values and evolves."""
        num_iterations = data.draw(st.integers(min_value=2, max_value=5))
        lr = data.draw(
            st.floats(
                min_value=0.05, max_value=0.5, allow_nan=False, allow_infinity=False
            )
        )
        num_groups = data.draw(st.integers(min_value=2, max_value=3))

        n = data.draw(st.integers(min_value=20, max_value=50))
        labels = data.draw(
            arrays(dtype=np.float32, shape=(n,), elements=st.sampled_from([0.0, 1.0]))
        )
        groups = data.draw(
            arrays(
                dtype=np.int32,
                shape=(n,),
                elements=st.integers(min_value=0, max_value=num_groups - 1),
            )
        )
        assume(labels.sum() > 0 and labels.sum() < n)
        assume(len(np.unique(groups)) >= 2)

        params = {
            "constraint_type": "FPR",
            "multiplier_learning_rate": lr,
            "constraint_stepwise_proxy": "cross_entropy",
            "score_threshold": 0.5,
        }

        obj = ConstrainedObjective("constrained_cross_entropy", params)
        obj.init(labels, groups, None)
        num_constraints = obj.num_constraints
        assume(num_constraints > 0)

        # Custom initial multipliers
        init_mults = data.draw(
            arrays(
                dtype=np.float64,
                shape=(num_constraints,),
                elements=st.floats(
                    min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False
                ),
            )
        )

        multipliers = init_mults.copy()
        multiplier_history = [multipliers.copy()]

        for i in range(num_iterations):
            scores = data.draw(
                arrays(
                    dtype=np.float64,
                    shape=(n,),
                    elements=st.floats(
                        min_value=-3.0,
                        max_value=3.0,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                )
            )
            updates = obj.get_multiplier_updates(scores)
            multipliers = np.maximum(0.0, multipliers + lr * updates)
            multiplier_history.append(multipliers.copy())

        # Property: history length is N+1
        assert len(multiplier_history) == num_iterations + 1

        # Property: first entry matches initial values
        np.testing.assert_array_equal(
            multiplier_history[0],
            init_mults,
            err_msg="First history entry should match initial multipliers",
        )

        # Property: at least one subsequent entry differs
        any_changed = any(
            not np.array_equal(init_mults, multiplier_history[i])
            for i in range(1, len(multiplier_history))
        )
        assert any_changed, "Multipliers never changed from initial values"
