# coding: utf-8
"""
Property-based tests for constraint group handling.

Feature: fairgbm-dependency-refactor
Property 8: Constraint group round-trip
Property 9: Constraint group validation rejects invalid inputs

Validates: Requirements 6.1, 6.2, 6.5
"""

import sys
import os

import numpy as np
import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

# Ensure fairgbm package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fairgbm.dataset import FairGBMDataset


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


# Valid constraint group arrays: non-negative integers
def valid_constraint_groups(min_size=1, max_size=200):
    """Strategy for valid constraint group arrays."""
    return arrays(
        dtype=np.int32,
        shape=st.integers(min_value=min_size, max_value=max_size),
        elements=st.integers(min_value=0, max_value=10),
    )


def matching_labels_and_groups(min_size=2, max_size=200):
    """Strategy for labels and constraint groups of matching length."""

    @st.composite
    def _inner(draw):
        n = draw(st.integers(min_value=min_size, max_value=max_size))
        labels = draw(
            arrays(
                dtype=np.float32,
                shape=n,
                elements=st.sampled_from([0.0, 1.0]),
            )
        )
        groups = draw(
            arrays(
                dtype=np.int32,
                shape=n,
                elements=st.integers(min_value=0, max_value=10),
            )
        )
        return labels, groups

    return _inner()


# ---------------------------------------------------------------------------
# Property 8: Constraint group round-trip
# ---------------------------------------------------------------------------


class TestConstraintGroupRoundTrip:
    """
    Feature: fairgbm-dependency-refactor
    Property 8: Constraint group round-trip

    For any valid integer array, setting and getting constraint group
    returns an equal array.
    """

    @given(data=matching_labels_and_groups())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_roundtrip_via_constructor(self, data):
        """Constructor constraint_group round-trips correctly."""
        labels, groups = data
        ds = FairGBMDataset(
            np.random.randn(len(labels), 3).astype(np.float32),
            label=labels,
            constraint_group=groups,
        )
        result = ds.get_constraint_group()
        np.testing.assert_array_equal(result, groups)

    @given(data=matching_labels_and_groups())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_roundtrip_via_setter(self, data):
        """Setter constraint_group round-trips correctly."""
        labels, groups = data
        ds = FairGBMDataset(
            np.random.randn(len(labels), 3).astype(np.float32),
            label=labels,
        )
        ds.set_constraint_group(groups)
        result = ds.get_constraint_group()
        np.testing.assert_array_equal(result, groups)

    @given(data=matching_labels_and_groups())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_roundtrip_preserves_dtype(self, data):
        """Returned constraint_group is always int32."""
        labels, groups = data
        # Pass as int64 — should be stored as int32
        groups_i64 = groups.astype(np.int64)
        ds = FairGBMDataset(
            np.random.randn(len(labels), 3).astype(np.float32),
            label=labels,
            constraint_group=groups_i64,
        )
        result = ds.get_constraint_group()
        assert result.dtype == np.int32
        np.testing.assert_array_equal(result, groups)

    @given(data=matching_labels_and_groups())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_roundtrip_is_contiguous(self, data):
        """Returned constraint_group is C-contiguous."""
        labels, groups = data
        ds = FairGBMDataset(
            np.random.randn(len(labels), 3).astype(np.float32),
            label=labels,
            constraint_group=groups,
        )
        result = ds.get_constraint_group()
        assert result.flags["C_CONTIGUOUS"]


# ---------------------------------------------------------------------------
# Property 9: Constraint group validation rejects invalid inputs
# ---------------------------------------------------------------------------


class TestConstraintGroupValidation:
    """
    Feature: fairgbm-dependency-refactor
    Property 9: Constraint group validation rejects invalid inputs

    For any array with wrong dtype, negative values, or wrong length,
    setting constraint group raises a validation error.
    """

    @given(
        n=st.integers(min_value=2, max_value=100),
        data=st.data(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_rejects_float_dtype(self, n, data):
        """Float arrays are rejected."""
        labels = np.zeros(n, dtype=np.float32)
        groups_float = data.draw(
            arrays(
                dtype=np.float64,
                shape=n,
                elements=st.floats(min_value=0.0, max_value=10.0),
            )
        )
        ds = FairGBMDataset(
            np.random.randn(n, 3).astype(np.float32),
            label=labels,
        )
        with pytest.raises(ValueError, match="integer-typed"):
            ds.set_constraint_group(groups_float)

    @given(
        n=st.integers(min_value=2, max_value=100),
        data=st.data(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_rejects_negative_values(self, n, data):
        """Arrays with negative values are rejected."""
        labels = np.zeros(n, dtype=np.float32)
        # Generate array with at least one negative value
        groups = data.draw(
            arrays(
                dtype=np.int32,
                shape=n,
                elements=st.integers(min_value=-10, max_value=10),
            )
        )
        assume(np.any(groups < 0))

        ds = FairGBMDataset(
            np.random.randn(n, 3).astype(np.float32),
            label=labels,
        )
        with pytest.raises(ValueError, match="non-negative"):
            ds.set_constraint_group(groups)

    @given(
        n=st.integers(min_value=2, max_value=100),
        offset=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_rejects_wrong_length(self, n, offset):
        """Arrays with wrong length are rejected."""
        labels = np.zeros(n, dtype=np.float32)
        # Wrong length: either too short or too long
        wrong_len = n + offset
        groups = np.zeros(wrong_len, dtype=np.int32)

        ds = FairGBMDataset(
            np.random.randn(n, 3).astype(np.float32),
            label=labels,
        )
        with pytest.raises(ValueError, match="length"):
            ds.set_constraint_group(groups)

    @given(n=st.integers(min_value=2, max_value=100))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_rejects_string_dtype(self, n):
        """String arrays are rejected."""
        labels = np.zeros(n, dtype=np.float32)
        groups_str = np.array(["a"] * n)

        ds = FairGBMDataset(
            np.random.randn(n, 3).astype(np.float32),
            label=labels,
        )
        with pytest.raises(ValueError, match="integer-typed"):
            ds.set_constraint_group(groups_str)

    def test_none_constraint_group_returns_none(self):
        """When no constraint_group is set, get returns None."""
        ds = FairGBMDataset(
            np.random.randn(10, 3).astype(np.float32),
            label=np.zeros(10, dtype=np.float32),
        )
        assert ds.get_constraint_group() is None
