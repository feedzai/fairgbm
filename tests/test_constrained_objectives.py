# coding: utf-8
"""
Property-based tests for constrained objective gradient validity.

Feature: fairgbm-dependency-refactor
Property 1: Constrained cross-entropy gradients are valid
Property 2: Constrained recall gradients are valid

These tests exercise the FairGBM C API (lib_fairgbm) via ctypes.
"""

import ctypes
import os
from platform import system

import numpy as np
import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


# ---------------------------------------------------------------------------
# Library loading helpers
# ---------------------------------------------------------------------------


def _find_lib_fairgbm():
    """Locate the lib_fairgbm shared library."""
    curr_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(curr_path, "..")

    search_paths = [
        os.path.join(root_path, "lib"),
        os.path.join(root_path, "build"),
        os.path.join(root_path, "native", "build"),
        os.path.join(root_path, "fairgbm"),
        root_path,
    ]

    if system() in ("Windows", "Microsoft"):
        lib_name = "lib_fairgbm.dll"
    elif system() == "Darwin":
        lib_name = "lib_fairgbm.dylib"
    else:
        lib_name = "lib_fairgbm.so"

    candidates = [os.path.join(p, lib_name) for p in search_paths]
    for path in candidates:
        if os.path.isfile(path):
            return path

    # Also check env variable
    env_path = os.environ.get("FAIRGBM_LIB_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    pytest.skip(
        f"lib_fairgbm not found. Searched: {candidates}. "
        "Build the native extension first or set FAIRGBM_LIB_PATH."
    )


def _load_lib():
    """Load lib_fairgbm and set up ctypes signatures."""
    lib_path = _find_lib_fairgbm()
    lib = ctypes.cdll.LoadLibrary(lib_path)

    # FairGBM_GetLastError
    lib.FairGBM_GetLastError.restype = ctypes.c_char_p
    lib.FairGBM_GetLastError.argtypes = []

    # FairGBM_CreateConstrainedObjective
    lib.FairGBM_CreateConstrainedObjective.restype = ctypes.c_int
    lib.FairGBM_CreateConstrainedObjective.argtypes = [
        ctypes.c_char_p,  # objective_type
        ctypes.c_char_p,  # params_str
        ctypes.POINTER(ctypes.c_void_p),  # out handle
    ]

    # FairGBM_FreeConstrainedObjective
    lib.FairGBM_FreeConstrainedObjective.restype = ctypes.c_int
    lib.FairGBM_FreeConstrainedObjective.argtypes = [ctypes.c_void_p]

    # FairGBM_ObjectiveInit
    lib.FairGBM_ObjectiveInit.restype = ctypes.c_int
    lib.FairGBM_ObjectiveInit.argtypes = [
        ctypes.c_void_p,  # handle
        ctypes.POINTER(ctypes.c_float),  # labels
        ctypes.POINTER(ctypes.c_int),  # constraint_groups
        ctypes.POINTER(ctypes.c_float),  # weights (nullable)
        ctypes.c_int,  # num_data
    ]

    # FairGBM_GetGradients
    lib.FairGBM_GetGradients.restype = ctypes.c_int
    lib.FairGBM_GetGradients.argtypes = [
        ctypes.c_void_p,  # handle
        ctypes.POINTER(ctypes.c_double),  # scores
        ctypes.POINTER(ctypes.c_float),  # out_gradients
        ctypes.POINTER(ctypes.c_float),  # out_hessians
    ]

    # FairGBM_GetNumConstraints
    lib.FairGBM_GetNumConstraints.restype = ctypes.c_int
    lib.FairGBM_GetNumConstraints.argtypes = [
        ctypes.c_void_p,  # handle
        ctypes.POINTER(ctypes.c_int),  # out_num_constraints
    ]

    return lib


# Module-level library (skip entire module if not available)
_LIB = None


@pytest.fixture(autouse=True, scope="module")
def load_library():
    global _LIB
    _LIB = _load_lib()


def _safe_call(ret):
    """Check return code and raise on error."""
    if ret != 0:
        err = _LIB.FairGBM_GetLastError()
        msg = err.decode("utf-8") if err else "Unknown error"
        raise RuntimeError(f"FairGBM C API error: {msg}")


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


def binary_labels(n):
    """Strategy for binary label arrays of length n."""
    return arrays(
        dtype=np.float32,
        shape=(n,),
        elements=st.sampled_from([0.0, 1.0]),
    )


def constraint_groups(n, max_groups=5):
    """Strategy for constraint group arrays of length n."""
    return arrays(
        dtype=np.int32,
        shape=(n,),
        elements=st.integers(min_value=0, max_value=max_groups - 1),
    )


def scores_array(n):
    """Strategy for score arrays (arbitrary floats, bounded to avoid overflow)."""
    return arrays(
        dtype=np.float64,
        shape=(n,),
        elements=st.floats(
            min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
    )


def positive_weights(n):
    """Strategy for positive weight arrays."""
    return arrays(
        dtype=np.float32,
        shape=(n,),
        elements=st.floats(
            min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
    )


# Composite strategy: generate consistent data of random size
@st.composite
def training_data(draw, min_size=4, max_size=200, max_groups=5):
    """Generate consistent training data: labels, groups, scores, weights."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    labels = draw(binary_labels(n))
    groups = draw(constraint_groups(n, max_groups))
    scores = draw(scores_array(n))

    # Ensure at least one positive and one negative label for meaningful gradients
    assume(np.any(labels == 0.0) and np.any(labels == 1.0))

    # Optionally include weights
    use_weights = draw(st.booleans())
    weights = draw(positive_weights(n)) if use_weights else None

    return n, labels, groups, scores, weights


# ---------------------------------------------------------------------------
# Helper: create and initialize an objective via C API
# ---------------------------------------------------------------------------


def _create_and_init_objective(objective_type, params_str, labels, groups, weights):
    """Create a constrained objective, initialize it, return the handle."""
    handle = ctypes.c_void_p()
    _safe_call(
        _LIB.FairGBM_CreateConstrainedObjective(
            objective_type.encode("utf-8"),
            params_str.encode("utf-8"),
            ctypes.byref(handle),
        )
    )

    n = len(labels)
    labels_c = labels.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    groups_c = groups.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    weights_c = (
        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        if weights is not None
        else None
    )

    _safe_call(_LIB.FairGBM_ObjectiveInit(handle, labels_c, groups_c, weights_c, n))
    return handle


def _free_objective(handle):
    """Free a constrained objective handle."""
    _safe_call(_LIB.FairGBM_FreeConstrainedObjective(handle))


def _get_gradients(handle, scores, n):
    """Call FairGBM_GetGradients and return (gradients, hessians) as numpy arrays."""
    gradients = np.zeros(n, dtype=np.float32)
    hessians = np.zeros(n, dtype=np.float32)
    scores_c = scores.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    grad_c = gradients.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    hess_c = hessians.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    _safe_call(_LIB.FairGBM_GetGradients(handle, scores_c, grad_c, hess_c))
    return gradients, hessians


# ---------------------------------------------------------------------------
# Property 1: Constrained cross-entropy gradients are valid
# ---------------------------------------------------------------------------


class TestConstrainedCrossEntropyGradients:
    """
    Feature: fairgbm-dependency-refactor
    Property 1: Constrained cross-entropy gradients are valid

    For any valid binary labels, scores, constraint groups, and weights,
    constrained_cross_entropy produces gradients and hessians of correct
    length with non-negative hessians.
    """

    @given(data=training_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_gradient_length_and_hessian_nonnegativity(self, data):
        """Gradients/hessians have correct length; hessians are non-negative."""
        n, labels, groups, scores, weights = data

        params = "constraint_type=FPR constraint_stepwise_proxy=cross_entropy"
        handle = _create_and_init_objective(
            "constrained_cross_entropy", params, labels, groups, weights
        )
        try:
            gradients, hessians = _get_gradients(handle, scores, n)

            # Correct length
            assert len(gradients) == n, f"Expected {n} gradients, got {len(gradients)}"
            assert len(hessians) == n, f"Expected {n} hessians, got {len(hessians)}"

            # Hessians non-negative (BCE hessian = sigmoid(s) * (1 - sigmoid(s)) >= 0)
            assert np.all(
                hessians >= 0.0
            ), f"Found negative hessians: min={hessians.min()}"

            # Gradients are finite
            assert np.all(np.isfinite(gradients)), "Gradients contain non-finite values"
            assert np.all(np.isfinite(hessians)), "Hessians contain non-finite values"
        finally:
            _free_objective(handle)

    @given(data=training_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_gradient_length_fnr_constraint(self, data):
        """FNR constraint type also produces valid gradients."""
        n, labels, groups, scores, weights = data

        params = "constraint_type=FNR constraint_stepwise_proxy=cross_entropy"
        handle = _create_and_init_objective(
            "constrained_cross_entropy", params, labels, groups, weights
        )
        try:
            gradients, hessians = _get_gradients(handle, scores, n)

            assert len(gradients) == n
            assert len(hessians) == n
            assert np.all(hessians >= 0.0)
            assert np.all(np.isfinite(gradients))
            assert np.all(np.isfinite(hessians))
        finally:
            _free_objective(handle)

    @given(data=training_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_gradient_length_combined_constraint(self, data):
        """Combined FPR,FNR constraint type also produces valid gradients."""
        n, labels, groups, scores, weights = data

        params = "constraint_type=FPR,FNR constraint_stepwise_proxy=cross_entropy"
        handle = _create_and_init_objective(
            "constrained_cross_entropy", params, labels, groups, weights
        )
        try:
            gradients, hessians = _get_gradients(handle, scores, n)

            assert len(gradients) == n
            assert len(hessians) == n
            assert np.all(hessians >= 0.0)
            assert np.all(np.isfinite(gradients))
            assert np.all(np.isfinite(hessians))
        finally:
            _free_objective(handle)

    @given(
        data=training_data(),
        proxy=st.sampled_from(["cross_entropy", "hinge", "quadratic"]),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_gradient_valid_all_proxy_types(self, data, proxy):
        """All proxy types produce valid gradients for cross-entropy objective."""
        n, labels, groups, scores, weights = data

        params = f"constraint_type=FPR constraint_stepwise_proxy={proxy}"
        handle = _create_and_init_objective(
            "constrained_cross_entropy", params, labels, groups, weights
        )
        try:
            gradients, hessians = _get_gradients(handle, scores, n)

            assert len(gradients) == n
            assert len(hessians) == n
            assert np.all(
                hessians >= 0.0
            ), f"Negative hessians with proxy={proxy}: min={hessians.min()}"
            assert np.all(np.isfinite(gradients))
            assert np.all(np.isfinite(hessians))
        finally:
            _free_objective(handle)


# ---------------------------------------------------------------------------
# Property 2: Constrained recall gradients are valid
# ---------------------------------------------------------------------------


class TestConstrainedRecallGradients:
    """
    Feature: fairgbm-dependency-refactor
    Property 2: Constrained recall gradients are valid

    For any valid binary labels, scores, constraint groups, and global FPR
    constraint, constrained_recall_objective produces gradients and hessians
    of correct length with non-negative hessians.
    """

    @given(
        data=training_data(),
        target_fpr=st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
        obj_proxy=st.sampled_from(["cross_entropy", "hinge", "quadratic"]),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_gradient_length_and_hessian_nonnegativity(
        self, data, target_fpr, obj_proxy
    ):
        """Gradients/hessians have correct length; hessians are non-negative."""
        n, labels, groups, scores, weights = data

        params = (
            f"global_constraint_type=FPR "
            f"global_target_fpr={target_fpr} "
            f"constraint_stepwise_proxy=cross_entropy "
            f"objective_stepwise_proxy={obj_proxy} "
            f"stepwise_proxy_margin=1.0"
        )
        handle = _create_and_init_objective(
            "constrained_recall_objective", params, labels, groups, weights
        )
        try:
            gradients, hessians = _get_gradients(handle, scores, n)

            # Correct length
            assert len(gradients) == n, f"Expected {n} gradients, got {len(gradients)}"
            assert len(hessians) == n, f"Expected {n} hessians, got {len(hessians)}"

            # Hessians non-negative
            assert np.all(hessians >= 0.0), (
                f"Negative hessians with obj_proxy={obj_proxy}: "
                f"min={hessians.min()}"
            )

            # Gradients are finite
            assert np.all(np.isfinite(gradients)), "Gradients contain non-finite values"
            assert np.all(np.isfinite(hessians)), "Hessians contain non-finite values"
        finally:
            _free_objective(handle)

    @given(
        data=training_data(),
        target_fpr=st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_gradient_valid_with_group_and_global_constraints(self, data, target_fpr):
        """Recall objective with both group FPR and global FPR constraints."""
        n, labels, groups, scores, weights = data

        params = (
            f"constraint_type=FPR "
            f"global_constraint_type=FPR "
            f"global_target_fpr={target_fpr} "
            f"constraint_stepwise_proxy=cross_entropy "
            f"objective_stepwise_proxy=cross_entropy "
            f"stepwise_proxy_margin=1.0"
        )
        handle = _create_and_init_objective(
            "constrained_recall_objective", params, labels, groups, weights
        )
        try:
            gradients, hessians = _get_gradients(handle, scores, n)

            assert len(gradients) == n
            assert len(hessians) == n
            assert np.all(hessians >= 0.0)
            assert np.all(np.isfinite(gradients))
            assert np.all(np.isfinite(hessians))
        finally:
            _free_objective(handle)

    @given(data=training_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_gradient_valid_without_weights(self, data):
        """Recall objective without weights produces valid gradients."""
        n, labels, groups, scores, _ = data

        params = (
            "global_constraint_type=FPR "
            "global_target_fpr=0.1 "
            "constraint_stepwise_proxy=cross_entropy "
            "objective_stepwise_proxy=quadratic "
            "stepwise_proxy_margin=1.0"
        )
        handle = _create_and_init_objective(
            "constrained_recall_objective", params, labels, groups, None
        )
        try:
            gradients, hessians = _get_gradients(handle, scores, n)

            assert len(gradients) == n
            assert len(hessians) == n
            assert np.all(hessians >= 0.0)
            assert np.all(np.isfinite(gradients))
            assert np.all(np.isfinite(hessians))
        finally:
            _free_objective(handle)

    @given(
        data=training_data(),
        target_fpr=st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
        margin=st.floats(min_value=0.1, max_value=5.0, allow_nan=False),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_gradient_valid_varying_margin(self, data, target_fpr, margin):
        """Different proxy margins produce valid gradients."""
        n, labels, groups, scores, weights = data

        params = (
            f"global_constraint_type=FPR "
            f"global_target_fpr={target_fpr} "
            f"constraint_stepwise_proxy=cross_entropy "
            f"objective_stepwise_proxy=cross_entropy "
            f"stepwise_proxy_margin={margin}"
        )
        handle = _create_and_init_objective(
            "constrained_recall_objective", params, labels, groups, weights
        )
        try:
            gradients, hessians = _get_gradients(handle, scores, n)

            assert len(gradients) == n
            assert len(hessians) == n
            assert np.all(hessians >= 0.0)
            assert np.all(np.isfinite(gradients))
            assert np.all(np.isfinite(hessians))
        finally:
            _free_objective(handle)
