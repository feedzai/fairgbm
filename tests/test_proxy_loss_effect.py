# coding: utf-8
"""
Property-based tests for proxy loss type effect on constraint gradients.

Feature: fairgbm-dependency-refactor
Property 4: Proxy loss type determines constraint gradients

For any valid training data and two different proxy loss configurations,
the constraint gradient contributions differ.

Validates: Requirements 4.5
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
# Library loading
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

    lib.FairGBM_GetLastError.restype = ctypes.c_char_p
    lib.FairGBM_GetLastError.argtypes = []

    lib.FairGBM_CreateConstrainedObjective.restype = ctypes.c_int
    lib.FairGBM_CreateConstrainedObjective.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]

    lib.FairGBM_FreeConstrainedObjective.restype = ctypes.c_int
    lib.FairGBM_FreeConstrainedObjective.argtypes = [ctypes.c_void_p]

    lib.FairGBM_ObjectiveInit.restype = ctypes.c_int
    lib.FairGBM_ObjectiveInit.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]

    lib.FairGBM_GetGradients.restype = ctypes.c_int
    lib.FairGBM_GetGradients.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]

    lib.FairGBM_GetConstraintGradients.restype = ctypes.c_int
    lib.FairGBM_GetConstraintGradients.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]

    lib.FairGBM_GetNumConstraints.restype = ctypes.c_int
    lib.FairGBM_GetNumConstraints.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
    ]

    return lib


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
# Helpers
# ---------------------------------------------------------------------------


def _create_and_init_objective(
    objective_type, params_str, labels, groups, weights=None
):
    """Create and init a constrained objective, return handle."""
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


def _get_num_constraints(handle):
    """Get number of constraints for a handle."""
    out = ctypes.c_int(0)
    _safe_call(_LIB.FairGBM_GetNumConstraints(handle, ctypes.byref(out)))
    return out.value


def _get_constraint_gradients(handle, multipliers, scores, n):
    """
    Compute constraint gradient contributions.

    Starts from zero gradients/hessians, calls GetConstraintGradients
    which adds constraint contributions in-place.
    """
    gradients = np.zeros(n, dtype=np.float32)
    hessians = np.zeros(n, dtype=np.float32)

    mult_c = multipliers.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    scores_c = scores.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    grad_c = gradients.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    hess_c = hessians.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    _safe_call(
        _LIB.FairGBM_GetConstraintGradients(handle, mult_c, scores_c, grad_c, hess_c)
    )
    return gradients, hessians


def _free_objective(handle):
    _safe_call(_LIB.FairGBM_FreeConstrainedObjective(handle))


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# All valid proxy loss pairs (distinct)
PROXY_TYPES = ["cross_entropy", "hinge", "quadratic"]

proxy_pairs = st.sampled_from(
    [(a, b) for a in PROXY_TYPES for b in PROXY_TYPES if a != b]
)


@st.composite
def training_data_for_proxy_comparison(draw, min_size=20, max_size=100, num_groups=2):
    """Generate training data ensuring proxy losses produce distinct outputs.

    Key insight: proxy functions (hinge, quadratic, cross_entropy) produce
    distinct FPR gradients only when scores for negative instances are in
    ranges where the functions diverge. With margin=1.0:
    - At score > 0: hinge_grad=1, quad_grad=score+1>1, xent_grad=sigmoid(...)
    - At score in (-1, 0): hinge_grad=1, quad_grad=score+1<1, xent_grad<1
    - At score < -1: hinge_grad=0, quad_grad=0, xent_grad≈0

    We ensure negative instances have varied scores in (0.5, 3) where all
    three proxy types produce clearly different gradient values.
    """
    n = draw(st.integers(min_value=min_size, max_value=max_size))

    # Generate balanced labels: first half positive, second half negative
    num_pos = n // 2
    labels = np.zeros(n, dtype=np.float32)
    labels[:num_pos] = 1.0

    # Assign groups in round-robin to ensure balanced groups
    groups = np.array([i % num_groups for i in range(n)], dtype=np.int32)

    # Generate scores in ranges where proxy functions clearly diverge:
    # For negatives (label=0): scores in (0.5, 3.0) — all proxies give
    #   different FPR gradients (hinge=1, quad=score+1>1.5, xent=sigmoid(...))
    # For positives (label=1): scores in (-3.0, -0.5) — all proxies give
    #   different FNR gradients
    scores = np.zeros(n, dtype=np.float64)

    pos_scores = draw(
        arrays(
            dtype=np.float64,
            shape=(num_pos,),
            elements=st.floats(
                min_value=-3.0,
                max_value=-0.5,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
    neg_scores = draw(
        arrays(
            dtype=np.float64,
            shape=(n - num_pos,),
            elements=st.floats(
                min_value=0.5,
                max_value=3.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
    scores[:num_pos] = pos_scores
    scores[num_pos:] = neg_scores

    # Ensure score variance (avoid degenerate all-same cases)
    assume(np.std(neg_scores) > 0.1)
    assume(np.std(pos_scores) > 0.1)

    # Ensure each group has enough negatives and positives
    for g in range(num_groups):
        assume(np.sum((groups == g) & (labels == 0)) >= 2)
        assume(np.sum((groups == g) & (labels == 1)) >= 2)

    return n, labels, groups, scores


# ---------------------------------------------------------------------------
# Property 4: Proxy loss type determines constraint gradients
# ---------------------------------------------------------------------------


class TestProxyLossTypeDeterminesConstraintGradients:
    """
    Feature: fairgbm-dependency-refactor
    Property 4: Proxy loss type determines constraint gradients

    For any valid training data and two different proxy loss
    configurations, the constraint gradient contributions differ.
    """

    @given(
        data=training_data_for_proxy_comparison(),
        proxies=proxy_pairs,
    )
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_different_proxy_fpr_constraint(self, data, proxies):
        """
        FPR constraint: two different proxy types yield different
        constraint gradient contributions on the same data.
        """
        n, labels, groups, scores = data
        proxy_a, proxy_b = proxies

        params_a = f"constraint_type=FPR " f"constraint_stepwise_proxy={proxy_a}"
        params_b = f"constraint_type=FPR " f"constraint_stepwise_proxy={proxy_b}"

        handle_a = _create_and_init_objective(
            "constrained_cross_entropy", params_a, labels, groups
        )
        handle_b = _create_and_init_objective(
            "constrained_cross_entropy", params_b, labels, groups
        )

        try:
            num_c = _get_num_constraints(handle_a)
            assert num_c == _get_num_constraints(handle_b)
            assert num_c > 0

            # Use non-zero multipliers so constraint gradients are active
            multipliers = np.ones(num_c, dtype=np.float64)

            grad_a, hess_a = _get_constraint_gradients(handle_a, multipliers, scores, n)
            grad_b, hess_b = _get_constraint_gradients(handle_b, multipliers, scores, n)

            # At least one of gradients or hessians must differ
            grads_differ = not np.allclose(grad_a, grad_b, atol=1e-7)
            hess_differ = not np.allclose(hess_a, hess_b, atol=1e-7)
            assert grads_differ or hess_differ, (
                f"Proxy types {proxy_a!r} and {proxy_b!r} produced "
                f"identical FPR constraint gradients.\n"
                f"grad_a[:5]={grad_a[:5]}\n"
                f"grad_b[:5]={grad_b[:5]}"
            )
        finally:
            _free_objective(handle_a)
            _free_objective(handle_b)

    @given(
        data=training_data_for_proxy_comparison(),
        proxies=proxy_pairs,
    )
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_different_proxy_fnr_constraint(self, data, proxies):
        """
        FNR constraint: two different proxy types yield different
        constraint gradient contributions.
        """
        n, labels, groups, scores = data
        proxy_a, proxy_b = proxies

        params_a = f"constraint_type=FNR " f"constraint_stepwise_proxy={proxy_a}"
        params_b = f"constraint_type=FNR " f"constraint_stepwise_proxy={proxy_b}"

        handle_a = _create_and_init_objective(
            "constrained_cross_entropy", params_a, labels, groups
        )
        handle_b = _create_and_init_objective(
            "constrained_cross_entropy", params_b, labels, groups
        )

        try:
            num_c = _get_num_constraints(handle_a)
            multipliers = np.ones(num_c, dtype=np.float64)

            grad_a, hess_a = _get_constraint_gradients(handle_a, multipliers, scores, n)
            grad_b, hess_b = _get_constraint_gradients(handle_b, multipliers, scores, n)

            grads_differ = not np.allclose(grad_a, grad_b, atol=1e-7)
            hess_differ = not np.allclose(hess_a, hess_b, atol=1e-7)
            assert grads_differ or hess_differ, (
                f"Proxy types {proxy_a!r} and {proxy_b!r} produced "
                f"identical FNR constraint gradients."
            )
        finally:
            _free_objective(handle_a)
            _free_objective(handle_b)

    @given(
        data=training_data_for_proxy_comparison(),
        proxies=proxy_pairs,
    )
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_different_proxy_recall_objective(self, data, proxies):
        """
        constrained_recall_objective: different constraint proxy types
        yield different constraint gradients.
        """
        n, labels, groups, scores = data
        proxy_a, proxy_b = proxies

        params_a = (
            f"constraint_type=FPR "
            f"constraint_stepwise_proxy={proxy_a} "
            f"objective_stepwise_proxy=cross_entropy "
            f"global_constraint_type=FPR "
            f"global_target_fpr=0.1 "
            f"stepwise_proxy_margin=1.0"
        )
        params_b = (
            f"constraint_type=FPR "
            f"constraint_stepwise_proxy={proxy_b} "
            f"objective_stepwise_proxy=cross_entropy "
            f"global_constraint_type=FPR "
            f"global_target_fpr=0.1 "
            f"stepwise_proxy_margin=1.0"
        )

        handle_a = _create_and_init_objective(
            "constrained_recall_objective", params_a, labels, groups
        )
        handle_b = _create_and_init_objective(
            "constrained_recall_objective", params_b, labels, groups
        )

        try:
            num_c = _get_num_constraints(handle_a)
            multipliers = np.ones(num_c, dtype=np.float64)

            grad_a, hess_a = _get_constraint_gradients(handle_a, multipliers, scores, n)
            grad_b, hess_b = _get_constraint_gradients(handle_b, multipliers, scores, n)

            grads_differ = not np.allclose(grad_a, grad_b, atol=1e-7)
            hess_differ = not np.allclose(hess_a, hess_b, atol=1e-7)
            assert grads_differ or hess_differ, (
                f"Proxy types {proxy_a!r} and {proxy_b!r} produced "
                f"identical recall constraint gradients."
            )
        finally:
            _free_objective(handle_a)
            _free_objective(handle_b)

    @given(data=training_data_for_proxy_comparison())
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_same_proxy_produces_same_constraint_grads(self, data):
        """
        Sanity check: same proxy type on same data yields identical
        constraint gradients (deterministic).
        """
        n, labels, groups, scores = data

        params = "constraint_type=FPR constraint_stepwise_proxy=hinge"

        handle_a = _create_and_init_objective(
            "constrained_cross_entropy", params, labels, groups
        )
        handle_b = _create_and_init_objective(
            "constrained_cross_entropy", params, labels, groups
        )

        try:
            num_c = _get_num_constraints(handle_a)
            multipliers = np.ones(num_c, dtype=np.float64)

            grad_a, hess_a = _get_constraint_gradients(handle_a, multipliers, scores, n)
            grad_b, hess_b = _get_constraint_gradients(handle_b, multipliers, scores, n)

            np.testing.assert_allclose(
                grad_a,
                grad_b,
                atol=1e-7,
                err_msg="Same proxy should produce identical gradients",
            )
            np.testing.assert_allclose(
                hess_a,
                hess_b,
                atol=1e-7,
                err_msg="Same proxy should produce identical hessians",
            )
        finally:
            _free_objective(handle_a)
            _free_objective(handle_b)
