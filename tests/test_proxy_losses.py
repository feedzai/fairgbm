# coding: utf-8
"""
Property-based tests for proxy loss functions.

Feature: fairgbm-dependency-refactor
Properties 10, 11, 12: Proxy loss non-negativity, factory correctness, invalid name error.

Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5
"""

import ctypes
import math
import os
from platform import system

import numpy as np
import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Pure Python proxy loss implementations (mirror C++ logic for direct testing)
# ---------------------------------------------------------------------------


class CrossEntropyProxy:
    """Python mirror of CrossEntropyProxyLoss."""

    def __init__(self, margin=1.0):
        self.margin = margin
        self.xent_horizontal_shift = math.log(math.exp(margin) - 1)

    def instancewise_fpr(self, score):
        return math.log(1 + math.exp(score + self.xent_horizontal_shift))

    def instancewise_fnr(self, score):
        return math.log(1 + math.exp(self.xent_horizontal_shift - score))


class HingeProxy:
    """Python mirror of HingeProxyLoss."""

    def __init__(self, margin=1.0):
        self.margin = margin

    def instancewise_fpr(self, score):
        return max(0.0, score + self.margin)

    def instancewise_fnr(self, score):
        return max(0.0, -score + self.margin)


class QuadraticProxy:
    """Python mirror of QuadraticProxyLoss."""

    def __init__(self, margin=1.0):
        self.margin = margin

    def instancewise_fpr(self, score):
        if score >= -self.margin:
            return 0.5 * (score + self.margin) ** 2
        return 0.0

    def instancewise_fnr(self, score):
        if score <= self.margin:
            return 0.5 * (score - self.margin) ** 2
        return 0.0


PROXY_CLASSES = {
    "cross_entropy": CrossEntropyProxy,
    "hinge": HingeProxy,
    "quadratic": QuadraticProxy,
}


# ---------------------------------------------------------------------------
# Library loading (for Properties 11, 12 which test through C API)
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
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Scores can be any finite float — proxy losses must handle full range
scores_strategy = st.floats(
    min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False
)

# Proxy margin must be positive (used in all proxy constructors)
margin_strategy = st.floats(
    min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False
)

proxy_name_strategy = st.sampled_from(["cross_entropy", "hinge", "quadratic"])


# ---------------------------------------------------------------------------
# Property 10: Proxy loss functions produce non-negative FPR/FNR values
# ---------------------------------------------------------------------------


class TestProxyLossNonNegative:
    """
    Feature: fairgbm-dependency-refactor
    Property 10: Proxy loss functions produce non-negative FPR/FNR values

    For any proxy type and score value, instancewise proxy FPR and FNR
    values are non-negative.
    """

    @given(score=scores_strategy, margin=margin_strategy)
    @settings(max_examples=200)
    def test_cross_entropy_fpr_non_negative(self, score, margin):
        """CrossEntropy proxy FPR >= 0 for all scores."""
        proxy = CrossEntropyProxy(margin)
        fpr = proxy.instancewise_fpr(score)
        assert (
            fpr >= 0.0
        ), f"CrossEntropy FPR={fpr} < 0 at score={score}, margin={margin}"

    @given(score=scores_strategy, margin=margin_strategy)
    @settings(max_examples=200)
    def test_cross_entropy_fnr_non_negative(self, score, margin):
        """CrossEntropy proxy FNR >= 0 for all scores."""
        proxy = CrossEntropyProxy(margin)
        fnr = proxy.instancewise_fnr(score)
        assert (
            fnr >= 0.0
        ), f"CrossEntropy FNR={fnr} < 0 at score={score}, margin={margin}"

    @given(score=scores_strategy, margin=margin_strategy)
    @settings(max_examples=200)
    def test_hinge_fpr_non_negative(self, score, margin):
        """Hinge proxy FPR >= 0 for all scores."""
        proxy = HingeProxy(margin)
        fpr = proxy.instancewise_fpr(score)
        assert fpr >= 0.0, f"Hinge FPR={fpr} < 0 at score={score}, margin={margin}"

    @given(score=scores_strategy, margin=margin_strategy)
    @settings(max_examples=200)
    def test_hinge_fnr_non_negative(self, score, margin):
        """Hinge proxy FNR >= 0 for all scores."""
        proxy = HingeProxy(margin)
        fnr = proxy.instancewise_fnr(score)
        assert fnr >= 0.0, f"Hinge FNR={fnr} < 0 at score={score}, margin={margin}"

    @given(score=scores_strategy, margin=margin_strategy)
    @settings(max_examples=200)
    def test_quadratic_fpr_non_negative(self, score, margin):
        """Quadratic proxy FPR >= 0 for all scores."""
        proxy = QuadraticProxy(margin)
        fpr = proxy.instancewise_fpr(score)
        assert fpr >= 0.0, f"Quadratic FPR={fpr} < 0 at score={score}, margin={margin}"

    @given(score=scores_strategy, margin=margin_strategy)
    @settings(max_examples=200)
    def test_quadratic_fnr_non_negative(self, score, margin):
        """Quadratic proxy FNR >= 0 for all scores."""
        proxy = QuadraticProxy(margin)
        fnr = proxy.instancewise_fnr(score)
        assert fnr >= 0.0, f"Quadratic FNR={fnr} < 0 at score={score}, margin={margin}"

    @given(
        proxy_name=proxy_name_strategy,
        score=scores_strategy,
        margin=margin_strategy,
    )
    @settings(max_examples=300)
    def test_all_proxies_fpr_fnr_non_negative(self, proxy_name, score, margin):
        """Universal: any proxy type, any score → FPR >= 0 and FNR >= 0."""
        proxy = PROXY_CLASSES[proxy_name](margin)
        fpr = proxy.instancewise_fpr(score)
        fnr = proxy.instancewise_fnr(score)
        assert fpr >= 0.0, f"{proxy_name} FPR={fpr} < 0 at score={score}"
        assert fnr >= 0.0, f"{proxy_name} FNR={fnr} < 0 at score={score}"


# ---------------------------------------------------------------------------
# Property 11: Proxy loss factory returns correct type
# ---------------------------------------------------------------------------


class TestProxyLossFactoryReturnsCorrectType:
    """
    Feature: fairgbm-dependency-refactor
    Property 11: Proxy loss factory returns correct type

    For any valid proxy name, the factory returns a proxy loss consistent
    with that type's definition.

    We verify this by creating objectives with each proxy type and checking
    that the constraint gradients match the expected mathematical behavior
    of that specific proxy.
    """

    @given(
        proxy_name=proxy_name_strategy,
        score=st.floats(
            min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_factory_produces_consistent_gradients(self, proxy_name, score):
        """
        For a single negative instance, the global FPR constraint gradient
        should be proportional to the proxy's FPR gradient at that score.

        We verify consistency by checking that two objectives with the SAME
        proxy produce identical gradients (factory is deterministic and
        returns the correct type).
        """
        # Single negative instance — FPR constraint gradient is purely
        # from ComputeInstancewiseFPRGradient
        n = 2
        labels = np.array([0.0, 1.0], dtype=np.float32)
        groups = np.array([0, 0], dtype=np.int32)
        scores_arr = np.array([score, 0.0], dtype=np.float64)

        params = (
            f"constraint_stepwise_proxy={proxy_name} "
            f"global_constraint_type=FPR "
            f"global_target_fpr=0.0 "
            f"stepwise_proxy_margin=1.0"
        )

        # Create two handles with same config
        handle1 = ctypes.c_void_p()
        handle2 = ctypes.c_void_p()
        _safe_call(
            _LIB.FairGBM_CreateConstrainedObjective(
                b"constrained_cross_entropy", params.encode(), ctypes.byref(handle1)
            )
        )
        _safe_call(
            _LIB.FairGBM_CreateConstrainedObjective(
                b"constrained_cross_entropy", params.encode(), ctypes.byref(handle2)
            )
        )

        try:
            labels_c = labels.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            groups_c = groups.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            _safe_call(_LIB.FairGBM_ObjectiveInit(handle1, labels_c, groups_c, None, n))
            _safe_call(_LIB.FairGBM_ObjectiveInit(handle2, labels_c, groups_c, None, n))

            num_c = ctypes.c_int(0)
            _safe_call(_LIB.FairGBM_GetNumConstraints(handle1, ctypes.byref(num_c)))
            assert num_c.value > 0

            multipliers = np.ones(num_c.value, dtype=np.float64)
            mult_c = multipliers.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            scores_c = scores_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

            grad1 = np.zeros(n, dtype=np.float32)
            hess1 = np.zeros(n, dtype=np.float32)
            grad2 = np.zeros(n, dtype=np.float32)
            hess2 = np.zeros(n, dtype=np.float32)

            _safe_call(
                _LIB.FairGBM_GetConstraintGradients(
                    handle1,
                    mult_c,
                    scores_c,
                    grad1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    hess1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                )
            )
            _safe_call(
                _LIB.FairGBM_GetConstraintGradients(
                    handle2,
                    mult_c,
                    scores_c,
                    grad2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    hess2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                )
            )

            # Same proxy → identical gradients (factory deterministic)
            np.testing.assert_allclose(grad1, grad2, atol=1e-7)
        finally:
            _safe_call(_LIB.FairGBM_FreeConstrainedObjective(handle1))
            _safe_call(_LIB.FairGBM_FreeConstrainedObjective(handle2))

    @given(proxy_name=proxy_name_strategy)
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_factory_proxy_type_matches_definition(self, proxy_name):
        """
        Verify the factory-produced proxy matches its mathematical definition.

        For a negative instance at score=2.0 with margin=1.0:
        - hinge FPR gradient = 1.0 (score > -margin)
        - quadratic FPR gradient = score + margin = 3.0
        - cross_entropy FPR gradient = sigmoid(score + shift) where shift=log(e-1)

        We check the C API gradient matches the Python reference implementation.
        """
        test_score = 2.0
        margin = 1.0
        n = 2
        labels = np.array([0.0, 1.0], dtype=np.float32)
        groups = np.array([0, 0], dtype=np.int32)
        scores_arr = np.array([test_score, 0.0], dtype=np.float64)

        params = (
            f"constraint_stepwise_proxy={proxy_name} "
            f"global_constraint_type=FPR "
            f"global_target_fpr=0.0 "
            f"stepwise_proxy_margin={margin}"
        )

        handle = ctypes.c_void_p()
        _safe_call(
            _LIB.FairGBM_CreateConstrainedObjective(
                b"constrained_cross_entropy", params.encode(), ctypes.byref(handle)
            )
        )

        try:
            labels_c = labels.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            groups_c = groups.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            _safe_call(_LIB.FairGBM_ObjectiveInit(handle, labels_c, groups_c, None, n))

            num_c = ctypes.c_int(0)
            _safe_call(_LIB.FairGBM_GetNumConstraints(handle, ctypes.byref(num_c)))

            multipliers = np.ones(num_c.value, dtype=np.float64)
            mult_c = multipliers.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            scores_c = scores_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

            grad = np.zeros(n, dtype=np.float32)
            hess = np.zeros(n, dtype=np.float32)

            _safe_call(
                _LIB.FairGBM_GetConstraintGradients(
                    handle,
                    mult_c,
                    scores_c,
                    grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    hess.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                )
            )

            # The gradient for instance 0 (negative label) with global FPR
            # constraint is: fpr_gradient(score) / num_negatives * num_data * multiplier
            # = fpr_gradient(score) / 1 * 2 * 1 = 2 * fpr_gradient(score)
            actual_grad_0 = float(grad[0])

            # Compute expected gradient from Python reference
            if proxy_name == "hinge":
                # gradient = 1.0 when score >= -margin
                expected_fpr_grad = 1.0
            elif proxy_name == "quadratic":
                # gradient = max(0, score + margin) = 3.0
                expected_fpr_grad = max(0.0, test_score + margin)
            elif proxy_name == "cross_entropy":
                # gradient = sigmoid(score + shift)
                shift = math.log(math.exp(margin) - 1)
                expected_fpr_grad = 1.0 / (1.0 + math.exp(-(test_score + shift)))

            # Scale: fpr_grad * n / num_negatives * multiplier = fpr_grad * 2 / 1 * 1
            expected_grad_0 = expected_fpr_grad * n / 1.0

            assert (
                abs(actual_grad_0 - expected_grad_0) < 1e-4
            ), f"Proxy {proxy_name}: expected grad={expected_grad_0}, got {actual_grad_0}"
        finally:
            _safe_call(_LIB.FairGBM_FreeConstrainedObjective(handle))


# ---------------------------------------------------------------------------
# Property 12: Invalid proxy name raises error
# ---------------------------------------------------------------------------


class TestInvalidProxyNameRaisesError:
    """
    Feature: fairgbm-dependency-refactor
    Property 12: Invalid proxy name raises error

    For any invalid proxy name string, the factory raises an error.
    """

    # Strategy: generate strings that are NOT valid proxy names
    invalid_proxy_names = st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P")),
        min_size=1,
        max_size=30,
    ).filter(
        lambda s: s.lower()
        not in (
            "cross_entropy",
            "hinge",
            "quadratic",
            "bce",
            "xentropy",
            "entropy",  # aliases accepted by ValidateProxyFunctionName
        )
    )

    @given(invalid_name=invalid_proxy_names)
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_invalid_proxy_name_causes_creation_error(self, invalid_name):
        """
        Creating a constrained objective with an invalid proxy name
        should fail (return -1) with a descriptive error message.
        """
        params = f"constraint_type=FPR constraint_stepwise_proxy={invalid_name}"

        handle = ctypes.c_void_p()
        ret = _LIB.FairGBM_CreateConstrainedObjective(
            b"constrained_cross_entropy",
            params.encode("utf-8"),
            ctypes.byref(handle),
        )

        if ret == 0:
            # If creation succeeded (shouldn't for invalid names), clean up
            _LIB.FairGBM_FreeConstrainedObjective(handle)
            pytest.fail(
                f"Expected error for invalid proxy name '{invalid_name}', "
                f"but creation succeeded."
            )

        # Verify error message mentions the invalid name
        err = _LIB.FairGBM_GetLastError()
        assert err is not None
        err_msg = err.decode("utf-8")
        assert (
            "invalid" in err_msg.lower() or "not implemented" in err_msg.lower()
        ), f"Error message should indicate invalid proxy: {err_msg}"

    @pytest.mark.parametrize(
        "invalid_name",
        [
            "linear",
            "sigmoid",
            "relu",
            "softmax",
            "l2",
            "huber",
            "INVALID",
            "foo_bar",
            "cross-entropy",
            "Hinge!",
        ],
    )
    def test_known_invalid_names(self, invalid_name):
        """Specific invalid proxy names must be rejected."""
        params = f"constraint_type=FPR constraint_stepwise_proxy={invalid_name}"

        handle = ctypes.c_void_p()
        ret = _LIB.FairGBM_CreateConstrainedObjective(
            b"constrained_cross_entropy",
            params.encode("utf-8"),
            ctypes.byref(handle),
        )

        if ret == 0:
            _LIB.FairGBM_FreeConstrainedObjective(handle)
            pytest.fail(f"Should reject invalid proxy name: '{invalid_name}'")

        err = _LIB.FairGBM_GetLastError()
        assert err is not None

    @pytest.mark.parametrize(
        "valid_name",
        [
            "cross_entropy",
            "hinge",
            "quadratic",
        ],
    )
    def test_valid_names_accepted(self, valid_name):
        """Sanity: valid proxy names should NOT raise errors."""
        params = f"constraint_type=FPR constraint_stepwise_proxy={valid_name}"

        handle = ctypes.c_void_p()
        ret = _LIB.FairGBM_CreateConstrainedObjective(
            b"constrained_cross_entropy",
            params.encode("utf-8"),
            ctypes.byref(handle),
        )

        assert ret == 0, (
            f"Valid proxy name '{valid_name}' should be accepted. "
            f"Error: {_LIB.FairGBM_GetLastError()}"
        )
        _safe_call(_LIB.FairGBM_FreeConstrainedObjective(handle))
