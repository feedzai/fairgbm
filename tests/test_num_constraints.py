# coding: utf-8
"""
Property-based tests for number of constraints.

Feature: fairgbm-dependency-refactor
Property 3: Number of constraints matches configuration

For any combination of group/global constraint types and number of unique
groups, NumConstraints equals:
  (num_groups × group_constraint_types) + global_constraint_types.

Validates: Requirements 4.3
"""

import ctypes
import os
from platform import system

import numpy as np
import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Library loading (reused from test_constrained_objectives.py)
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


def _create_and_init_objective(objective_type, params_str, labels, groups):
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

    _safe_call(_LIB.FairGBM_ObjectiveInit(handle, labels_c, groups_c, None, n))
    return handle


def _get_num_constraints(handle):
    """Call FairGBM_GetNumConstraints, return int."""
    out = ctypes.c_int(0)
    _safe_call(_LIB.FairGBM_GetNumConstraints(handle, ctypes.byref(out)))
    return out.value


def _free_objective(handle):
    _safe_call(_LIB.FairGBM_FreeConstrainedObjective(handle))


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Group constraint type: "", "FPR", "FNR", "FPR,FNR"
group_constraint_types = st.sampled_from(["", "FPR", "FNR", "FPR,FNR"])

# Global constraint type: "", "FPR", "FNR", "FPR,FNR"
global_constraint_types = st.sampled_from(["", "FPR", "FNR", "FPR,FNR"])

# Number of unique groups (1..8)
num_groups_st = st.integers(min_value=1, max_value=8)


def _count_group_constraint_types(constraint_type_str):
    """Count how many group constraint types are active."""
    if constraint_type_str == "FPR,FNR":
        return 2
    elif constraint_type_str in ("FPR", "FNR"):
        return 1
    return 0


def _count_global_constraint_types(global_type_str):
    """Count how many global constraint types are active."""
    if global_type_str == "FPR,FNR":
        return 2
    elif global_type_str in ("FPR", "FNR"):
        return 1
    return 0


@st.composite
def constraint_config(draw):
    """Generate a constraint configuration with expected num_constraints."""
    group_type = draw(group_constraint_types)
    global_type = draw(global_constraint_types)
    num_groups = draw(num_groups_st)

    # Need at least one constraint active for meaningful test
    assume(group_type != "" or global_type != "")

    expected = num_groups * _count_group_constraint_types(
        group_type
    ) + _count_global_constraint_types(global_type)

    return group_type, global_type, num_groups, expected


# ---------------------------------------------------------------------------
# Property 3: Number of constraints matches configuration
# ---------------------------------------------------------------------------


class TestNumConstraintsMatchesConfiguration:
    """
    Feature: fairgbm-dependency-refactor
    Property 3: Number of constraints matches configuration

    For any combination of group/global constraint types and number of
    unique groups, NumConstraints equals:
      (num_groups × group_constraint_types) + global_constraint_types.
    """

    @given(config=constraint_config())
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_num_constraints_cross_entropy_objective(self, config):
        """constrained_cross_entropy reports correct NumConstraints."""
        group_type, global_type, num_groups, expected = config

        # Build minimal training data with exactly num_groups unique groups
        # Need at least one pos and one neg label
        n = max(num_groups * 2, 4)
        labels = np.zeros(n, dtype=np.float32)
        labels[: n // 2] = 1.0
        groups = np.zeros(n, dtype=np.int32)
        for i in range(n):
            groups[i] = i % num_groups

        # Build params string — always explicitly set constraint_type
        # (default in Config is "FPR,FNR", so we must override to NONE
        # when no group constraint is desired)
        parts = [
            "constraint_stepwise_proxy=cross_entropy",
            f"constraint_type={group_type if group_type else 'NONE'}",
        ]
        if global_type:
            parts.append(f"global_constraint_type={global_type}")
            if "FPR" in global_type:
                parts.append("global_target_fpr=0.1")
            if "FNR" in global_type:
                parts.append("global_target_fnr=0.1")
        params_str = " ".join(parts)

        handle = _create_and_init_objective(
            "constrained_cross_entropy", params_str, labels, groups
        )
        try:
            actual = _get_num_constraints(handle)
            assert actual == expected, (
                f"NumConstraints mismatch: "
                f"group_type={group_type!r}, "
                f"global_type={global_type!r}, "
                f"num_groups={num_groups} → "
                f"expected {expected}, got {actual}"
            )
        finally:
            _free_objective(handle)

    @given(config=constraint_config())
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_num_constraints_recall_objective(self, config):
        """constrained_recall_objective reports correct NumConstraints."""
        group_type, global_type, num_groups, expected = config

        # Recall objective requires global FPR constraint
        assume("FPR" in global_type or global_type == "FPR,FNR")

        n = max(num_groups * 2, 4)
        labels = np.zeros(n, dtype=np.float32)
        labels[: n // 2] = 1.0
        groups = np.zeros(n, dtype=np.int32)
        for i in range(n):
            groups[i] = i % num_groups

        parts = [
            "constraint_stepwise_proxy=cross_entropy",
            "objective_stepwise_proxy=cross_entropy",
            "stepwise_proxy_margin=1.0",
            f"constraint_type={group_type if group_type else 'NONE'}",
        ]
        if global_type:
            parts.append(f"global_constraint_type={global_type}")
            if "FPR" in global_type:
                parts.append("global_target_fpr=0.1")
            if "FNR" in global_type:
                parts.append("global_target_fnr=0.1")
        params_str = " ".join(parts)

        handle = _create_and_init_objective(
            "constrained_recall_objective", params_str, labels, groups
        )
        try:
            actual = _get_num_constraints(handle)
            assert actual == expected, (
                f"NumConstraints mismatch: "
                f"group_type={group_type!r}, "
                f"global_type={global_type!r}, "
                f"num_groups={num_groups} → "
                f"expected {expected}, got {actual}"
            )
        finally:
            _free_objective(handle)

    @given(
        num_groups=st.integers(min_value=1, max_value=10),
        proxy=st.sampled_from(["cross_entropy", "hinge", "quadratic"]),
    )
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_combined_fpr_fnr_equals_twice_num_groups(self, num_groups, proxy):
        """FPR,FNR group constraint → NumConstraints == 2 * num_groups."""
        n = max(num_groups * 2, 4)
        labels = np.zeros(n, dtype=np.float32)
        labels[: n // 2] = 1.0
        groups = np.zeros(n, dtype=np.int32)
        for i in range(n):
            groups[i] = i % num_groups

        params_str = f"constraint_type=FPR,FNR " f"constraint_stepwise_proxy={proxy}"

        handle = _create_and_init_objective(
            "constrained_cross_entropy", params_str, labels, groups
        )
        try:
            actual = _get_num_constraints(handle)
            assert actual == 2 * num_groups, (
                f"Expected 2*{num_groups}={2*num_groups}, " f"got {actual}"
            )
        finally:
            _free_objective(handle)

    @given(
        num_groups=st.integers(min_value=1, max_value=10),
    )
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_all_constraints_active(self, num_groups):
        """All group + global constraints → 2*groups + 2."""
        n = max(num_groups * 2, 4)
        labels = np.zeros(n, dtype=np.float32)
        labels[: n // 2] = 1.0
        groups = np.zeros(n, dtype=np.int32)
        for i in range(n):
            groups[i] = i % num_groups

        params_str = (
            "constraint_type=FPR,FNR "
            "global_constraint_type=FPR,FNR "
            "global_target_fpr=0.1 "
            "global_target_fnr=0.1 "
            "constraint_stepwise_proxy=cross_entropy"
        )

        handle = _create_and_init_objective(
            "constrained_cross_entropy", params_str, labels, groups
        )
        try:
            expected = 2 * num_groups + 2
            actual = _get_num_constraints(handle)
            assert actual == expected, f"Expected {expected}, got {actual}"
        finally:
            _free_objective(handle)
