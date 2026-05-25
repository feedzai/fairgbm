"""
Python wrapper around the FairGBM native constrained objective C API.
"""

import ctypes

import numpy as np

from fairgbm._lib import _LIB, _safe_call


def _check_native_extension():
    """Raise ImportError if native extension is not available."""
    if _LIB is None:
        raise ImportError(
            "FairGBM native extension (lib_fairgbm) is not available. "
            "Constrained objectives require the compiled C++ extension. "
            "Build it with: cd native && mkdir -p build && cd build && cmake .. && make"
        )


def _params_to_str(params):
    """Convert params dict to key=value string for C API.

    Parameters
    ----------
    params : dict
        FairGBM configuration parameters.

    Returns
    -------
    bytes
        UTF-8 encoded parameter string.
    """
    parts = []
    for key, value in params.items():
        if value is not None and value != "":
            parts.append(f"{key}={value}")
    return " ".join(parts).encode("utf-8")


class ConstrainedObjective:
    """Python wrapper around the FairGBM native constrained objective.

    Parameters
    ----------
    objective_type : str
        One of "constrained_cross_entropy" or "constrained_recall_objective".
    params : dict
        FairGBM-specific configuration parameters.
    """

    def __init__(self, objective_type, params):
        _check_native_extension()

        self._handle = ctypes.c_void_p()
        params_str = _params_to_str(params)

        _safe_call(
            _LIB.FairGBM_CreateConstrainedObjective(
                objective_type.encode("utf-8"),
                params_str,
                ctypes.byref(self._handle),
            )
        )
        self._num_data = 0
        self._initialized = False

    def init(self, labels, constraint_groups, weights=None):
        """Initialize the objective with training data metadata.

        Parameters
        ----------
        labels : numpy.ndarray
            Binary labels array (values in {0, 1}), shape (num_data,).
        constraint_groups : numpy.ndarray
            Integer constraint group array, shape (num_data,).
        weights : numpy.ndarray or None
            Per-instance weights, shape (num_data,). None for uniform weights.
        """
        labels = np.ascontiguousarray(labels, dtype=np.float32)
        constraint_groups = np.ascontiguousarray(constraint_groups, dtype=np.int32)
        num_data = len(labels)

        labels_ptr = labels.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        groups_ptr = constraint_groups.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        if weights is not None:
            weights = np.ascontiguousarray(weights, dtype=np.float32)
            weights_ptr = weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
            weights_ptr = ctypes.POINTER(ctypes.c_float)()  # null pointer

        _safe_call(
            _LIB.FairGBM_ObjectiveInit(
                self._handle,
                labels_ptr,
                groups_ptr,
                weights_ptr,
                ctypes.c_int(num_data),
            )
        )
        self._num_data = num_data
        self._initialized = True

    def get_gradients(self, scores):
        """Compute predictive loss gradients (without constraint terms).

        Parameters
        ----------
        scores : numpy.ndarray
            Raw model scores, shape (num_data,).

        Returns
        -------
        tuple of numpy.ndarray
            (gradients, hessians), each shape (num_data,) with dtype float32.
        """
        scores = np.ascontiguousarray(scores, dtype=np.float64)
        gradients = np.empty(self._num_data, dtype=np.float32)
        hessians = np.empty(self._num_data, dtype=np.float32)

        scores_ptr = scores.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        grad_ptr = gradients.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        hess_ptr = hessians.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        _safe_call(
            _LIB.FairGBM_GetGradients(self._handle, scores_ptr, grad_ptr, hess_ptr)
        )
        return gradients, hessians

    def get_constraint_gradients(self, multipliers, scores, gradients, hessians):
        """Add constraint gradient contributions in-place.

        Parameters
        ----------
        multipliers : numpy.ndarray
            Current Lagrangian multiplier values, shape (num_constraints,).
        scores : numpy.ndarray
            Raw model scores, shape (num_data,).
        gradients : numpy.ndarray
            Gradients array to modify in-place, shape (num_data,), dtype float32.
        hessians : numpy.ndarray
            Hessians array to modify in-place, shape (num_data,), dtype float32.
        """
        multipliers = np.ascontiguousarray(multipliers, dtype=np.float64)
        scores = np.ascontiguousarray(scores, dtype=np.float64)
        # gradients/hessians modified in-place — must already be float32 contiguous
        if not gradients.flags["C_CONTIGUOUS"] or gradients.dtype != np.float32:
            raise ValueError("gradients must be contiguous float32 array")
        if not hessians.flags["C_CONTIGUOUS"] or hessians.dtype != np.float32:
            raise ValueError("hessians must be contiguous float32 array")

        mult_ptr = multipliers.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        scores_ptr = scores.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        grad_ptr = gradients.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        hess_ptr = hessians.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        _safe_call(
            _LIB.FairGBM_GetConstraintGradients(
                self._handle, mult_ptr, scores_ptr, grad_ptr, hess_ptr
            )
        )

    def get_multiplier_updates(self, scores):
        """Compute constraint violation values for Lagrangian multiplier update.

        Parameters
        ----------
        scores : numpy.ndarray
            Raw model scores, shape (num_data,).

        Returns
        -------
        numpy.ndarray
            Constraint violation values, shape (num_constraints,).
        """
        scores = np.ascontiguousarray(scores, dtype=np.float64)
        num_constraints = self.num_constraints

        constraint_values = np.empty(num_constraints, dtype=np.float64)
        out_n = ctypes.c_int(0)

        scores_ptr = scores.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        values_ptr = constraint_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        _safe_call(
            _LIB.FairGBM_GetLagrangianGradientsWRTMultipliers(
                self._handle, scores_ptr, values_ptr, ctypes.byref(out_n)
            )
        )
        return constraint_values

    @property
    def num_constraints(self):
        """Get the number of constraints for the current configuration.

        Returns
        -------
        int
            Number of constraints.
        """
        out = ctypes.c_int(0)
        _safe_call(_LIB.FairGBM_GetNumConstraints(self._handle, ctypes.byref(out)))
        return out.value

    @property
    def boost_from_score(self):
        """Get the initial score (boost from average).

        Returns
        -------
        float
            Initial score value.
        """
        out = ctypes.c_double(0.0)
        _safe_call(_LIB.FairGBM_BoostFromScore(self._handle, ctypes.byref(out)))
        return out.value

    def __del__(self):
        """Free the native handle."""
        if hasattr(self, "_handle") and self._handle and _LIB is not None:
            _LIB.FairGBM_FreeConstrainedObjective(self._handle)
            self._handle = None
