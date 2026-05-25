"""
FairGBM native extension loader and ctypes bindings.

Loads lib_fairgbm shared library and exposes C API functions
with proper ctypes signatures.
"""

import ctypes
import sys
from pathlib import Path


def _find_fairgbm_lib():
    """Locate the lib_fairgbm shared library.

    Search order:
      1. Package directory (same dir as this file)
      2. Package parent directory (for development installs)
      3. native/build/ directory (for development builds)
      4. Common system paths

    Returns
    -------
    str
        Path to the shared library.

    Raises
    ------
    OSError
        If the library cannot be found.
    """
    if sys.platform == "win32":
        lib_names = ["lib_fairgbm.dll", "fairgbm.dll"]
    elif sys.platform == "darwin":
        lib_names = ["lib_fairgbm.dylib", "liblib_fairgbm.dylib"]
    else:
        lib_names = ["lib_fairgbm.so", "liblib_fairgbm.so"]

    search_dirs = [
        # Package directory
        Path(__file__).parent,
        # Package parent (repo root for dev installs)
        Path(__file__).parent.parent,
        # Native build directory (CMake default)
        Path(__file__).parent.parent / "native" / "build",
        # Common install paths
        Path("/usr/local/lib"),
        Path("/usr/lib"),
    ]

    searched_paths = []
    for directory in search_dirs:
        for lib_name in lib_names:
            lib_path = directory / lib_name
            searched_paths.append(str(lib_path))
            if lib_path.exists():
                return str(lib_path)

    raise OSError(
        "Cannot find FairGBM native extension. Searched:\n"
        + "\n".join(f"  - {p}" for p in searched_paths)
        + "\n\nEnsure the native extension is compiled. "
        "Run: cd native && mkdir build && cd build && cmake .. && make"
    )


def _load_lib():
    """Load the FairGBM native extension and set up function signatures.

    Returns
    -------
    ctypes.CDLL
        The loaded shared library with configured argtypes/restype.

    Raises
    ------
    RuntimeError
        If the library cannot be loaded.
    """
    try:
        lib_path = _find_fairgbm_lib()
    except OSError as e:
        raise RuntimeError(str(e)) from e

    try:
        lib = ctypes.cdll.LoadLibrary(lib_path)
    except OSError as e:
        raise RuntimeError(
            f"Found FairGBM library at {lib_path} but failed to load it: {e}"
        ) from e

    # --- Set up function signatures ---

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
    lib.FairGBM_FreeConstrainedObjective.argtypes = [
        ctypes.c_void_p,  # handle
    ]

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

    # FairGBM_GetConstraintGradients
    lib.FairGBM_GetConstraintGradients.restype = ctypes.c_int
    lib.FairGBM_GetConstraintGradients.argtypes = [
        ctypes.c_void_p,  # handle
        ctypes.POINTER(ctypes.c_double),  # lagrangian_multipliers
        ctypes.POINTER(ctypes.c_double),  # scores
        ctypes.POINTER(ctypes.c_float),  # inout_gradients
        ctypes.POINTER(ctypes.c_float),  # inout_hessians
    ]

    # FairGBM_GetLagrangianGradientsWRTMultipliers
    lib.FairGBM_GetLagrangianGradientsWRTMultipliers.restype = ctypes.c_int
    lib.FairGBM_GetLagrangianGradientsWRTMultipliers.argtypes = [
        ctypes.c_void_p,  # handle
        ctypes.POINTER(ctypes.c_double),  # scores
        ctypes.POINTER(ctypes.c_double),  # out_constraint_values
        ctypes.POINTER(ctypes.c_int),  # out_num_constraints
    ]

    # FairGBM_GetNumConstraints
    lib.FairGBM_GetNumConstraints.restype = ctypes.c_int
    lib.FairGBM_GetNumConstraints.argtypes = [
        ctypes.c_void_p,  # handle
        ctypes.POINTER(ctypes.c_int),  # out_num_constraints
    ]

    # FairGBM_BoostFromScore
    lib.FairGBM_BoostFromScore.restype = ctypes.c_int
    lib.FairGBM_BoostFromScore.argtypes = [
        ctypes.c_void_p,  # handle
        ctypes.POINTER(ctypes.c_double),  # out_score
    ]

    return lib


def _safe_call(ret):
    """Check C API return code and raise on error.

    Parameters
    ----------
    ret : int
        Return code from a C API function (0 = success, -1 = error).

    Raises
    ------
    RuntimeError
        If ret != 0, with the last error message from the native extension.
    """
    if ret != 0:
        err_msg = _LIB.FairGBM_GetLastError()
        if err_msg:
            msg = err_msg.decode("utf-8", errors="replace")
        else:
            msg = "Unknown error in FairGBM native extension"
        raise RuntimeError(f"FairGBM C API error: {msg}")


# Module-level library handle.
# Set to None if loading fails — constrained objectives will raise ImportError.
try:
    _LIB = _load_lib()
except RuntimeError:
    _LIB = None
