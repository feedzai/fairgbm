"""FairGBM — Fairness-constrained Gradient Boosting.

Provides fairness-constrained gradient boosting by extending LightGBM
with Lagrangian-based optimization for group-wise and global FPR/FNR constraints.
"""

from pathlib import Path

# Re-export core LightGBM classes so users can import from fairgbm namespace
from lightgbm import Dataset, Booster, LGBMClassifier, LGBMRegressor, LGBMRanker, cv

# FairGBM-specific exports
from fairgbm.sklearn import FairGBMClassifier
from fairgbm.training import train

# Native extension availability check
from fairgbm._lib import _LIB  # noqa: E402

# Version from VERSION.txt
_VERSION_FILE = Path(__file__).parent.parent / "VERSION.txt"
if _VERSION_FILE.exists():
    __version__ = _VERSION_FILE.read_text().strip()
else:
    __version__ = "0.0.0"

# Flag indicating whether the native extension loaded successfully.
# If False, constrained objectives will raise ImportError when used.
NATIVE_EXTENSION_AVAILABLE = _LIB is not None

__all__ = [
    # LightGBM re-exports
    "Dataset",
    "Booster",
    "LGBMClassifier",
    "LGBMRegressor",
    "LGBMRanker",
    "cv",
    # FairGBM
    "FairGBMClassifier",
    "train",
    # Metadata
    "__version__",
    "NATIVE_EXTENSION_AVAILABLE",
]
