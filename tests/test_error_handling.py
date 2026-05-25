"""Unit tests for error handling and graceful degradation.

Tests that:
- ImportError raised when native extension missing and constrained objective used
- ValueError for invalid constraint_type
- ValueError for invalid constraint_stepwise_proxy
- ValueError for mismatched init_lagrangian_multipliers length
- Non-constrained training works without native extension
"""

import importlib
import sys
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from fairgbm.params import split_params, _is_constrained, _validate_fairgbm_params


# --- Parameter validation tests ---


class TestConstraintTypeValidation:
    """ValueError for invalid constraint_type values."""

    @pytest.mark.parametrize("valid", ["FPR", "FNR", "FPR,FNR", ""])
    def test_valid_constraint_types_accepted(self, valid):
        _validate_fairgbm_params({"constraint_type": valid})

    @pytest.mark.parametrize("invalid", ["fpr", "TPR", "FPR FNR", "FPR,", "INVALID"])
    def test_invalid_constraint_type_raises(self, invalid):
        with pytest.raises(ValueError, match="Invalid constraint_type"):
            _validate_fairgbm_params({"constraint_type": invalid})

    @pytest.mark.parametrize("invalid", ["fpr", "TPR", "INVALID"])
    def test_invalid_global_constraint_type_raises(self, invalid):
        with pytest.raises(ValueError, match="Invalid global_constraint_type"):
            _validate_fairgbm_params({"global_constraint_type": invalid})


class TestProxyTypeValidation:
    """ValueError for invalid proxy type values."""

    @pytest.mark.parametrize("valid", ["cross_entropy", "hinge", "quadratic", ""])
    def test_valid_proxy_types_accepted(self, valid):
        _validate_fairgbm_params({"constraint_stepwise_proxy": valid})

    @pytest.mark.parametrize("invalid", ["sigmoid", "linear", "HINGE", "invalid"])
    def test_invalid_constraint_proxy_raises(self, invalid):
        with pytest.raises(ValueError, match="Invalid constraint_stepwise_proxy"):
            _validate_fairgbm_params({"constraint_stepwise_proxy": invalid})

    @pytest.mark.parametrize("invalid", ["sigmoid", "linear", "QUADRATIC"])
    def test_invalid_objective_proxy_raises(self, invalid):
        with pytest.raises(ValueError, match="Invalid objective_stepwise_proxy"):
            _validate_fairgbm_params({"objective_stepwise_proxy": invalid})


# --- Native extension missing tests ---


class TestMissingNativeExtension:
    """ImportError when native extension missing and constrained objective used."""

    def test_constrained_objective_raises_importerror_without_lib(self):
        """ConstrainedObjective raises ImportError when _LIB is None."""
        with patch("fairgbm.constrained._LIB", None):
            from fairgbm.constrained import ConstrainedObjective

            with pytest.raises(ImportError, match="native extension"):
                ConstrainedObjective("constrained_cross_entropy", {})

    def test_check_native_extension_raises_with_instructions(self):
        """_check_native_extension provides build instructions."""
        with patch("fairgbm.constrained._LIB", None):
            from fairgbm.constrained import _check_native_extension

            with pytest.raises(ImportError, match="cmake"):
                _check_native_extension()


# --- Multiplier length mismatch tests ---


class TestMultiplierLengthValidation:
    """ValueError for mismatched init_lagrangian_multipliers length."""

    def test_mismatched_multiplier_length_raises(self):
        """Training raises ValueError when multiplier count != num_constraints."""
        from fairgbm.training import train

        # Create a mock dataset
        mock_dataset = MagicMock()
        mock_dataset.get_label.return_value = np.array([0, 1, 0, 1], dtype=np.float32)
        mock_dataset.get_weight.return_value = None

        # Mock ConstrainedObjective to control num_constraints
        mock_obj = MagicMock()
        mock_obj.num_constraints = 2

        with patch("fairgbm.training.ConstrainedObjective", return_value=mock_obj):
            params = {
                "constraint_type": "FPR",
                "init_lagrangian_multipliers": [0.1, 0.2, 0.3],  # 3 != 2
            }
            with pytest.raises(ValueError, match="init_lagrangian_multipliers length"):
                train(
                    params,
                    mock_dataset,
                    num_boost_round=1,
                    constraint_group=np.array([0, 0, 1, 1], dtype=np.int32),
                )


# --- Non-constrained works without native extension ---


class TestNonConstrainedWithoutExtension:
    """Non-constrained training works regardless of native extension."""

    def test_non_constrained_delegates_to_lightgbm(self):
        """When no constraints set, training delegates to lightgbm.train directly."""
        mock_dataset = MagicMock()
        mock_booster = MagicMock()

        with patch("fairgbm.training.lightgbm") as mock_lgbm:
            mock_lgbm.train.return_value = mock_booster

            from fairgbm.training import train

            # No constraint_type or global_constraint_type
            params = {"num_leaves": 31, "learning_rate": 0.1}
            result = train(params, mock_dataset, num_boost_round=10)

            assert result == mock_booster
            mock_lgbm.train.assert_called_once()

    def test_non_constrained_ignores_native_extension(self):
        """Non-constrained path never touches ConstrainedObjective."""
        mock_dataset = MagicMock()

        with patch("fairgbm.training.lightgbm") as mock_lgbm, patch(
            "fairgbm.training.ConstrainedObjective"
        ) as mock_cls:
            mock_lgbm.train.return_value = MagicMock()

            from fairgbm.training import train

            params = {"num_leaves": 31}
            train(params, mock_dataset, num_boost_round=5)

            mock_cls.assert_not_called()
