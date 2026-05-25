# coding: utf-8
"""
Unit tests for FairGBMClassifier sklearn API.

Validates: Requirements 7.1, 7.2, 7.3
"""

import sys
import os

import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lightgbm import LGBMClassifier

from fairgbm.sklearn import FairGBMClassifier


class TestFairGBMClassifierSubclass:
    """Test that FairGBMClassifier is a proper LGBMClassifier subclass."""

    def test_is_subclass_of_lgbm_classifier(self):
        """FairGBMClassifier extends LGBMClassifier."""
        assert issubclass(FairGBMClassifier, LGBMClassifier)

    def test_instance_is_lgbm_classifier(self):
        """Instance passes isinstance check."""
        clf = FairGBMClassifier()
        assert isinstance(clf, LGBMClassifier)


class TestDefaultObjective:
    """Test default objective is constrained_cross_entropy."""

    def test_default_objective(self):
        """Default objective is constrained_cross_entropy when not specified."""
        clf = FairGBMClassifier()
        assert clf.objective == "constrained_cross_entropy"

    def test_custom_objective_preserved(self):
        """Explicit objective overrides default."""
        clf = FairGBMClassifier(objective="binary")
        assert clf.objective == "binary"


class TestFairGBMParams:
    """Test FairGBM-specific parameter handling."""

    def test_default_fairgbm_params(self):
        """Default FairGBM params are set correctly."""
        clf = FairGBMClassifier()
        assert clf.constraint_type == ""
        assert clf.multiplier_learning_rate == 0.1
        assert clf.constraint_stepwise_proxy == "cross_entropy"
        assert clf.objective_stepwise_proxy == ""
        assert clf.stepwise_proxy_margin == 1.0
        assert clf.constraint_fpr_tolerance == 0.0
        assert clf.constraint_fnr_tolerance == 0.0
        assert clf.score_threshold == 0.5
        assert clf.global_constraint_type == ""
        assert clf.global_target_fpr == 0.0
        assert clf.global_target_fnr == 0.0
        assert clf.global_score_threshold == 0.5
        assert clf.init_lagrangian_multipliers == []

    def test_custom_fairgbm_params(self):
        """Custom FairGBM params stored correctly."""
        clf = FairGBMClassifier(
            constraint_type="FPR",
            multiplier_learning_rate=0.5,
            global_constraint_type="FNR",
            global_target_fnr=0.1,
        )
        assert clf.constraint_type == "FPR"
        assert clf.multiplier_learning_rate == 0.5
        assert clf.global_constraint_type == "FNR"
        assert clf.global_target_fnr == 0.1

    def test_get_params_includes_fairgbm(self):
        """get_params() returns FairGBM-specific params."""
        clf = FairGBMClassifier(constraint_type="FPR")
        params = clf.get_params()
        assert "constraint_type" in params
        assert params["constraint_type"] == "FPR"
        assert "multiplier_learning_rate" in params

    def test_get_params_includes_lgbm(self):
        """get_params() also returns LGBMClassifier params."""
        clf = FairGBMClassifier(num_leaves=50)
        params = clf.get_params()
        assert params["num_leaves"] == 50


class TestFitAcceptsConstraintGroup:
    """Test that fit() accepts constraint_group parameter."""

    @patch("fairgbm.sklearn.fairgbm_training.train")
    def test_fit_with_constraint_group(self, mock_train):
        """fit() accepts and passes constraint_group to training."""
        # Set up mock booster
        mock_booster = MagicMock()
        mock_booster.num_feature.return_value = 5
        mock_booster.best_iteration = 10
        mock_booster.best_score = {}
        mock_train.return_value = mock_booster

        clf = FairGBMClassifier(constraint_type="FPR")
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        groups = np.random.randint(0, 3, 100)

        clf.fit(X, y, constraint_group=groups)

        # Verify train was called with constraint_group
        call_kwargs = mock_train.call_args[1]
        assert call_kwargs["constraint_group"] is not None
        np.testing.assert_array_equal(
            call_kwargs["constraint_group"], groups.astype(np.int32)
        )

    @patch("fairgbm.sklearn.fairgbm_training.train")
    def test_fit_without_constraint_group(self, mock_train):
        """fit() works without constraint_group (defaults to None)."""
        mock_booster = MagicMock()
        mock_booster.num_feature.return_value = 5
        mock_booster.best_iteration = 10
        mock_booster.best_score = {}
        mock_train.return_value = mock_booster

        clf = FairGBMClassifier()
        X = np.random.rand(50, 3)
        y = np.random.randint(0, 2, 50)

        clf.fit(X, y)

        call_kwargs = mock_train.call_args[1]
        assert call_kwargs["constraint_group"] is None


class TestPredictDelegation:
    """Test predict() and predict_proba() delegate to booster."""

    @patch("fairgbm.sklearn.fairgbm_training.train")
    def test_predict_returns_correct_shape(self, mock_train):
        """predict() returns array of shape (n_samples,)."""
        mock_booster = MagicMock()
        mock_booster.num_feature.return_value = 5
        mock_booster.best_iteration = 10
        mock_booster.best_score = {}
        # For binary classification, booster.predict returns probabilities
        mock_booster.predict.return_value = np.random.rand(20)
        mock_train.return_value = mock_booster

        clf = FairGBMClassifier()
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, 100)
        clf.fit(X_train, y_train)

        X_test = np.random.rand(20, 5)
        preds = clf.predict(X_test)
        assert preds.shape == (20,)

    @patch("fairgbm.sklearn.fairgbm_training.train")
    def test_predict_proba_returns_correct_shape(self, mock_train):
        """predict_proba() returns array of shape (n_samples, n_classes)."""
        mock_booster = MagicMock()
        mock_booster.num_feature.return_value = 5
        mock_booster.best_iteration = 10
        mock_booster.best_score = {}
        # For binary classification, booster.predict returns probabilities
        mock_booster.predict.return_value = np.random.rand(20)
        mock_train.return_value = mock_booster

        clf = FairGBMClassifier()
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, 100)
        clf.fit(X_train, y_train)

        X_test = np.random.rand(20, 5)
        proba = clf.predict_proba(X_test)
        assert proba.shape == (20, 2)


class TestSklearnCompatibility:
    """Test sklearn interface compatibility."""

    def test_clone(self):
        """FairGBMClassifier can be cloned via sklearn."""
        from sklearn.base import clone

        clf = FairGBMClassifier(
            constraint_type="FPR",
            multiplier_learning_rate=0.5,
            num_leaves=50,
        )
        clf2 = clone(clf)
        assert clf2.constraint_type == "FPR"
        assert clf2.multiplier_learning_rate == 0.5
        assert clf2.num_leaves == 50
        assert clf2 is not clf
