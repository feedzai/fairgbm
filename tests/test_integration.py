# coding: utf-8
"""
End-to-end integration tests for FairGBM.

Validates: Requirements 4.1, 4.2, 4.3, 5.1, 7.1

Tests train FairGBMClassifier on small synthetic datasets with various
constraint configurations and verify predictions are valid.
"""

import numpy as np
import pytest

from fairgbm.sklearn import FairGBMClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _native_extension_available():
    """Check if the native extension is loadable."""
    try:
        from fairgbm._lib import _LIB

        return _LIB is not None
    except (ImportError, OSError, RuntimeError):
        return False


skip_no_native = pytest.mark.skipif(
    not _native_extension_available(),
    reason="lib_fairgbm native extension not available",
)


def _make_synthetic_data(n_samples=200, n_features=5, n_groups=3, seed=42):
    """Generate a small synthetic binary classification dataset with groups.

    Returns (X, y, constraint_group) where:
      - X: (n_samples, n_features) float array
      - y: (n_samples,) binary labels
      - constraint_group: (n_samples,) int32 group assignments
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    # Make labels correlated with first feature
    logits = X[:, 0] + 0.5 * X[:, 1]
    y = (logits > 0).astype(np.int32)
    constraint_group = rng.randint(0, n_groups, size=n_samples).astype(np.int32)
    return X, y, constraint_group


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@skip_no_native
class TestFPRConstraint:
    """Train FairGBMClassifier with group-wise FPR constraint."""

    def test_train_with_fpr_constraint(self):
        """Model trains successfully with FPR constraint."""
        X, y, groups = _make_synthetic_data()

        clf = FairGBMClassifier(
            constraint_type="FPR",
            multiplier_learning_rate=0.1,
            n_estimators=10,
            num_leaves=8,
            verbose=-1,
        )
        clf.fit(X, y, constraint_group=groups)

        preds = clf.predict(X)
        assert preds.shape == (len(X),)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_fpr_constraint_probabilities_valid(self):
        """Predicted probabilities are in [0, 1]."""
        X, y, groups = _make_synthetic_data()

        clf = FairGBMClassifier(
            constraint_type="FPR",
            n_estimators=10,
            num_leaves=8,
            verbose=-1,
        )
        clf.fit(X, y, constraint_group=groups)

        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


@skip_no_native
class TestFNRConstraint:
    """Train FairGBMClassifier with group-wise FNR constraint."""

    def test_train_with_fnr_constraint(self):
        """Model trains successfully with FNR constraint."""
        X, y, groups = _make_synthetic_data()

        clf = FairGBMClassifier(
            constraint_type="FNR",
            multiplier_learning_rate=0.1,
            n_estimators=10,
            num_leaves=8,
            verbose=-1,
        )
        clf.fit(X, y, constraint_group=groups)

        preds = clf.predict(X)
        assert preds.shape == (len(X),)

    def test_fnr_constraint_probabilities_valid(self):
        """Predicted probabilities are in [0, 1]."""
        X, y, groups = _make_synthetic_data()

        clf = FairGBMClassifier(
            constraint_type="FNR",
            n_estimators=10,
            num_leaves=8,
            verbose=-1,
        )
        clf.fit(X, y, constraint_group=groups)

        proba = clf.predict_proba(X)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)


@skip_no_native
class TestCombinedFPRFNRConstraint:
    """Train FairGBMClassifier with combined FPR+FNR constraints."""

    def test_train_with_combined_constraints(self):
        """Model trains successfully with FPR,FNR constraint."""
        X, y, groups = _make_synthetic_data()

        clf = FairGBMClassifier(
            constraint_type="FPR,FNR",
            multiplier_learning_rate=0.1,
            n_estimators=10,
            num_leaves=8,
            verbose=-1,
        )
        clf.fit(X, y, constraint_group=groups)

        preds = clf.predict(X)
        assert preds.shape == (len(X),)

    def test_combined_constraints_probabilities_valid(self):
        """Predicted probabilities are in [0, 1]."""
        X, y, groups = _make_synthetic_data()

        clf = FairGBMClassifier(
            constraint_type="FPR,FNR",
            n_estimators=10,
            num_leaves=8,
            verbose=-1,
        )
        clf.fit(X, y, constraint_group=groups)

        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)


@skip_no_native
class TestGlobalFPRConstraint:
    """Train FairGBMClassifier with global FPR constraint."""

    def test_train_with_global_fpr(self):
        """Model trains successfully with global FPR constraint."""
        X, y, groups = _make_synthetic_data()

        clf = FairGBMClassifier(
            global_constraint_type="FPR",
            global_target_fpr=0.1,
            multiplier_learning_rate=0.1,
            n_estimators=10,
            num_leaves=8,
            verbose=-1,
        )
        clf.fit(X, y, constraint_group=groups)

        preds = clf.predict(X)
        assert preds.shape == (len(X),)

    def test_global_fpr_probabilities_valid(self):
        """Predicted probabilities are in [0, 1]."""
        X, y, groups = _make_synthetic_data()

        clf = FairGBMClassifier(
            global_constraint_type="FPR",
            global_target_fpr=0.1,
            n_estimators=10,
            num_leaves=8,
            verbose=-1,
        )
        clf.fit(X, y, constraint_group=groups)

        proba = clf.predict_proba(X)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)


@skip_no_native
class TestNonConstrainedDelegation:
    """Non-constrained FairGBMClassifier delegates to LightGBM."""

    def test_non_constrained_trains_successfully(self):
        """FairGBMClassifier with no constraints trains like LightGBM."""
        X, y, _ = _make_synthetic_data()

        clf = FairGBMClassifier(
            constraint_type="",
            global_constraint_type="",
            objective="binary",
            n_estimators=10,
            num_leaves=8,
            verbose=-1,
        )
        clf.fit(X, y)

        preds = clf.predict(X)
        assert preds.shape == (len(X),)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_non_constrained_probabilities_valid(self):
        """Non-constrained predictions are valid probabilities."""
        X, y, _ = _make_synthetic_data()

        clf = FairGBMClassifier(
            constraint_type="",
            global_constraint_type="",
            objective="binary",
            n_estimators=10,
            num_leaves=8,
            verbose=-1,
        )
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


@skip_no_native
class TestProxyLossVariants:
    """Test training with different proxy loss configurations."""

    @pytest.mark.parametrize("proxy", ["cross_entropy", "hinge", "quadratic"])
    def test_train_with_proxy(self, proxy):
        """Training succeeds with each proxy loss type."""
        X, y, groups = _make_synthetic_data()

        clf = FairGBMClassifier(
            constraint_type="FPR",
            constraint_stepwise_proxy=proxy,
            n_estimators=10,
            num_leaves=8,
            verbose=-1,
        )
        clf.fit(X, y, constraint_group=groups)

        proba = clf.predict_proba(X)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)


@skip_no_native
class TestPredictionValidity:
    """Verify predictions are valid across configurations."""

    def test_predict_proba_sums_to_one(self):
        """predict_proba columns sum to 1 for constrained model."""
        X, y, groups = _make_synthetic_data()

        clf = FairGBMClassifier(
            constraint_type="FPR",
            n_estimators=20,
            num_leaves=8,
            verbose=-1,
        )
        clf.fit(X, y, constraint_group=groups)

        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_consistent_with_predict_proba(self):
        """predict() labels match argmax of predict_proba()."""
        X, y, groups = _make_synthetic_data()

        clf = FairGBMClassifier(
            constraint_type="FPR",
            n_estimators=20,
            num_leaves=8,
            verbose=-1,
        )
        clf.fit(X, y, constraint_group=groups)

        preds = clf.predict(X)
        proba = clf.predict_proba(X)
        expected = np.argmax(proba, axis=1)
        np.testing.assert_array_equal(preds, expected)
