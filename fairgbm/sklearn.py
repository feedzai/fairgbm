# coding: utf-8
"""Scikit-learn compatible FairGBM classifier wrapper."""

import numpy as np
from lightgbm import LGBMClassifier

from fairgbm import training as fairgbm_training
from fairgbm.params import FAIRGBM_PARAMS


class FairGBMClassifier(LGBMClassifier):
    """Fairness-constrained gradient boosting classifier.

    Extends LightGBM's LGBMClassifier with Lagrangian-based fairness
    constraints. When constraint parameters are set, training uses a
    custom objective that enforces group-wise or global FPR/FNR constraints
    via Lagrangian multiplier updates.

    Parameters
    ----------
    constraint_type : str, optional (default='')
        Group constraint type: "FPR", "FNR", or "FPR,FNR".
    multiplier_learning_rate : float, optional (default=0.1)
        Learning rate for Lagrangian multiplier updates.
    constraint_stepwise_proxy : str, optional (default='cross_entropy')
        Proxy function for constraints: "cross_entropy", "hinge", or "quadratic".
    objective_stepwise_proxy : str, optional (default='')
        Proxy function for objective (recall objective only).
    stepwise_proxy_margin : float, optional (default=1.0)
        Margin parameter for proxy functions.
    constraint_fpr_tolerance : float, optional (default=0.0)
        Tolerance for FPR constraints.
    constraint_fnr_tolerance : float, optional (default=0.0)
        Tolerance for FNR constraints.
    score_threshold : float, optional (default=0.5)
        Probability threshold for confusion matrix computation.
    global_constraint_type : str, optional (default='')
        Global constraint type: "FPR", "FNR", or "FPR,FNR".
    global_target_fpr : float, optional (default=0.0)
        Target global FPR.
    global_target_fnr : float, optional (default=0.0)
        Target global FNR.
    global_score_threshold : float, optional (default=0.5)
        Threshold for global constraints.
    init_lagrangian_multipliers : list of float, optional (default=[])
        Initial Lagrangian multiplier values.
    **kwargs
        Additional parameters passed to LGBMClassifier.
    """

    def __init__(
        self,
        *,
        constraint_type="",
        multiplier_learning_rate=0.1,
        constraint_stepwise_proxy="cross_entropy",
        objective_stepwise_proxy="",
        stepwise_proxy_margin=1.0,
        constraint_fpr_tolerance=0.0,
        constraint_fnr_tolerance=0.0,
        score_threshold=0.5,
        global_constraint_type="",
        global_target_fpr=0.0,
        global_target_fnr=0.0,
        global_score_threshold=0.5,
        init_lagrangian_multipliers=None,
        # LGBMClassifier params
        boosting_type="gbdt",
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.1,
        n_estimators=100,
        subsample_for_bin=200000,
        objective=None,
        class_weight=None,
        min_split_gain=0.0,
        min_child_weight=1e-3,
        min_child_samples=20,
        subsample=1.0,
        subsample_freq=0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=None,
        n_jobs=None,
        importance_type="split",
        **kwargs,
    ):
        # Default objective to constrained_cross_entropy
        if objective is None:
            objective = "constrained_cross_entropy"

        # Store FairGBM-specific params
        self.constraint_type = constraint_type
        self.multiplier_learning_rate = multiplier_learning_rate
        self.constraint_stepwise_proxy = constraint_stepwise_proxy
        self.objective_stepwise_proxy = objective_stepwise_proxy
        self.stepwise_proxy_margin = stepwise_proxy_margin
        self.constraint_fpr_tolerance = constraint_fpr_tolerance
        self.constraint_fnr_tolerance = constraint_fnr_tolerance
        self.score_threshold = score_threshold
        self.global_constraint_type = global_constraint_type
        self.global_target_fpr = global_target_fpr
        self.global_target_fnr = global_target_fnr
        self.global_score_threshold = global_score_threshold
        self.init_lagrangian_multipliers = (
            init_lagrangian_multipliers
            if init_lagrangian_multipliers is not None
            else []
        )

        super().__init__(
            boosting_type=boosting_type,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample_for_bin=subsample_for_bin,
            objective=objective,
            class_weight=class_weight,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            min_child_samples=min_child_samples,
            subsample=subsample,
            subsample_freq=subsample_freq,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            importance_type=importance_type,
            **kwargs,
        )

    def fit(
        self,
        X,
        y,
        constraint_group=None,
        sample_weight=None,
        init_score=None,
        eval_set=None,
        eval_names=None,
        eval_sample_weight=None,
        eval_class_weight=None,
        eval_init_score=None,
        eval_metric=None,
        feature_name="auto",
        categorical_feature="auto",
        callbacks=None,
        init_model=None,
    ):
        """Fit the FairGBM classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target values (binary: 0 or 1).
        constraint_group : array-like of shape (n_samples,) or None
            Integer array identifying sensitive group membership per instance.
            If None, all instances belong to group 0.
        sample_weight : array-like of shape (n_samples,) or None
            Sample weights.
        init_score : array-like or None
            Initial score for training data.
        eval_set : list of (X, y) tuples or None
            Validation sets.
        eval_names : list of str or None
            Names for validation sets.
        eval_sample_weight : list of arrays or None
            Weights for eval data.
        eval_class_weight : list or None
            Class weights for eval data.
        eval_init_score : list of arrays or None
            Init scores for eval data.
        eval_metric : str, callable, list, or None
            Evaluation metric(s).
        feature_name : list of str or 'auto'
            Feature names.
        categorical_feature : list or 'auto'
            Categorical features.
        callbacks : list of callables or None
            Callback functions.
        init_model : str, Booster, or None
            Model for continued training.

        Returns
        -------
        self
        """
        import copy
        from lightgbm import Dataset
        from lightgbm.callback import record_evaluation

        # Store constraint group for later use
        self._constraint_group = (
            np.ascontiguousarray(constraint_group, dtype=np.int32)
            if constraint_group is not None
            else None
        )

        # Build combined params dict (LightGBM + FairGBM)
        params = self.get_params()

        # Separate FairGBM params from the dict that will go to LightGBM
        fairgbm_params_dict = {}
        lgbm_params_dict = {}
        for key, value in params.items():
            if key in FAIRGBM_PARAMS:
                fairgbm_params_dict[key] = value
            else:
                lgbm_params_dict[key] = value

        # Merge back into a single dict for fairgbm.training.train()
        combined_params = {**lgbm_params_dict, **fairgbm_params_dict}

        # Handle objective: map sklearn-style params
        if (
            "objective" not in combined_params
            or combined_params.get("objective") is None
        ):
            combined_params["objective"] = "constrained_cross_entropy"

        # Handle n_estimators -> num_boost_round
        num_boost_round = combined_params.pop("n_estimators", self.n_estimators)

        # Remove sklearn-specific params that LightGBM doesn't understand
        for sklearn_key in ("importance_type", "class_weight"):
            combined_params.pop(sklearn_key, None)

        # Handle class weights
        if self.class_weight is not None:
            from lightgbm.compat import _LGBMComputeSampleWeight

            class_sample_weight = _LGBMComputeSampleWeight(self.class_weight, y)
            if sample_weight is None:
                sample_weight = class_sample_weight
            else:
                sample_weight = np.multiply(sample_weight, class_sample_weight)

        # Encode labels for classification
        from lightgbm.compat import _LGBMLabelEncoder

        self._le = _LGBMLabelEncoder().fit(y)
        _y = self._le.transform(y)
        self._classes = self._le.classes_
        self._n_classes = len(self._classes)

        # Build training Dataset
        train_set = Dataset(
            data=X,
            label=_y,
            weight=sample_weight,
            init_score=init_score,
            categorical_feature=categorical_feature,
            feature_name=feature_name,
            params=combined_params,
        )

        # Build validation sets
        valid_sets = []
        if eval_set is not None:
            if isinstance(eval_set, tuple):
                eval_set = [eval_set]
            for i, valid_data in enumerate(eval_set):
                valid_weight = (
                    eval_sample_weight[i]
                    if eval_sample_weight is not None and i < len(eval_sample_weight)
                    else None
                )
                valid_init = (
                    eval_init_score[i]
                    if eval_init_score is not None and i < len(eval_init_score)
                    else None
                )
                valid_set = Dataset(
                    data=valid_data[0],
                    label=self._le.transform(valid_data[1]),
                    weight=valid_weight,
                    init_score=valid_init,
                    params=combined_params,
                )
                valid_sets.append(valid_set)

        # Callbacks
        if callbacks is None:
            callbacks = []
        else:
            callbacks = copy.copy(callbacks)

        evals_result = {}
        callbacks.append(record_evaluation(evals_result))

        # Train using fairgbm.training.train()
        self._Booster = fairgbm_training.train(
            params=combined_params,
            train_set=train_set,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets if valid_sets else None,
            valid_names=eval_names,
            feval=eval_metric if callable(eval_metric) else None,
            init_model=(
                init_model.booster_ if hasattr(init_model, "booster_") else init_model
            ),
            callbacks=callbacks,
            constraint_group=self._constraint_group,
        )

        self._n_features = self._Booster.num_feature()
        self.n_features_in_ = X.shape[1] if hasattr(X, "shape") else len(X[0])
        self._evals_result = evals_result
        self._best_iteration = self._Booster.best_iteration
        self._best_score = self._Booster.best_score
        self.fitted_ = True

        self._Booster.free_dataset()
        return self

    def predict_proba(
        self,
        X,
        raw_score=False,
        start_iteration=0,
        num_iteration=None,
        pred_leaf=False,
        pred_contrib=False,
        validate_features=False,
        **kwargs,
    ):
        """Predict class probabilities for X.

        For constrained objectives (custom fobj), applies sigmoid to raw
        scores to produce valid probabilities. For standard objectives,
        delegates to LGBMClassifier.predict_proba.
        """
        from fairgbm.params import _is_constrained

        fairgbm_params = {
            "constraint_type": self.constraint_type,
            "global_constraint_type": self.global_constraint_type,
        }

        if _is_constrained(fairgbm_params) and not (
            raw_score or pred_leaf or pred_contrib
        ):
            # Custom objective was used — raw scores are logits, apply sigmoid
            raw = self._Booster.predict(
                X,
                raw_score=True,
                start_iteration=start_iteration,
                num_iteration=num_iteration,
                pred_leaf=False,
                pred_contrib=False,
            )
            proba_pos = 1.0 / (1.0 + np.exp(-raw))
            return np.vstack((1.0 - proba_pos, proba_pos)).transpose()
        else:
            return super().predict_proba(
                X,
                raw_score=raw_score,
                start_iteration=start_iteration,
                num_iteration=num_iteration,
                pred_leaf=pred_leaf,
                pred_contrib=pred_contrib,
                validate_features=validate_features,
                **kwargs,
            )

    def predict(
        self,
        X,
        raw_score=False,
        start_iteration=0,
        num_iteration=None,
        pred_leaf=False,
        pred_contrib=False,
        validate_features=False,
        **kwargs,
    ):
        """Predict class labels for X.

        For constrained objectives, uses sigmoid on raw scores and applies
        threshold at 0.5.
        """
        from fairgbm.params import _is_constrained

        fairgbm_params = {
            "constraint_type": self.constraint_type,
            "global_constraint_type": self.global_constraint_type,
        }

        if _is_constrained(fairgbm_params) and not (
            raw_score or pred_leaf or pred_contrib
        ):
            proba = self.predict_proba(
                X,
                start_iteration=start_iteration,
                num_iteration=num_iteration,
            )
            return self._le.inverse_transform(np.argmax(proba, axis=1))
        else:
            return super().predict(
                X,
                raw_score=raw_score,
                start_iteration=start_iteration,
                num_iteration=num_iteration,
                pred_leaf=pred_leaf,
                pred_contrib=pred_contrib,
                validate_features=validate_features,
                **kwargs,
            )

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Includes both FairGBM-specific and LGBMClassifier parameters.
        """
        params = super().get_params(deep=deep)
        # Ensure FairGBM params are included
        params["constraint_type"] = self.constraint_type
        params["multiplier_learning_rate"] = self.multiplier_learning_rate
        params["constraint_stepwise_proxy"] = self.constraint_stepwise_proxy
        params["objective_stepwise_proxy"] = self.objective_stepwise_proxy
        params["stepwise_proxy_margin"] = self.stepwise_proxy_margin
        params["constraint_fpr_tolerance"] = self.constraint_fpr_tolerance
        params["constraint_fnr_tolerance"] = self.constraint_fnr_tolerance
        params["score_threshold"] = self.score_threshold
        params["global_constraint_type"] = self.global_constraint_type
        params["global_target_fpr"] = self.global_target_fpr
        params["global_target_fnr"] = self.global_target_fnr
        params["global_score_threshold"] = self.global_score_threshold
        params["init_lagrangian_multipliers"] = self.init_lagrangian_multipliers
        return params
