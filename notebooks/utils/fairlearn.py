from typing import Union
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import pandas as pd
from fairlearn.reductions import Moment, GridSearch
from fairlearn.reductions._grid_search._grid_generator import _GridGenerator

from .fairautoml_utils_classpath import import_object


def parse_kwargs(kwargs: dict) -> dict:
    MODEL_ARGS_KEY = 'model__'
    CONSTRAINT_ARGS_KEY = 'constraint__'

    # Parse key-word arguments
    model_kwargs = {
        k[len(MODEL_ARGS_KEY):]: v for k, v in kwargs.items()
        if k.startswith(MODEL_ARGS_KEY)
    }

    # -> for the constraint
    constraint_kwargs = {
        k[len(CONSTRAINT_ARGS_KEY):]: v for k, v in kwargs.items()
        if k.startswith(CONSTRAINT_ARGS_KEY)
    }

    # -> finally, everything left is a kwarg to the meta_estimator
    kwargs = {
        k: v for k, v in kwargs.items() if not any([
            k.startswith(MODEL_ARGS_KEY),
            k.startswith(CONSTRAINT_ARGS_KEY),
        ])
    }

    # Build results dict
    results = OrderedDict()
    results['model_kwargs'] = model_kwargs
    results['constraint_kwargs'] = constraint_kwargs
    results['kwargs'] = kwargs
    return results


class FairlearnClassifier:
    """Module for wrapping a classifier under restrictions from the
    fairlearn package.
    """

    def __init__(
            self,
            fairlearn_reduction: Union[str, callable],
            estimator: Union[str, callable],
            constraint: Union[str, Moment],
            protected_column: str,
            random_state: int = 42,
            n_jobs: int = -1,
            unawareness: bool = False,
            **kwargs,
        ):
        if isinstance(fairlearn_reduction, str):
            fairlearn_reduction = import_object(fairlearn_reduction)
        if isinstance(estimator, str):
            estimator = import_object(estimator)
        if isinstance(constraint, str):
            constraint = import_object(constraint)

        self.protected_column = protected_column
        self.random_state = random_state
        self.n_jobs = n_jobs    # NOTE: currently being ignored
        self.unawareness = unawareness

        # Parse key-word arguments
        self.model_kwargs, self.constraint_kwargs, self.kwargs = parse_kwargs(kwargs).values()

        # Build fairlearn object
        self.base_estimator = estimator(random_state=self.random_state, **self.model_kwargs)
        self.constraint = constraint(**self.constraint_kwargs)

        self.fairlearn_reduction = fairlearn_reduction(
            estimator=self.base_estimator,
            constraints=self.constraint,
            **self.kwargs,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        A = X[self.protected_column]
        if self.unawareness and self.protected_column in X.columns:
            X = X[set(X.columns) - {self.protected_column}]
        return self.fairlearn_reduction.fit(X, y, sensitive_features=A, **kwargs)

    def predict(self, X: pd.DataFrame):
        if self.unawareness and self.protected_column in X.columns:
            X = X[set(X.columns) - {self.protected_column}]
        return self.fairlearn_reduction.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        if self.unawareness and self.protected_column in X.columns:
            X = X[set(X.columns) - {self.protected_column}]
        return self.fairlearn_reduction.predict_proba(X)


class GridSearchReductionClassifier:

    def __init__(
            self,
            estimator: Union[str, callable],
            constraint: Union[str, Moment],
            protected_column: str,
            sample_from_grid: int = 10,
            random_state: int = 42,
            n_jobs: int = -1,
            unawareness: bool = False,
            **kwargs,
        ):
        if isinstance(estimator, str):
            estimator = import_object(estimator)
        if isinstance(constraint, str):
            constraint = import_object(constraint)

        self.protected_column = protected_column
        self.sample_from_grid = sample_from_grid
        self.random_state = random_state
        self.n_jobs = n_jobs    # NOTE: currently being ignored
        self.unawareness = unawareness
        self.rng = np.random.RandomState(self.random_state)

        # Parse key-word arguments
        self.model_kwargs, self.constraint_kwargs, self.kwargs = parse_kwargs(kwargs).values()

        # Build fairlearn object
        self.base_estimator = estimator(random_state=self.random_state, **self.model_kwargs)
        self.constraint = constraint(**self.constraint_kwargs)
        self.fairlearn_reduction = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):

        # Get protected attributes column
        A = X[self.protected_column]
        if self.unawareness and self.protected_column in X.columns:
            X = X[set(X.columns) - {self.protected_column}]

        # Load data to constraint (necessary to access pos_basis, neg_basis, etc.)
        dummy_constraint = deepcopy(self.constraint)
        dummy_constraint.load_data(X, y, sensitive_features=A, **kwargs)
            
        # Randomly select a set of Lagrangian multipliers from the generated grid
        grid = _GridGenerator(
            grid_size=kwargs.pop('grid_size', 50),
            grid_limit=kwargs.pop('grid_limit', 3.0),
            pos_basis=kwargs.pop('pos_basis', dummy_constraint.pos_basis),
            neg_basis=kwargs.pop('neg_basis', dummy_constraint.neg_basis),
            neg_allowed=kwargs.pop('neg_allowed', dummy_constraint.neg_basis_present),
            force_L1_norm=kwargs.pop('force_L1_norm', dummy_constraint.default_objective_lambda_vec is not None),
            grid_offset=None).grid

        rng_indices = self.rng.choice(grid.shape[1], self.sample_from_grid, replace=False)
        grid = grid.iloc[:, rng_indices]   # NOTE: fairlearn crashes if you train a single model with GS...

        self.fairlearn_reduction = GridSearch(
            estimator=self.base_estimator,
            constraints=self.constraint,
            grid=grid,
            **self.kwargs,
        )

        # Fit GridSearch classifier
        return self.fairlearn_reduction.fit(X, y, sensitive_features=A, **kwargs)

    def predict(self, X: pd.DataFrame):
        if self.unawareness and self.protected_column in X.columns:
            X = X[set(X.columns) - {self.protected_column}]
        return self.fairlearn_reduction.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        if self.unawareness and self.protected_column in X.columns:
            X = X[set(X.columns) - {self.protected_column}]
        return self.fairlearn_reduction.predict_proba(X)
