import sys
import logging

import numpy as np
import pandas as pd

from numpy.random import RandomState
from lightgbm import Booster


def randomized_classifier_predictions(
        model: Booster,
        df: pd.DataFrame,
        min_iter: int = 1,
        n_threads: int = 5,
        seed: int = 42,
    ) -> np.ndarray:
    """
    Produce predictions from the given GBM model, using its iterates
    as possible models to draw predictions from.
    
    i.e., a GBM randomized classifier.
    """

    # Object to generate random values
    rng = RandomState(seed)

    # Maximum number of iterations in GBM
    max_iter = model.num_trees()
    if min_iter >= max_iter:
        logging.error(f"Got min_iter={min_iter}, max_iter={max_iter};")
        min_iter = max(max_iter - 1, 0)
        logging.error(f"> Trying to use min_iter={min_iter}, max_iter={max_iter};")

    num_rows, _num_feats = df.shape
    y_scores = np.zeros(num_rows)
    
    # Assign a random model iteration to each row
    y_iter = rng.randint(low=min_iter, high=max_iter, size=(num_rows,))
    
    # For each model iteration, run predict for all corresponding rows
    for curr_iter in range(min_iter, max_iter):
        
        # Rows assigned to the current model iteration
        rows_for_curr_iter = np.argwhere(y_iter == curr_iter).flatten()

        curr_iter_preds = model.predict(
            df.iloc[rows_for_curr_iter],
            num_iteration=curr_iter,
            num_threads=n_threads)

        y_scores[rows_for_curr_iter] = curr_iter_preds

    return y_scores
