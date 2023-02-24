import random

from .fairautoml_tuners_suggest import suggest_hyperparams, suggest_callable_hyperparams
from .fairautoml_utils_trial import RandomValueTrial


def suggest_random_hyperparams(hyperparam_space: dict, seed: int = None) -> dict:
    return suggest_hyperparams(
        trial=RandomValueTrial(seed=seed if seed else random.randrange(2**32 - 1)),
        hyperparameter_space=hyperparam_space,
    )


def suggest_random_hyperparams_with_classpath(hyperparam_space: dict, seed: int = None) -> dict:
    return suggest_callable_hyperparams(
        trial=RandomValueTrial(seed=seed if seed else random.randrange(2**32 - 1)),
        hyperparameter_space=hyperparam_space,
    )
