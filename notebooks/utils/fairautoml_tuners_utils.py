""" ** A COPY OF THE FILE AT fairautoml.tuners.utils **
"""
"""Utils specific to the tuners module.
"""
from inspect import signature
from numbers import Number
from typing import Union
from pathlib import Path

import yaml
from schema import And
from schema import Optional as Optional_
from schema import Or, Schema

from .fairautoml_utils_classpath import import_object
from .fairautoml_utils_types import require_all_of_type


class YamlValidator:
    def __init__(self):
        self.obs_class = None

    def assert_class_exists(self, path: str) -> bool:
        """Checks if a given module and Class exists in the current python
        environment. Saves class in global object to assert arguments.

        Parameters
        ----------
        path : str
            Classpath to the Class to be checked.

        Returns    AttributeError
            If Class does not exist within module.
        ValueError
            If classpath is malformed.
        -------
        bool
            True if check passes.

        Raises
        ------
        ModuleNotFoundError
            If module does not exist.
        AttributeError
            If Class does not exist within module.
        ValueError
            If classpath is malformed.
        """
        try:
            self.obs_class = import_object(path)

        except ModuleNotFoundError as e:
            module_name = path[: path.rindex(".")]
            raise ModuleNotFoundError(
                f"Provided module for model in YAML file does not exist. Check "
                f"for errors in '{module_name}'."
            ) from e

        except AttributeError as e:
            class_name = path[path.rindex(".") + 1:]
            raise AttributeError(
                f"Provided class for model in YAML file is not contained in "
                f"module. Check for errors in '{class_name}'."
            ) from e

        except ValueError as e:
            raise ValueError(
                f"Provided classpath is malformed. You should specify the "
                f"module and class name. Check for errors in '{path}'."
            ) from e

        return True

    def assert_argument_exists(self, argument: str) -> bool:
        """Checks if a given argument for the Model Class to be isntantiated is
        expected by the Class signature. Uses global object of model to check
        signature.

        Parameters
        ----------
        argument : str
            Argument to be checked.

        Returns
        -------
        bool
            True if check passes.

        Raises
        ------
        TypeError
            If argument is not expected in Class.
        """
        signature_params = list(signature(self.obs_class).parameters)
        if "kwargs" in signature_params or argument in signature_params:
            return True
        else:
            raise TypeError(
                f"Unexpected argument for model '{self.obs_class}':" f" '{argument}'"
            )


validator = YamlValidator()

# Schema for YAML file.
HYPERPARAMETER_SPACE_SCHEMA = Schema(
    {
        str: {  # Model name given by the user
            "classpath": And(str, validator.assert_class_exists),
            "kwargs": {
                Optional_(And(str, validator.assert_argument_exists)): Or(
                    Or(str, Number),
                    And(list, lambda x: len(x) >= 1),
                    And(
                        {
                            "type": Or(
                                "int",
                                "float",
                            ),
                            "range": And(
                                list,
                                lambda values: len(values) == 2,
                                lambda values: require_all_of_type(values, Number),
                            ),
                            Optional_("log", default=False): Or(True, False),
                        }
                    ),
                )
            },
        },
    }
)


def load_hyperparameter_space(path_or_dict: Union[str, dict]) -> dict:
    """Loads the hyperparameter space encoded as a YAML in the given path.
    If given a dict, space is already loaded and this function will return the
    same object.

    Parameters
    ----------
    path_or_dict : Union[str, dict]
        Either the path to the YAML file, or a dictionary containing a
        hyperparameter space following the expected structure.

    Returns
    -------
    The loaded hyperparameter space.
    """
    # Read hyperparameter space from the YAML file (if given)
    if isinstance(path_or_dict, (str, Path)):
        try:
            with open(path_or_dict, "r") as f_in:
                hyperparameter_space = yaml.safe_load(f_in)
        except yaml.YAMLError as err:
            raise ValueError(f"{err}. Did you pass a valid YAML file ?") from err
    # Else, assume the given dictionary describes hyperparameter space
    elif isinstance(path_or_dict, dict):
        hyperparameter_space = path_or_dict
    else:
        raise ValueError(
            "Invalid value for `learner_hyperparams`. "
            "Must be either a path to a YAML file or a dict following the same structure."
        )
    # validate the configuration file
    HYPERPARAMETER_SPACE_SCHEMA.validate(hyperparameter_space)

    return hyperparameter_space
