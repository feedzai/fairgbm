""" ** A COPY OF THE FILE AT fairautoml.utils.types **
"""
#  The copyright of this file belongs to anonymized. The file cannot be
#  reproduced in whole or in part, stored in a retrieval system,
#  transmitted in any form, or by any means electronic, mechanical,
#  photocopying, or otherwise, without the prior permission of the owner.
#
#  © 2019 anonymized, Strictly Confidential

"""
Precondition functions to help and assert inputs and outputs condition and type

Notes
-----
From: https://gitlab.anonymized.com/commons/pulse-data-science-python/-/raw/master/ds_api/preconditions.py
"""

from typing import Iterable, Optional, Type, TypeVar, Union, no_type_check

X = TypeVar("X")
XY = TypeVar("XY", bound=Iterable)
TS = TypeVar("TS", bound=Iterable)


@no_type_check
def require_type(
    variable: X, expected_type: Type, variable_name: Optional[str] = None
) -> X:
    """Requite type of variable.

    Parameters
    ----------
    variable : X
        Variable to be type tested.
    expected_type : Type
        Type allowed apart from None.
    variable_name : str
        Name of the variable

    Returns
    -------
    variable : X
        If assert is True, it returns the variable.
    """
    message = message_with_name(
        f"Expected type {expected_type}, got {type(variable)}!", variable_name
    )
    if not isinstance(variable, expected_type):
        raise TypeError(message)

    return variable


@no_type_check
def require_one_of_types(
    variable: X, allowed_types: TS, variable_name: Optional[str] = None
) -> X:
    """Require one of the types specified.
    """
    message = message_with_name(
        f"Expected one of {allowed_types} types in variable, got {type(variable)}!",
        variable_name,
    )
    if not any(isinstance(variable, allowed_type) for allowed_type in allowed_types):
        raise TypeError(message)

    return variable


@no_type_check
def require_all_of_type(
        iterable: XY, expected_type: Type, iterable_name: Optional[str] = None
    ) -> XY:
    """
    Require all objects from an iterable to be of the type specified
    """
    message = message_with_name(
        f"Expected all values in variable to be of type {expected_type}!", iterable_name
    )
    if not all(isinstance(variable, expected_type) for variable in iterable):
        raise TypeError(message)

    return iterable


@no_type_check
def require_type_or_none(
        variable: X, expected_type: Type, variable_name: Optional[str] = None
    ) -> Union[X, None]:
    """Require a type or None.

    Parameters
    ----------
    variable : X
        Variable to be type tested.
    expected_type : Type
        Type allowed apart from None.
    variable_name : str
        Name of the variable

    Returns
    -------
    Union[X, None]
        If assert is True, it returns the iterable passed
    """
    if variable is None:
        return None
    return require_type(variable, expected_type, variable_name=variable_name)

@no_type_check
def require_not_none(variable: X, variable_name: Optional[str] = None) -> X:
    """Require a variable to be not None.

    Parameters
    ----------
    variable : X
        Variable to be type tested.
    variable_name : str
        Name of the variable

    Returns
    -------
    X
        If assert is True, it returns the variable passed.
    """
    message = message_with_name(
        "Expected variable to be different than None!", variable_name,
    )
    if variable is None:
        raise TypeError(message)
    return variable


# helper function
def message_with_name(message: str, variable_name: Optional[str]) -> str:
    """Appends the name of the variable in the message if name given."""
    if variable_name is None:
        return message
    else:
        return f"Variable {variable_name}: {message}"
