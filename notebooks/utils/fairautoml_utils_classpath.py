""" ** A COPY OF THE FILE AT fairautoml.utils.classpath **
"""
"""Utils pertaining to importing and loading objects.
"""
import importlib
from typing import Union, Callable, Any


def import_object(import_path: str) -> Union[object, Callable]:
    """Imports the object at the given module/class path.

    Parameters
    ----------
    import_path : str
        The import path for the object to import.

    Returns
    -------
    The imported object (this can be a class, a callable, a variable).
    """
    separator_idx = import_path.rindex('.')
    module_path = import_path[: separator_idx]
    obj_name = import_path[separator_idx + 1:]

    module = importlib.import_module(module_path)
    return getattr(module, obj_name)


def get_full_name(obj: Any) -> str:
    """Returns identifier name for the given callable.

    Should be equal to the import path:
        obj == import_object(get_full_name(obj))

    Parameters
    ----------
    obj : object
        The object to find the classpath for.

    Returns
    -------
    The object's classpath.
    """
    if callable(obj):
        return obj.__module__ + '.' + obj.__qualname__
    else:
        return obj.__class__.__module__ + '.' + obj.__class__.__qualname__
