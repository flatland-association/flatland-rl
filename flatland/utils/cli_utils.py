import importlib
from typing import Any


def resolve_type(fully_qualified: Any = None, pkg: str = None, cls: str = None):
    """
    Returns fully_qualified if it's not a string. Otherwise, resolves from fully_qualified first, else from pkg and cls.
    Parameters
    ----------
    fully_qualified : str or type
    pkg : Optional[str]
    cls : Optional[str]

    Returns
    -------
    The resolved type.

    """
    if fully_qualified is not None:
        if not isinstance(fully_qualified, str):
            return fully_qualified
        parts = fully_qualified.split(".")
        pkg = ".".join(parts[:-1])
        cls = parts[-1]
    if pkg is None or cls is None:
        return None
    module = importlib.import_module(pkg)
    return getattr(module, cls)
