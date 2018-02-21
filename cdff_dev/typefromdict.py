from . import cdff_types


def create_cpp(typename):
    """Create C++ object from typename.

    Parameters
    ----------
    typename : str
        Name of the type

    Returns
    -------
    obj : object
        The C++ class wrapped in Python. It can be passed to wrapped C++
        extensions in Python.
    """
    if not hasattr(cdff_types, typename):
        raise ValueError("Type '%s' has no Python bindings." % typename)
    Type = getattr(cdff_types, typename)
    obj = Type()
    return obj

