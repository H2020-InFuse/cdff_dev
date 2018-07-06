import cdff_types


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


def create_from_dict(typename, data):
    """Convert intermediate logfile format to InFuse C++ type.

    Parameters
    ----------
    typename : str
        Name of the type

    data : object
        Contains the actual data. Only basic types like list, dict, float,
        str and int are allowed here.

    Returns
    -------
    obj : object
        The corresponding C++ class wrapped in Python. It can be passed to
        wrapped C++ extensions in Python.
    """
    obj = create_cpp(typename)
    return _convert(obj, data)


def _convert(obj, data):
    if isinstance(data, list):
        _assign_list(obj, data)
    elif isinstance(data, dict):
        _assign_dict(obj, data)
    else:
        raise ValueError("Cannot handle data of type '%s'" % type(data))
    return obj


def _assign_list(obj, data):
    """List corresponds to a sequence in ASN.1."""
    for i in range(len(data)):
        if type(data[i]) == list:  # special case: matrices
            for j in range(len(data[i])):
                obj[i, j] = data[i][j]
        elif type(data[i]) == dict:  # nested types
            _assign_dict(obj[i], data[i])
        else:
            obj[i] = data[i]


def _assign_dict(obj, data):
    for key, value in data.items():
        _assign_element(obj, key, value)


def _assign_element(obj, fieldname, data):
    fieldname = _camel_case_to_snake_case(fieldname)

    if not hasattr(obj, fieldname):
        raise ValueError("Type '%s' has no field with name '%s'"
                         % (type(obj), fieldname))

    field = getattr(obj, fieldname)
    if type(field).__module__ == "cdff_types":
        _convert(field, data)
    else:
        try:
            setattr(obj, fieldname, data)
        except TypeError as e:
            raise TypeError("Failed to set %s.%s = %s, error message: %s"
                            % (obj.__class__.__name__, fieldname, data, e))


def _camel_case_to_snake_case(name):
    """Convert camel case to snake case and '-' to '_'."""
    new_name = str(name)
    i = 0
    while i < len(new_name):
        if new_name[i].isupper() and i > 0:
            new_name = new_name[:i] + "_" + new_name[i:]
            i += 1
        i += 1
    new_name = new_name.replace("-", "_")
    return new_name.lower()

