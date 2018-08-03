import cdff_types
import warnings


def create_cpp(typename):
    """Create C++ object from typename.

    Parameters
    ----------
    typename : str
        Name of the type, there must be a corresponding type in cdff_types.
        Some conversion steps will be done automatically, for example,
        * /bla/blub -> BlaBlub
        * /gps/Solution -> GpsSolution

    Returns
    -------
    obj : object
        The C++ class wrapped in Python. It can be passed to wrapped C++
        extensions in Python.
    """
    typename = _translate_typename(typename)
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
        Name of the type, there must be a corresponding type in cdff_types.
        Some conversion steps will be done automatically, for example,
        * /bla/blub -> BlaBlub
        * /gps/Solution -> GpsSolution

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


def _translate_typename(typename):
    """Translate typename from string to a valid class name.

    Parameters
    ----------
    typename : str
        Name of the type

    Returns
    -------
    typename : str
        Some conversion steps will be done, for example,
        * /bla/blub -> BlaBlub
        * /gps/Solution -> GpsSolution
    """
    while "/" in typename:
        i = typename.find("/")
        if i + 1 < len(typename):
            typename = typename[:i] + typename[i + 1].upper() + typename[i + 2:]
    return typename


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
            try:
                _assign_dict(obj[i], data[i])
            except KeyError as e:
                warnings.warn(
                    "Failed to assign data[%s] to obj[%s], reason:\n%s"
                    % (i, i, str(e)))
        else:
            obj[i] = data[i]


def _assign_dict(obj, data):
    for key, value in data.items():
        _assign_element(obj, key, value)


def _assign_element(obj, fieldname, data):
    fieldname = _camel_case_to_snake_case(fieldname)

    if not hasattr(obj, fieldname):
        # HACK for current version of ASN.1 types
        if fieldname == "ref_time" and hasattr(obj, "timestamp"):
            fieldname = "timestamp"
        elif fieldname == "timestamp" and hasattr(obj, "ref_time"):
            fieldname = "ref_time"
        else:
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

