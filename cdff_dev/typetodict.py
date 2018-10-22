def convert_to_dict(obj):
    """Convert object to dictionary.

    Parameters
    ----------
    obj : class defined in cdff_types
        Object

    Returns
    -------
    data : object
        Contains the actual data. Only basic types like list, dict, float,
        str and int are allowed here.
    """
    if hasattr(obj, "shape"):
        data = _convert_ndarray(obj)
    elif hasattr(obj, "__len__"):
        data = _convert_list(obj)
    else:
        data = _convert_dict(obj)
    return data


def _convert_list(obj):
    data = []
    for i in range(len(obj)):
        data.append(obj[i])
    return data


def _convert_ndarray(obj):
    shape = obj.shape
    if len(shape) > 2:
        raise NotImplementedError("Cannot handle nd arrays with n > 2.")
    data = []
    for i in range(shape[0]):
        data.append([])
        for j in range(shape[1]):
            data[i].append(obj[i, j])
    return data


def _convert_dict(obj):
    data = {}
    fields = _get_fieldnames(obj)
    for f in fields:
        data[f] = getattr(obj, f)
    return data


def _get_fieldnames(obj):
    return [f for f in dir(obj)
            if not callable(getattr(obj, f)) and not f.startswith('__')]
