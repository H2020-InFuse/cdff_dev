def convert_to_dict(obj):
    """Convert object to dictionary.

    Parameters
    ----------
    obj : class defined in cdff_types
        Object

    Returns
    -------
    data : dict or list
        Contains the actual data. Only basic types like list, dict, float,
        str and int are allowed here.
    """
    if type(obj) in [float, int, str, bool]:
        return obj
    if hasattr(obj, "shape"):
        return _convert_ndarray(obj)
    elif hasattr(obj, "__len__"):
        return _convert_list(obj)
    else:
        return _convert_dict(obj)


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
        value = getattr(obj, f)
        data[f] = convert_to_dict(value)
    return data


def _get_fieldnames(obj):
    return [f for f in dir(obj)
            if not callable(getattr(obj, f)) and not f.startswith('__')]
