import pandas
from . import logloader


def get_port_names(logfile):
    """Get port names of a logfile.

    Parameters
    ----------
    logfile : str
        Name of the logfile

    Returns
    -------
    port_names : iterable
        Names of available ports
    """
    log = logloader.load_log(logfile)
    port_names = [k for k in log.keys() if not k.endswith(".meta")]
    return sorted(port_names)


def object2csv(input_filename, output_filename, port, fields=None,
               whitelist=None):
    """Convert a object-oriented MsgPack logfile to CSV.

    The dataframe will use the field 'timestamp.microseconds' as index.
    If there is no such field in the data type of the port, it will be
    created from the meta data.

    Parameters
    ----------
    input_filename : str
        Name of the original logfile

    output_filename : str
        Name of the converted logfile

    port : list or tuple
        Names of port that will be converted

    fields : list or tuple, optional (default: all)
        Field that will be exported

    whitelist : list or tuple, optional (default: [])
        Usually arrays and vectors (represented as lists in Python) are handled
        as basic types and are put in a single column because they can have
        dynamic sizes. This is a list of fields that will be scanned
        recursively and interpreted as arrays with a fixed length. Note that
        you only have to give the name of the field, not the port name.
        An example would be ["elements", "names"] if you want fully unravel
        a JointState object.
    """
    log = logloader.load_log(input_filename)
    df = object2dataframe(log, port, fields, whitelist)
    df.to_csv(output_filename, na_rep="NaN")


def join_labels(df, labels, column_name="label"):
    """Join features with labels column.

    Parameters
    ----------
    df : DataFrame
        Table containing the features

    labels : array-like
        Labels for classification or outlier detection

    column_name : str
        Name of the label

    Returns
    -------
    labeled_df : DataFrame
        A new dataframe that contains the label column
    """
    if df.shape[0] != len(labels):
        raise ValueError("Number of samples (%d) and labels (%d) do not match."
                         % (df.shape[0], len(labels)))
    return df.assign(**{column_name: labels})


def object2dataframe(log, port, fields=None, whitelist=None):
    """Convert from object-oriented storage to pandas dataframe.

    The dataframe will use the field 'timestamp.microseconds' as index.
    If there is no such field in the data type of the port, it will be
    created from the meta data.

    Parameters
    ----------
    log : dict
        Log data

    port : list or tuple
        Names of port that will be converted

    fields : list or tuple, optional (default: all)
        Field that will be exported

    whitelist : list or tuple, optional (default: [])
        Usually arrays and vectors (represented as lists in Python) are handled
        as basic types and are put in a single column because they can have
        dynamic sizes. This is a list of fields that will be scanned
        recursively and interpreted as arrays with a fixed length. Note that
        you only have to give the name of the field, not the port name.
        An example would be ["elements", "names"] if you want fully unravel
        a JointState object.
    """
    if whitelist is None:
        whitelist = ()

    converted_log = object2relational(log, whitelist)

    if fields is None:
        fields = list(converted_log[port].keys())

    port_log = dict()
    for field in converted_log[port].keys():
        if field in fields:
            port_log[field] = converted_log[port][field]
        else:
            for whitelisted_field in whitelist:
                if field.startswith(whitelisted_field + "."):
                    port_log[field] = converted_log[port][field]

    if "timestamp.microseconds" not in port_log:
        port_log["timestamp.microseconds"] = log[port + ".meta"]["timestamps"]

    df = pandas.DataFrame(port_log)
    df.reindex(sorted(df.columns), axis=1, copy=False)
    df.set_index("timestamp.microseconds", inplace=True)
    return df


def object2relational(log, whitelist=()):
    """Convert from object-oriented to relational storage.

    The log data can usually be accessed with
    log[port_name][sample_idx][field_a]...[field_b]. This is an
    object-oriented view of the data because you can easily access a whole
    object from the log. This is not convenient if you want to use the data
    for, e.g., machine learning, where you typically need the whole dataset
    in a 2D array, i.e. a relational view on the data, in which you can
    access data in the form log[port_name][feature][sample_idx].

    Parameters
    ----------
    log : dict
        Log data

    whitelist : list or tuple, optional (default: [])
        Usually arrays and vectors (represented as lists in Python) are handled
        as basic types and are put in a single column because they can have
        dynamic sizes. This is a list of fields that will be scanned
        recursively and interpreted as arrays with a fixed length. Note that
        you only have to give the name of the field, not the port name.
        An example would be ["elements", "names"] if you want fully unravel
        a JointState object.

    Returns
    -------
    converted_log : dict
        Converted log data
    """
    port_names = [k for k in log.keys() if not k.endswith(".meta")]

    converted_log = dict()
    for port_name in port_names:
        if len(log[port_name]) == 0:
            continue
        all_keys = _extract_keys(log[port_name][0], whitelist)
        _convert_data(converted_log, log, port_name, all_keys)
        _convert_metadata(converted_log, log, port_name)
    return converted_log


def _extract_keys(sample, whitelist=(), keys=()):
    if isinstance(sample, dict):
        result = []
        for k in sample.keys():
            result.extend(_extract_keys(sample[k], whitelist, keys + (k,)))
        return result
    elif isinstance(sample, list) and (".".join(keys) in whitelist):
        result = []
        for i in range(len(sample)):
            result.extend(_extract_keys(sample[i], whitelist, keys + (i,)))
        return result
    else:
        return [keys]


def _convert_data(converted_log, log, port_name, all_keys):
    converted_log[port_name] = dict()
    for keys in all_keys:
        new_key = ".".join(map(str, keys))
        if new_key == "":
            new_key = "data"
        converted_log[port_name][new_key] = []
        for t in range(len(log[port_name])):
            value = log[port_name][t]
            for k in keys:
                value = value[k]
            converted_log[port_name][new_key].append(value)


def _convert_metadata(converted_log, log, port_name):
    metadata = log[port_name + ".meta"]
    converted_log[port_name]["timestamp"] = metadata["timestamps"]
    n_rows = len(metadata["timestamps"])
    converted_log[port_name]["type"] = [metadata["type"]] * n_rows
