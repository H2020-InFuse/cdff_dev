import os
import math
import msgpack


def load_log(filename):
    """Load logfile.

    Parameters
    ----------
    filename : str
        Name of the logfile

    Returns
    -------
    log : dict
        Log data
    """
    with open(filename, "rb") as f:
        streams = msgpack.unpack(f, encoding="utf8")
    return streams


def print_stream_info(log):
    """Print meta information about streams.

    Parameters
    ----------
    log : dict
        Log data
    """
    stream_meta_data = []
    for stream_name in log.keys():
        if stream_name.endswith(".meta"):
            continue
        typename = log[stream_name + ".meta"]["type"]
        n_samples = len(log[stream_name])
        stream_meta_data.append((stream_name, typename, str(n_samples).rjust(14)))

    print("=" * 80)
    MAXLENGTHS = (35, 30, 15)
    print("".join(map(lambda t: t[0].ljust(t[1]), zip(["stream name", "type", "# samples"], MAXLENGTHS))))
    print("-" * 80)
    for stream_meta_data in stream_meta_data:
        line = ""
        for i in range(len(MAXLENGTHS)):
            if len(stream_meta_data[i]) >= MAXLENGTHS[i]:
                line += stream_meta_data[i][:MAXLENGTHS[i] - 4] + "... "
            else:
                line += stream_meta_data[i].ljust(MAXLENGTHS[i])
        print(line)
    print("=" * 80)


def replay(stream_names, log, verbose=0):
    """Generator that will output samples in the correct temporal order.

    Parameters
    ----------
    stream_names : list
        Names of the streams

    log : dict
        Log data

    verbose : int, optional (default: 0)
        Verbosity level

    Returns
    -------
    current_timestamp : int
        Current time in microseconds

    stream_name : str
        Name of the currently active stream

    typename : str
        Name of the data type of the stream

    sample : dict
        Current sample
    """
    streams = [log[sn] for sn in stream_names]
    meta_streams = [log[sn + ".meta"] for sn in stream_names]

    n_streams = len(streams)
    current_stream_indices = [-1] * n_streams
    timestamps_per_stream = [meta_stream["timestamps"]
                             for meta_stream in meta_streams]
    stream_lens = [len(t) for t in timestamps_per_stream]

    if verbose >= 1:
        print("[replay] Replaying %d streams:" % n_streams)
        stream_statistics = ["%s - %d samples" % (sn, len(log[sn]))
                             for sn in stream_names]
        print("    " + (os.linesep + "    ").join(stream_statistics))

    while True:
        next_timestamps = [float("inf")] * n_streams
        for i in range(n_streams):
            stream_idx = current_stream_indices[i]
            if stream_idx + 1 < stream_lens[i]:
                next_timestamps[i] = timestamps_per_stream[i][stream_idx + 1]
        if verbose >= 2:
            print("[replay] Next timestamps: %s" % (next_timestamps,))
        if all(map(math.isinf, next_timestamps)):
            return

        current_stream = next_timestamps.index(min(next_timestamps))
        if verbose >= 2:
            print("[replay] Selected stream #%d '%s'"
                  % (current_stream, stream_names[current_stream]))

        current_stream_indices[current_stream] += 1
        if verbose >= 2:
            print("[replay] Stream indices: %s" % (current_stream_indices,))

        current_stream_idx = current_stream_indices[current_stream]
        current_timestamp = timestamps_per_stream[current_stream][current_stream_idx]
        typename = meta_streams[current_stream]["type"]
        sample = streams[current_stream][current_stream_idx]
        yield current_timestamp, stream_names[current_stream], typename, sample
