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
    """Generator that yields samples of one log in correct temporal order.

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
    current_sample_indices = [-1] * n_streams

    if verbose >= 1:
        print("[replay] Replaying %d streams:" % n_streams)
        stream_statistics = ["%s - %d samples" % (sn, len(log[sn]))
                             for sn in stream_names]
        print("    " + (os.linesep + "    ").join(stream_statistics))

    while True:
        try:
            current_stream, _ = next_timestamp(
                meta_streams, current_sample_indices)
        except StopIteration:
            return

        if verbose >= 2:
            print("[replay] Selected stream #%d '%s'"
                  % (current_stream, stream_names[current_stream]))

        current_sample_indices[current_stream] += 1
        if verbose >= 2:
            print("[replay] Stream indices: %s" % (current_sample_indices,))

        timestamp, stream_name, typename, sample = extract_sample(
            streams, meta_streams, stream_names, current_stream,
            current_sample_indices[current_stream])
        yield timestamp, stream_name, typename, sample


def replay_files(filename_groups, stream_names):
    groups = [LogfileGroup(group) for group in filename_groups]

    raise NotImplementedError()


class LogfileGroup:
    def __init__(self, filenames, stream_names):
        self.filenames = filenames
        self.stream_names = stream_names

        self.file_index = -1
        self.current_sample_indices = None
        self.streams = None
        self.group_stream_names = None
        self.next_stream = -1

        self._load_streams()

    def next_timestamp(self):
        for i in range(self.group_stream_names):
            stream_idx = self.current_sample_indices[i]
            if stream_idx + 1 >= len(self.streams[i]):
                self._load_streams()

        self.next_stream, timestamp = next_timestamp()
        return timestamp

    def _load_streams(self):
        # TODO: extend existing logs...
        if self.file_index + 1 >= len(self.filenames):
            return

        self.file_index += 1

        streams = load_log(self.filenames[self.file_index])
        self.group_stream_names = sorted(
            k for k in streams.keys() if k in self.stream_names)

        self.streams = [streams[k] for k in self.group_stream_names]
        self.meta_streams = [
            streams[k + ".meta"] for k in self.group_stream_names]
        self.current_sample_indices = [-1] * len(self.streams)

    def next_sample(self):
        sample_idx = self.current_sample_indices[self.next_stream]
        return extract_sample(
            self.streams, self.meta_streams, self.group_stream_names,
            self.next_stream, sample_idx)


def next_timestamp(meta_streams, current_stream_indices):
    n_streams = len(current_stream_indices)

    next_timestamps = [float("inf")] * n_streams
    for i in range(n_streams):
        stream_idx = current_stream_indices[i]
        if stream_idx + 1 < len(meta_streams[i]["timestamps"]):
            next_timestamps[i] = meta_streams[i]["timestamps"][stream_idx + 1]

    if all(map(math.isinf, next_timestamps)):
        raise StopIteration("reached end of log")

    next_stream, next_timestamp = min(
        enumerate(next_timestamps), key=lambda p: p[1])
    return next_stream, next_timestamp


def extract_sample(streams, meta_streams, stream_names, stream_idx,
                   sample_idx):
    timestamp = meta_streams[stream_idx]["timestamps"][sample_idx]
    typename = meta_streams[stream_idx]["type"]
    sample = streams[stream_idx][sample_idx]
    stream_name = stream_names[stream_idx]
    return timestamp, stream_name, typename, sample