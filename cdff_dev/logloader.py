import os
import math
import warnings
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

    n_streams = len(stream_names)
    current_sample_indices = [-1] * n_streams

    if verbose >= 1:
        print("[replay] Replaying %d streams:" % n_streams)
        stream_statistics = ["%s - %d samples" % (sn, len(log[sn]))
                             for sn in stream_names]
        print("    " + (os.linesep + "    ").join(stream_statistics))

    while True:
        current_stream, _ = next_timestamp(
            meta_streams, current_sample_indices)
        if current_stream < 0:
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
    """Generator that yields samples of multiple logs in correct temporal order.

    Parameters
    ----------
    filename_groups : list of list of str
        Groups of filenames in chronological order

    stream_names : list
        Names of the streams that we want to load

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
    groups = [LogfileGroup(group, stream_names) for group in filename_groups]
    while True:
        next_timestamps = [g.next_timestamp() for g in groups]
        if all(map(math.isinf, next_timestamps)):
            return
        current_group, _ = _argmin(next_timestamps)
        timestamp, stream_name, typename, sample = groups[
            current_group].next_sample()
        yield timestamp, stream_name, typename, sample


class LogfileGroup:
    """A group of logfiles that are actually one log cut into smaller chunks.

    Parameters
    ----------
    filenames : list
        Names of the files in chronological order

    stream_names : list
        Names of the streams that we want to load

    verbose : int
        Verbosity level
    """
    def __init__(self, filenames, stream_names, verbose=0):
        self.filenames = filenames
        self.stream_names = stream_names
        self.verbose = verbose

        self.file_index = -1
        self.current_sample_indices = []
        self.group_stream_names = []
        self.streams = []
        self.meta_streams = []
        self.next_stream = -1

        self._load_streams()

    def next_timestamp(self):
        for i in range(len(self.group_stream_names)):
            sample_idx = self.current_sample_indices[i]
            if sample_idx + 1 >= len(self.streams[i]):
                self._load_streams()

        self.next_stream, timestamp = next_timestamp(
            self.meta_streams, self.current_sample_indices)

        if self.verbose >= 2:
            if self.next_stream < 0:
                print("[logloader] Reached the end of the logfile.")
            else:
                stream_name = self.group_stream_names[self.next_stream]
                print("[logloader] Next sample from stream #%d (%s), clock: %f"
                      % (self.next_stream, stream_name, timestamp))

        return timestamp

    def _load_streams(self):
        if self.file_index + 1 >= len(self.filenames):
            if self.verbose >= 1:
                print("[logloader] No more logfiles in this group.")
            return

        self.file_index += 1

        if self.verbose >= 1:
            print("[logloader] Loading logfile #%d: %s"
                  % (self.file_index + 1, self.filenames[self.file_index]))

        self._delete_processed_samples()

        new_streams = load_log(self.filenames[self.file_index])
        new_stream_names = [sn for sn in new_streams.keys()
                            if not sn.endswith(".meta")]

        if self.verbose >= 1:
            print_stream_info(new_streams)

        for new_stream_name in new_stream_names:
            if new_stream_name in self.stream_names:
                new_stream = new_streams[new_stream_name]
                new_meta_stream = new_streams[new_stream_name + ".meta"]
                assert len(new_stream) == len(new_meta_stream["timestamps"]), \
                    "got %d samples, but %d timestamps in stream '%s'" % (
                        len(new_stream), len(new_meta_stream["timestamps"]),
                        new_stream_name)
                if new_stream_name not in self.group_stream_names:
                    self.group_stream_names.append(new_stream_name)
                    self.streams.append(new_stream)
                    self.meta_streams.append(new_meta_stream)
                    self.current_sample_indices.append(-1)
                else:
                    stream_idx = self.group_stream_names.index(new_stream_name)

                    assert (
                        self.meta_streams[stream_idx]["type"] ==
                        new_meta_stream["type"]), (
                        "%s != %s" % (self.meta_streams[stream_idx]["type"],
                                      new_meta_stream["type"]))

                    self.streams[stream_idx].extend(new_stream)
                    self.meta_streams[stream_idx]["timestamps"].extend(
                        new_meta_stream["timestamps"])

        if len(self.group_stream_names) == 0:
            warnings.warn(
                "Could not load any streams in this logfile group. "
                "Allowed stream names: %s; Streams in this group: %s"
                % (self.stream_names, new_stream_names)
            )

    def _delete_processed_samples(self):
        for stream_idx in range(len(self.streams)):
            sample_idx = self.current_sample_indices[stream_idx]
            self.streams[stream_idx] = self.streams[stream_idx][sample_idx:]
            self.meta_streams[stream_idx]["timestamps"] = \
                self.meta_streams[stream_idx]["timestamps"][sample_idx:]
            self.current_sample_indices[stream_idx] = 0

    def next_sample(self):
        if self.verbose >= 2:
            print("[logloader] Processing sample from stream %d/%d"
                  % (self.next_stream + 1, len(self.group_stream_names)))
        self.current_sample_indices[self.next_stream] += 1
        return extract_sample(
            self.streams, self.meta_streams, self.group_stream_names,
            self.next_stream, self.current_sample_indices[self.next_stream])


def next_timestamp(meta_streams, current_sample_indices):
    n_streams = len(current_sample_indices)

    next_timestamps = [float("inf")] * n_streams
    for i in range(n_streams):
        sample_idx = current_sample_indices[i]
        if sample_idx + 1 < len(meta_streams[i]["timestamps"]):
            next_timestamps[i] = meta_streams[i]["timestamps"][sample_idx + 1]

    if all(map(math.isinf, next_timestamps)):
        return -1, float("inf")

    return _argmin(next_timestamps)


def _argmin(next_timestamps):
    return min(enumerate(next_timestamps), key=lambda p: p[1])


def extract_sample(streams, meta_streams, stream_names, stream_idx,
                   sample_idx):
    timestamp = meta_streams[stream_idx]["timestamps"][sample_idx]
    typename = meta_streams[stream_idx]["type"]
    sample = streams[stream_idx][sample_idx]
    stream_name = stream_names[stream_idx]
    return timestamp, stream_name, typename, sample