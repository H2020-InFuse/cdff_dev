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


def summarize_logfile(filename):
    """Extract summary of log file.

    Parameters
    ----------
    filename : str
        Name of the logfile

    Returns
    -------
    typenames : dict
        Mapping from stream names to type names

    n_samples : dict
        Mapping from stream names to number of samples
    """
    log = load_log(filename)
    return summarize_log(log)


def summarize_logfiles(filename_groups, only_first_of_group=True):
    """Extract summary of log files.

    We do not count the number of samples.

    Parameters
    ----------
    filename_groups : list of list of str
        Groups of file names in chronological order. Each entry in the top-level
        list represents a group of filenames. Each group contains multiple
        files with the same streams, but they are sliced temporally. File names
        of each group have to be ordered chronologically.

    only_first_of_group : bool, optional (default: True)
        Only load the first log file of each group. We assume that all
        following files contain the same streams.

    Returns
    -------
    typenames : dict
        Mapping from stream names to type names
    """
    typenames = {}
    for group in filename_groups:
        for filename in group:
            t, _ = summarize_logfile(filename)
            typenames.update(t)
            if only_first_of_group:
                break
    return typenames


def summarize_log(log):
    """Extract summary of log file.

    Parameters
    ----------
    log : dict
        Log data

    Returns
    -------
    typenames : dict
        Mapping from stream names to type names

    n_samples : dict
        Mapping from stream names to number of samples
    """
    typenames = {}
    n_samples = {}
    for stream_name in log.keys():
        if stream_name.endswith(".meta"):
            continue
        typenames[stream_name] = log[stream_name + ".meta"]["type"]
        n_samples[stream_name] = len(log[stream_name])
    return typenames, n_samples


def chunk_log(log, stream_name, chunk_size):
    """Extract chunks from log data.

    Parameters
    ----------
    log : dict
        Log data

    stream_name : str
        Name of the stream that will be extracted

    chunk_size : int
        Maximum size of each chunk

    Returns
    -------
    chunks : list
        List of log chunks. Each chunk is log data object with a maximum of
        chunk_size samples of one stream. Chunks are ordered chronologically.
    """
    meta_key = stream_name + ".meta"
    total_size = len(log[stream_name])
    chunks = []
    for i in range(0, total_size, chunk_size):
        chunk = {
            stream_name: log[stream_name][i:i + chunk_size],
            meta_key: {
                "type": log[meta_key]["type"],
                "timestamps": log[meta_key]["timestamps"][i:i + chunk_size]
            }
        }
        chunks.append(chunk)
    return chunks


def save_chunks(chunks, filename_prefix):
    """Save log chunks in files.

    Parameters
    ----------
    chunks : list
        List of log chunks. Each chunk is log data object with a maximum of
        chunk_size samples of one stream. Chunks are ordered chronologically.

    filename_prefix : str
        Prefix of the output files. The names of output files will have the
        form '<filename_prefix>_001.msg'.
    """
    n_digits = int(math.log10(len(chunks))) + 1
    format_str = "%s_%0" + str(n_digits) + "d.msg"
    for i, chunk in enumerate(chunks):
        filename = format_str % (filename_prefix, i)
        with open(filename, "wb") as f:
            msgpack.pack(chunk, f, encoding="utf8")


def print_stream_info(log):
    """Print meta information about streams.

    Parameters
    ----------
    log : dict
        Log data
    """
    typenames, n_samples = summarize_log(log)
    stream_names = list(typenames.keys())
    stream_meta_data = [(sn, typenames[sn], str(n_samples[sn]).rjust(14))
                        for sn in stream_names]

    print("=" * 80)
    MAXLENGTHS = (35, 30, 15)
    print("".join(map(lambda t: t[0].ljust(t[1]),
                      zip(["stream name", "type", "# samples"], MAXLENGTHS))))
    print("-" * 80)
    for smd in stream_meta_data:
        line = ""
        for i in range(len(MAXLENGTHS)):
            if len(smd[i]) >= MAXLENGTHS[i]:
                line += smd[i][:MAXLENGTHS[i] - 4] + "... "
            else:
                line += smd[i].ljust(MAXLENGTHS[i])
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
        current_stream, _ = _next_timestamp(
            meta_streams, current_sample_indices)
        if current_stream is None:
            return

        if verbose >= 2:
            print("[replay] Selected stream #%d '%s'"
                  % (current_stream, stream_names[current_stream]))

        current_sample_indices[current_stream] += 1
        if verbose >= 2:
            print("[replay] Stream indices: %s" % (current_sample_indices,))

        timestamp, stream_name, typename, sample = _extract_sample(
            streams, meta_streams, stream_names, current_stream,
            current_sample_indices[current_stream])
        yield timestamp, stream_name, typename, sample


def replay_files(filename_groups, stream_names, verbose=0):
    """Generator that yields samples of multiple logs in correct temporal order.

    Parameters
    ----------
    filename_groups : list of list of str
        Groups of file names in chronological order. Each entry in the top-level
        list represents a group of filenames. Each group contains multiple
        files with the same streams, but they are sliced temporally. File names
        of each group have to be ordered chronologically.

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
    groups = [LogfileGroup(group, stream_names, verbose)
              for group in filename_groups]
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
        """Get next timestamp from any of the streams in this group.

        Returns
        -------
        timestamp : int
            Next timestamp
        """
        self._check_load_next_logfile()

        self.next_stream, timestamp = _next_timestamp(
            self.meta_streams, self.current_sample_indices)

        if self.verbose >= 2:
            if self.next_stream is None:
                print("[logloader] Reached the end of the logfile.")
            else:
                stream_name = self.group_stream_names[self.next_stream]
                print("[logloader] Next sample from stream #%d (%s), clock: %f"
                      % (self.next_stream, stream_name, timestamp))

        return timestamp

    def _check_load_next_logfile(self):
        """Check if next logfile should be loaded and do this if required."""
        for i in range(len(self.group_stream_names)):
            sample_idx = self.current_sample_indices[i]
            if sample_idx + 1 >= len(self.streams[i]):
                self._load_streams()

    def _load_streams(self):
        """Load next logfile."""
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
                    self._add_new_stream(
                        new_meta_stream, new_stream, new_stream_name)
                else:
                    self._extend_existing_stream(
                        new_meta_stream, new_stream, new_stream_name)

        if len(self.group_stream_names) == 0:
            warnings.warn(
                "Could not load any streams in this logfile group. "
                "Allowed stream names: %s; Streams in this group: %s"
                % (self.stream_names, new_stream_names)
            )

    def _delete_processed_samples(self):
        """Remove samples from the buffer that have already been processed."""
        for stream_idx in range(len(self.streams)):
            sample_idx = self.current_sample_indices[stream_idx]
            self.streams[stream_idx] = self.streams[stream_idx][sample_idx:]
            self.meta_streams[stream_idx]["timestamps"] = \
                self.meta_streams[stream_idx]["timestamps"][sample_idx:]
            self.current_sample_indices[stream_idx] = 0

    def _add_new_stream(self, new_meta_stream, new_stream, new_stream_name):
        self.group_stream_names.append(new_stream_name)
        self.streams.append(new_stream)
        self.meta_streams.append(new_meta_stream)
        self.current_sample_indices.append(-1)

    def _extend_existing_stream(self, new_meta_stream, new_stream,
                                new_stream_name):
        stream_idx = self.group_stream_names.index(new_stream_name)
        assert (
            self.meta_streams[stream_idx]["type"] ==
            new_meta_stream["type"]), (
            "%s != %s" % (self.meta_streams[stream_idx]["type"],
                          new_meta_stream["type"]))
        self.streams[stream_idx].extend(new_stream)
        self.meta_streams[stream_idx]["timestamps"].extend(
            new_meta_stream["timestamps"])

    def next_sample(self):
        """Return next sample.

        Should only be called after next_timestamp()! This function will also
        increment the sample counter for the current stream as a side effect.

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
        if self.next_stream is None:
            raise StopIteration("Reached end of logfiles.")

        if self.verbose >= 2:
            print("[logloader] Processing sample from stream %d/%d"
                  % (self.next_stream + 1, len(self.group_stream_names)))

        self.current_sample_indices[self.next_stream] += 1
        return _extract_sample(
            self.streams, self.meta_streams, self.group_stream_names,
            self.next_stream, self.current_sample_indices[self.next_stream])


def _next_timestamp(meta_streams, current_sample_indices):
    """Determine next timestamp from meta streams and sample indices.

    Parameters
    ----------
    meta_streams : dict
        Meta data streams, must contain a field "timestamps" with a list of
        timestamps

    current_sample_indices : list
        For each meta data stream the index of the current sample

    Returns
    -------
    stream_idx : int
        Index of the stream that contains the chronologically next sample

    timestamp : float
        Next timestamp
    """
    n_streams = len(current_sample_indices)

    next_timestamps = [float("inf")] * n_streams
    for i in range(n_streams):
        sample_idx = current_sample_indices[i]
        if sample_idx + 1 < len(meta_streams[i]["timestamps"]):
            next_timestamps[i] = meta_streams[i]["timestamps"][sample_idx + 1]

    if all(map(math.isinf, next_timestamps)):
        return None, float("inf")

    return _argmin(next_timestamps)


def _argmin(next_timestamps):
    return min(enumerate(next_timestamps), key=lambda p: p[1])


def _extract_sample(streams, meta_streams, stream_names, stream_idx,
                    sample_idx):
    timestamp = meta_streams[stream_idx]["timestamps"][sample_idx]
    typename = meta_streams[stream_idx]["type"]
    sample = streams[stream_idx][sample_idx]
    stream_name = stream_names[stream_idx]
    return timestamp, stream_name, typename, sample