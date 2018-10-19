import os
import glob
import math
import warnings
import mmap
import pprint
import pickle
import contextlib
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


load_logfile = load_log


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
    n_samples = {}
    typenames = {}
    with mmap_readfile(filename) as m:
        unpacker = msgpack.Unpacker(file_like=m, encoding="utf8")
        n_keys = unpacker.read_map_header()
        for i in range(n_keys):
            key = unpacker.unpack()
            if key.endswith(".meta"):
                stream_name = key[:-5]
                n_meta_keys = unpacker.read_map_header()
                for j in range(n_meta_keys):
                    meta_key = unpacker.unpack()
                    if meta_key == "type":
                        typenames[stream_name] = unpacker.unpack()
                    elif meta_key == "timestamps":
                        n_samples[stream_name] = unpacker.read_array_header()
                        for k in range(n_samples[stream_name]):
                            unpacker.skip()
                    else:
                        unpacker.skip()
            else:
                unpacker.skip()
    return typenames, n_samples


def summarize_logfiles(filename_groups, only_first_of_group=True):
    """Extract summary of multiple log files given as homogeneous groups.

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


def chunk_and_save_logfile(filename, stream_name, chunk_size):
    """Split large logfile.

    Parameters
    ----------
    filename : str
        Name of the logfile

    stream_name : str
        Name of the stream that will be extracted

    chunk_size : int
        Maximum size of each chunk
    """
    output_filename = filename
    if output_filename.endswith(".msg"):
        output_filename = output_filename[:-4] + \
            stream_name.replace("/", "_").replace(".", "_")

    with mmap_readfile(filename) as m:
        metadata = _extract_metastreams(m, [stream_name])
        metastream = metadata[stream_name + ".meta"]
        m.seek(0)
        _chunk_and_save(
            stream_name, chunk_size, m, metastream, output_filename)


def _chunk_and_save(stream_name, chunk_size, m, metastream, output_filename):
    u = msgpack.Unpacker(m, encoding="utf8")
    file_counter = 0
    for _ in range(u.read_map_header()):
        key = u.unpack()
        if key == stream_name:
            n_samples = u.read_array_header()
            for i in range(0, n_samples, chunk_size):
                lo, hi = i, min(i + chunk_size, n_samples)
                sliced_samples = [u.unpack() for _ in range(lo, hi)]
                sliced_metadata = dict()
                sliced_metadata["type"] = metastream["type"]
                sliced_metadata["timestamps"] = metastream["timestamps"][lo:hi]
                outfilename = output_filename + "_%09d.msg" % file_counter
                with open(outfilename, "wb") as f:
                    p = msgpack.Packer(encoding="utf8")
                    f.write(
                        p.pack_map_pairs(
                            ((stream_name, sliced_samples),
                             (stream_name + ".meta", sliced_metadata))))
                file_counter += 1
        else:
            u.skip()


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


def print_stream_info(filename):
    """Print meta information about streams.

    Parameters
    ----------
    filename : str
        Name of the logfile
    """
    typenames, n_samples = summarize_logfile(filename)
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


def print_sample(filename, stream_name, sample_index):
    """Print individual sample from stream.

    Parameters
    ----------
    filename : str
        Name of the logfile

    stream_name : str
        Names of the stream

    sample_index : int
        Index of the sample
    """
    sample = _extract_sample_from_logfile(filename, stream_name, sample_index)
    if sample is None:
        print("Could not find sample.")
        return
    pprint.pprint(sample)


def _extract_sample_from_logfile(filename, stream_name, sample_index):
    if sample_index < 0:
        raise ValueError("sample_index must be at least 0")
    with mmap_readfile(filename) as m:
        u = msgpack.Unpacker(m, encoding="utf8")
        for _ in range(u.read_map_header()):
            key = u.unpack()
            if key == stream_name:
                n_samples = u.read_array_header()
                if sample_index >= n_samples:
                    raise ValueError("Maximum sample_index is %d"
                                     % (n_samples - 1))
                for _ in range(sample_index):
                    u.skip()
                return u.unpack()
            else:
                u.skip()
    return None


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


def replay_logfile(filename, stream_names, verbose=0):
    """Generator that yields samples of a logfile in correct temporal order.

    This is a memory-efficient version. It won't load the whole logfile at
    once.

    Parameters
    ----------
    filename : str
        Name of the logfile

    stream_names : list
        Names of the streams

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
    with mmap_readfile(filename) as m:
        current_positions, metadata = build_index(filename, m, verbose)

        try:
            meta_streams = [metadata[name + ".meta"] for name in stream_names]
        except KeyError:
            raise ValueError(
                "Mismatch between stream names %s and actual streams %s"
                % (stream_names, current_positions.keys()))

        n_streams = len(stream_names)
        current_sample_indices = [-1] * n_streams

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
                print("[replay] Stream indices: %s"
                      % (current_sample_indices,))

            current_stream_name = stream_names[current_stream]
            current_sample_idx = current_sample_indices[current_stream]
            timestamp = meta_streams[current_stream]["timestamps"][
                current_sample_idx]
            typename = meta_streams[current_stream]["type"]
            m.seek(current_positions[current_stream_name])
            u = msgpack.Unpacker(m, encoding="utf8")
            sample = u.unpack()
            current_positions[current_stream_name] += u.tell()
            yield timestamp, current_stream_name, typename, sample


def build_index(filename, m, verbose=0):
    index_filename = filename + ".cdff_idx"
    if os.path.exists(index_filename):
        if verbose >= 2:
            print("Loading cached log index from '%s'" % index_filename)
        with open(index_filename, "rb") as index_file:
            index_data = pickle.load(index_file)
            metadata = index_data["metadata"]
            current_positions = index_data["positions"]
    else:
        if verbose:
            print("Building log index...")
        metadata = _extract_metastreams(m)
        m.seek(0)
        try:
            current_positions = _extract_stream_positions(m)
        except msgpack.UnpackValueError as e:
            raise IOError("Could not build index for file '%s'. Reason: %s"
                          % (filename, e))
        with open(index_filename, "wb") as index_file:
            index = {"metadata": metadata, "positions": current_positions}
            pickle.dump(index, index_file)
        if verbose:
            print("Saving cache to '%s'" % index_filename)
    return current_positions, metadata


def group_pattern(prefix_path, pattern):
    """Get group of logfiles in lexicographical order.

    Parameters
    ----------
    prefix_path : str
        Prefix for the path of the logfiles

    pattern : str
        Pattern that should be matched to find logfiles

    Returns
    -------
    filenames : list
        List of logfiles, ordered lexicographically
    """
    files = glob.glob(prefix_path + pattern)
    if len(files) == 0:
        if prefix_path.endswith(os.sep):
            prefix_path = prefix_path[:-1]  # remove trailing '/'
        dirname = os.sep.join(prefix_path.split(os.sep)[:-1])
        if not os.path.exists(dirname):
            raise ValueError("Directory '%s' does not exist" % dirname)
        actual_files = glob.glob(os.path.join(dirname, "*"))
        raise ValueError("Could not find any matching files, only found %s"
                         % actual_files)
    return list(sorted(files))


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


def replay_join(log_iterators):
    """Generator that joins multiple log iterators.

    Parameters
    ----------
    log_iterators : list
        List of log iterators that should be joined

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
    next_samples = [next(log_iterator) for log_iterator in log_iterators]
    while True:
        next_timestamps = [next_sample[0] for next_sample in next_samples]
        if all(map(math.isinf, next_timestamps)):
            return
        log_iterator_idx, _ = _argmin(next_timestamps)
        yield next_samples[log_iterator_idx]
        try:
            next_samples[log_iterator_idx] = next(
                log_iterators[log_iterator_idx])
        except StopIteration:
            next_samples[log_iterator_idx] = (float("inf"), None, None, None)


def replay_logfile_sequence(filenames, stream_names):
    """Generator that joins multiple logfiles sequentially.

    Parameters
    ----------
    filenames : list
        List of logfiles

    stream_names : list
        Names of the streams that we want to load

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
    return replay_sequence(
        [replay_logfile(filename, stream_names) for filename in filenames])


def replay_sequence(log_iterators):
    """Generator that joins multiple log iterators sequentially.

    Parameters
    ----------
    log_iterators : list
        List of log iterators that should be joined sequentially

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
    if len(log_iterators) == 0:
        raise ValueError("Expected at least one log iterator")

    log_iterator_idx = 0
    while log_iterator_idx < len(log_iterators):
        try:
            timestamp, stream_name, typename, sample = next(
                log_iterators[log_iterator_idx])
            yield timestamp, stream_name, typename, sample
        except StopIteration:
            log_iterator_idx += 1


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
                try:
                    from cdff_envire import Time
                    t = Time()
                    t.microseconds = timestamp
                except ImportError:
                    t = timestamp
                print("[logloader] Next sample from stream #%d (%s), clock: %s"
                      % (self.next_stream, stream_name, t))

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


@contextlib.contextmanager
def mmap_readfile(filename):
    """Memory-map logfile for reading.

    Memory-mapping helps accessing a large log file on hard disk without
    reading the entire file.

    Parameters
    ----------
    filename : str
        Name of the logfile

    Returns
    -------
    m : mmap
        Memory-mapped file
    """
    with open(filename, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            yield m


def _extract_metastreams(m, streams=None):
    metastreams = dict()
    u = msgpack.Unpacker(m, encoding="utf8")
    for _ in range(u.read_map_header()):
        key = u.unpack()
        if key.endswith(".meta") and (streams is None or key[:-5] in streams):
            metastreams[key] = u.unpack()
        else:
            u.skip()
    return metastreams


def _extract_stream_positions(m, streams=None):
    positions = dict()
    u = msgpack.Unpacker(m, encoding="utf8")
    for _ in range(u.read_map_header()):
        key = u.unpack()
        if not key.endswith(".meta") and (streams is None or key in streams):
            n_samples = u.read_array_header()
            positions[key] = u.tell()
            for _ in range(n_samples):
                u.skip()
        else:
            u.skip()
    return positions


def _next_timestamp(meta_streams, current_sample_indices):
    """Determine next timestamp from meta streams and sample indices.

    Parameters
    ----------
    meta_streams : list of dicts
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
