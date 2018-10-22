import msgpack
from . import dataflowcontrol, typetodict
from collections import defaultdict


class MsgPackLogger(dataflowcontrol.LoggerBase):
    """Log to MsgPack files.

    Parameters
    ----------
    output_prefix : str
        Prefix for output files. The logger will automatically append an index
        and the ending '.msg' to each log file.

    max_samples : int, optional (default: 1000)
        Number of samples per log file

    stream_names : list, optional (default: all)
        Names of streams that will be saved
    """
    def __init__(self, output_prefix, max_samples=1000, stream_names=None):
        self.output_prefix = output_prefix
        self.max_samples = max_samples
        self.stream_names = stream_names
        self.sample_idx = 0
        self.file_idx = 0
        self._init_buffers()

    def __del__(self):
        self.save()

    def _init_buffers(self):
        self.stream_buffer = defaultdict(lambda: [])
        self.metastream_buffer = defaultdict(
            lambda: {"type": None, "timestamps": []})

    def report_node_output(self, port_name, sample, timestamp):
        if self.stream_names is not None and port_name not in self.stream_names:
            return

        # TODO decide if we do this online or as a batch
        data = typetodict.convert_to_dict(sample)
        self.stream_buffer[port_name].append(data)
        metastream = port_name + ".meta"
        self.metastream_buffer[metastream]["timestamps"].append(timestamp)
        if self.metastream_buffer[metastream]["type"] is None:
            self.metastream_buffer[metastream]["type"] = \
                sample.__class__.__name__
        self.sample_idx += 1
        if self.sample_idx >= self.max_samples:
            self.save()

    def save(self):
        self.stream_buffer.update(self.metastream_buffer)
        filename = "%s_%09d.msg" % (self.output_prefix, self.file_idx)
        with open(filename, "wb") as f:
            msgpack.pack(self.stream_buffer, f, encoding="utf8")

        self.sample_idx = 0
        self.file_idx += 1
        self._init_buffers()
