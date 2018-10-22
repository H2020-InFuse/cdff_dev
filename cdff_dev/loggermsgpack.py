import msgpack
from . import dataflowcontrol, typetodict
from collections import defaultdict


class MsgPackLogger(dataflowcontrol.LoggerBase):  # TODO test
    """TODO"""
    def __init__(self, output_prefix):
        self.output_prefix = output_prefix
        # TODO save only some streams
        # TODO save after n samples
        self.file_idx = 0
        self._init_buffers()

    def __del__(self):
        self.save()

    def _init_buffers(self):
        self.stream_buffer = defaultdict(lambda: [])
        self.metastream_buffer = defaultdict(
            lambda: {"type": None, "timestamps": []})

    def report_node_output(self, port_name, sample, timestamp):
        # TODO decide if we do this online or as a batch
        data = typetodict.convert_to_dict(sample)
        self.stream_buffer[port_name].append(data)
        metastream = port_name + ".meta"
        self.metastream_buffer[metastream]["timestamps"].append(timestamp)
        if self.metastream_buffer[metastream]["type"] is None:
            self.metastream_buffer[metastream]["type"] = \
                sample.__class__.__name__

    def save(self):
        self.stream_buffer.update(self.metastream_buffer)
        filename = "%s_%09d.msg" % (self.output_prefix, self.file_idx)
        with open(filename, "wb") as f:
            msgpack.pack(self.stream_buffer, f, encoding="utf8")

        self.file_idx += 1
        self._init_buffers()
