import numpy as np
from cdff_dev import (logloader, typefromdict, dataflowcontrol, replay,
                      transformer, envirevisualization, dfnhelpers)
import cdff_types
from nose.tools import assert_equal, assert_in
from numpy.testing import assert_array_less


def test_replay():
    log = logloader.load_log("test/test_data/logs/test_log.msg")
    stream_names = ["/hokuyo.scans", "/dynamixel.transforms",
                    "/dynamixel.status_samples"]
    timestamps = []
    step_index = 0
    for timestamp, stream_name, typename, sample in logloader.replay(
            stream_names, log, verbose=0):
        if step_index >= 100:
            break
        step_index += 1
        timestamps.append(timestamp)
        obj = typefromdict.create_from_dict(typename, sample)
        assert_equal(type(obj).__module__, "cdff_types")
    assert_array_less(timestamps[:-1], timestamps[1:])


class Transformer(transformer.EnvireDFN):
    def __init__(self):
        super(Transformer, self).__init__()


class GraphVisualizer(dataflowcontrol.VisualizationBase):
    """Simulates EnviRe visualizer without GUI."""
    def __init__(self, world_state):
        self.world_state = world_state

    def report_node_output(self, port_name, sample, timestamp):
        self.world_state.report_node_output(port_name, sample, timestamp)


class LaserFilterDummyDFN:
    def __init__(self):
        self.scanSample = cdff_types.LaserScan()

    def set_configuration_file(self, filename):
        pass

    def configure(self):
        pass

    def scanSampleInput(self, scanSample):
        self.scanSample = scanSample

    def process(self):
        # filter all values that are too far away from the median
        ranges = np.array([self.scanSample.ranges[i]
                           for i in range(self.scanSample.ranges.size())])
        med = np.median(ranges)
        outliers = np.where(np.abs(ranges - med) > 200)[0]
        for i in outliers:
            self.scanSample.ranges[i] = med

    def filteredScanOutput(self):
        return self.scanSample


class PointcloudBuilderDummyDFN:
    def __init__(self):
        self.pointcloud = cdff_types.Pointcloud()

    def set_configuration_file(self, filename):
        pass

    def configure(self):
        pass

    def scanInput(self, scan):
        self.scan = scan

    def transformInput(self, transform):
        self.transform = transform

    def process(self):
        pass

    def pointcloudOutput(self):
        return self.pointcloud


def test_feed_data_flow_control():
    nodes = {
        "laser_filter": LaserFilterDummyDFN(),
        "pointcloud_builder": PointcloudBuilderDummyDFN(),
        "transformer": Transformer(),
        "extract_size": dfnhelpers.LambdaDFN(
            lambda x: x.ranges.size(), "scanSample", "size")
    }
    periods = {
        "laser_filter": 0.025,
        "pointcloud_builder": 0.1,
        "transformer": 1.0,
        "extract_size": 0.025,
    }
    connections = (
        ("/hokuyo.scans", "laser_filter.scanSample"),
        ("laser_filter.filteredScan", "pointcloud_builder.scan"),
        ("laser_filter.filteredScan", "extract_size.scanSample"),
        ("/dynamixel.transforms", "pointcloud_builder.transform"),
        ("pointcloud_builder.pointcloud", "result.pointcloud")
    )
    dfc = dataflowcontrol.DataFlowControl(nodes, connections, periods)
    dfc.setup()
    world_state = envirevisualization.WorldState(
        frames={"laser_filter.filteredScan": "center",
                "extract_size.size": "center"}, urdf_files=[])
    dfc.register_world_state(world_state)
    graph_vis = GraphVisualizer(world_state)
    dfc.set_visualization(graph_vis)
    vis = dataflowcontrol.NoVisualization()
    dfc.set_visualization(vis)

    log = logloader.load_log("test/test_data/logs/test_log.msg")
    stream_names = ["/hokuyo.scans", "/dynamixel.transforms"]
    replay.replay_and_process(
        dfc, logloader.replay(stream_names, log), max_samples=100)
    assert_in("laser_filter.filteredScan", vis.data)
    assert_in("pointcloud_builder.pointcloud", vis.data)


def test_feed_data_flow_control_async():
    nodes = {
        "laser_filter": LaserFilterDummyDFN(),
        "pointcloud_builder": PointcloudBuilderDummyDFN()
    }
    periods = {
        "laser_filter": 0.025,
        "pointcloud_builder": 0.1
    }
    connections = (
        ("/hokuyo.scans", "laser_filter.scanSample"),
        ("laser_filter.filteredScan", "pointcloud_builder.scan"),
        ("/dynamixel.transforms", "pointcloud_builder.transform"),
        ("pointcloud_builder.pointcloud", "result.pointcloud")
    )
    dfc = dataflowcontrol.DataFlowControl(nodes, connections, periods)
    dfc.setup()
    vis = dataflowcontrol.NoVisualization()
    dfc.set_visualization(vis)

    log = logloader.load_log("test/test_data/logs/test_log.msg")
    stream_names = ["/hokuyo.scans", "/dynamixel.transforms"]
    replay.replay_and_process_async(
        dfc, logloader.replay(stream_names, log), queue_size=10,
        max_samples=100)
    assert_in("laser_filter.filteredScan", vis.data)
    assert_in("pointcloud_builder.pointcloud", vis.data)
