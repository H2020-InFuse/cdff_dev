import warnings
import math
import numpy as np
from cdff_dev import logloader, dataflowcontrol, envirevisualization, transformer, dfnhelpers
import cdff_envire
import cdff_types


class MapFilter(dfnhelpers.DFNBase):
    def __init__(self):
        self.map = cdff_types.Map()

    def mapInput(self, data):
        self.map.metadata.time_stamp.microseconds = \
            data.metadata.time_stamp.microseconds
        self.map.metadata.scale = data.metadata.scale
        self.map.data.rows = data.data.rows
        self.map.data.cols = data.data.cols
        self.map.data.channels = data.data.channels
        self.map.data.depth = data.data.depth
        self.map.data.row_size = data.data.row_size
        self.map.data.array_reference()[:, :, :] = \
            data.data.array_reference()[:, :, :]

    def process(self):
        not_nan = self.map.data.array_reference() != 0.0
        self.map.data.array_reference()[not_nan] += 50.0

    def filteredMapOutput(self):
        return self.map


class Transformer(transformer.EnvireDFN):
    def __init__(self):
        super(Transformer, self).__init__()

    def initialize_graph(self, graph):
        pass

    def _set_transform_with_cov(self, tf, frame_transformation=True):  # TODO move to base
        """Update transformation in an EnviRe graph.

        Parameters
        ----------
        tf : TransformWithCovariance
            Transformation

        frame_transformation : bool, optional (default: True)
            A frame transformation represents the transformation of a source
            frame to a target frame or tells us where a target frame is in
            the source frame. The opposite is a data transformation that
            transforms data from a source frame to a target frame.
        """
        if self.graph_ is None:
            warnings.warn("EnviRe Graph is not initialized.")
            return

        if frame_transformation:
            origin = tf.metadata.parent_frame_id
            target = tf.metadata.child_frame_id
        else:
            origin = tf.metadata.child_frame_id
            target = tf.metadata.parent_frame_id

        if not self.graph_.contains_frame(origin):
            self.graph_.add_frame(origin)
        if not self.graph_.contains_frame(target):
            self.graph_.add_frame(target)

        t = cdff_envire.TransformWithCovariance()
        t.translation.fromarray(tf.data.translation.toarray())
        t.orientation.fromarray(tf.data.orientation.toarray())
        timestamp = cdff_envire.Time()
        timestamp.microseconds = tf.metadata.parent_time.microseconds
        transform = cdff_envire.Transform(
            time=timestamp, transform_with_covariance=t)
        if self.graph_.contains_edge(origin, target):
            self.graph_.update_transform(origin, target, transform)
        else:
            self.graph_.add_transform(origin, target, transform)

    def pom_poseInput(self, data):
        self._set_transform_with_cov(data, frame_transformation=True)

    def map_transformInput(self, data):
        self._set_transform_with_cov(data.metadata.pose_fixed_frame_map_frame,
                                     frame_transformation=True)


class RBSConverter(dfnhelpers.DFNBase):
    def __init__(self):
        self.transform_with_covariance = None
        self.output = cdff_types.RigidBodyState()

    def transformWithCovarianceInput(self, data):
        self.transform_with_covariance = data

    def process(self):
        if self.transform_with_covariance is None:
            return
        pos = self.transform_with_covariance.data.translation.toarray()
        pos[-1] += 50.0
        orient = self.transform_with_covariance.data.orientation.toarray()
        self.output.pos.fromarray(pos)
        self.output.orient.fromarray(orient)

    def rbsOutput(self):
        return self.output


def main():
    app = envirevisualization.EnvireVisualizerApplication(
        frames={
            "map_filter.filteredMap": "MapFrame",
            "rbs_converter.rbs": "LocalTerrainFrame",
        },
        center_frame="MapFrame"
    )

    stream_aliases = {
        "/dem_building/fusedMap": "dem_building.fusedMap",
        "/pose_robot_pom": "pose_robot.pom",
    }
    nodes = {
        "transformer": Transformer(),
        "rbs_converter": RBSConverter(),
        "map_filter": MapFilter(),
    }
    connections = (
        ("pose_robot.pom", "rbs_converter.transformWithCovariance"),
        ("pose_robot.pom", "transformer.pom_pose"),
        ("dem_building.fusedMap", "transformer.map_transform"),
        ("dem_building.fusedMap", "map_filter.map"),
    )
    dfc = dataflowcontrol.DataFlowControl(
        nodes=nodes, connections=connections,
        periods={"transformer": 1.0,
                 "map_filter": 10.0},
        trigger_ports={"rbs_converter": "transformWithCovariance"},
        stream_aliases=stream_aliases, verbose=2)
    dfc.setup()

    folder = "logs/LAAS_DEM/"
    log_iterator = logloader.replay_logfile_sequence(
        logloader.group_pattern(folder, "20181203*.msg"), stream_names=[
            "/dem_building/fusedMap",
            "/pose_robot_pom",
        ])
    app.show_controls(log_iterator, dfc)
    app.exec_()

    dfc.node_statistics_.print_statistics()


if __name__ == "__main__":
    main()

