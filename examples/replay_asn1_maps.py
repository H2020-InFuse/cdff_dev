import warnings
from cdff_dev import logloader, dataflowcontrol, envirevisualization, transformer
import cdff_types
import cdff_envire


class Transformer(transformer.EnvireDFN):
    def __init__(self):
        super(Transformer, self).__init__()

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


class HotfixMap:
    def __init__(self, downsampling_factor=1):
        self.downsampling_factor = downsampling_factor
        self.map = None
        self.small_map = cdff_types.Map()

    def set_configuration_file(self, filename):
        pass

    def configure(self):
        pass

    def mapInput(self, data):
        self.map = data

    def process(self):
        if self.map is None:
            return

        self.small_map.metadata.type = self.map.metadata.type
        self.small_map.metadata.scale = self.map.metadata.scale * self.downsampling_factor
        self.small_map.data.rows = self.map.data.rows // self.downsampling_factor
        self.small_map.data.cols = self.map.data.cols // self.downsampling_factor
        self.small_map.data.channels = self.map.data.channels
        self.small_map.data.depth = self.map.data.depth
        # HACK fix incorrect row_size from logfile
        self.small_map.data.row_size = self.map.data.cols
        self.small_map.data.array_reference()[:, :, :] = \
            self.map.data.array_reference()[::self.downsampling_factor,
                                            ::self.downsampling_factor, :]

    def smallMapOutput(self):
        return self.small_map


def main():
    app = envirevisualization.EnvireVisualizerApplication(
        frames={
            "hotfix_map.smallMap": "MapFrame",
            "pom.pose": "LocalTerrainFrame",
        },
        center_frame="LocalTerrainFrame"
    )

    stream_aliases = {
        "/dem_building/fusedMap": "dem_building.fusedMap",
        "/pom_pose": "pom.pose",
    }
    nodes = {"transformer": Transformer(),
             "hotfix_map": HotfixMap(8)}
    connections = (
        ("pom.pose", "transformer.pom_pose"),
        ("dem_building.fusedMap", "hotfix_map.map"),
        ("dem_building.fusedMap", "transformer.map_transform"),
    )
    dfc = dataflowcontrol.DataFlowControl(
        nodes=nodes, connections=connections,
        periods={"transformer": 1.0},
        trigger_ports={"hotfix_map": "map"},
        stream_aliases=stream_aliases, verbose=2)
    dfc.setup()

    # Note that the logfiles are not in the repository because they are too
    # large. Ask Alexander Fabisch about it.
    folder = "logs/20181031_LAAS_Maps/"
    #filename = folder + "2018-10-31-17-15-49_0_000000000.msg"
    filename = folder + "2018-10-31-17-18-12_14_000000000.msg"
    log_iterator = logloader.replay_logfile(
        filename, stream_names=[
            "/dem_building/fusedMap",
            "/pom_pose",
        ])
    app.show_controls(log_iterator, dfc)
    app.exec_()

    dfc.node_statistics_.print_statistics()


if __name__ == "__main__":
    main()
