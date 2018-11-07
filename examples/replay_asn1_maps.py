import warnings
from cdff_dev import logloader, dataflowcontrol, envirevisualization, transformer
import cdff_envire
import cdff_types


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


class RBSConverter:  # TODO we need a dummy base class for this...
    def __init__(self):
        self.transform_with_covariance = None
        self.output = cdff_types.RigidBodyState()
    def set_configuration_file(self, filename):
        raise NotImplementedError("No configuration supported")
    def configure(self):
        pass
    def transformWithCovarianceInput(self, data):
        self.transform_with_covariance = data
    def process(self):
        if self.transform_with_covariance is None:
            return
        pos = self.transform_with_covariance.data.translation.toarray()
        orient = self.transform_with_covariance.data.orientation.toarray()
        self.output.pos.fromarray(pos)
        self.output.orient.fromarray(orient)
    def rbsOutput(self):
        return self.output


def main():
    app = envirevisualization.EnvireVisualizerApplication(
        frames={
            "dem_building.fusedMap": "MapFrame",
            "rbs_converter.rbs": "LocalTerrainFrame",
        },
        center_frame="LocalTerrainFrame"
    )

    stream_aliases = {
        "/dem_building/fusedMap": "dem_building.fusedMap",
        "/pom_pose": "pom.pose",
    }
    nodes = {
        "transformer": Transformer(),
        "rbs_converter": RBSConverter()
    }
    connections = (
        ("pom.pose", "rbs_converter.transformWithCovariance"),
        ("pom.pose", "transformer.pom_pose"),
        ("dem_building.fusedMap", "transformer.map_transform"),
    )
    dfc = dataflowcontrol.DataFlowControl(
        nodes=nodes, connections=connections,
        periods={"transformer": 1.0},
        trigger_ports={"rbs_converter": "transformWithCovariance"},
        stream_aliases=stream_aliases, verbose=2)
    dfc.setup()

    # Note that the logfiles are not in the repository because they are too
    # large. Ask Alexander Fabisch about it.
    folder = "logs/20181031_LAAS_Maps/2018-10-31_"
    log_iterator = logloader.replay_logfile_sequence(
        logloader.group_pattern(folder, "*.msg"), stream_names=[
            "/dem_building/fusedMap",
            "/pom_pose",
        ])
    app.show_controls(log_iterator, dfc)
    app.exec_()

    dfc.node_statistics_.print_statistics()


if __name__ == "__main__":
    main()
