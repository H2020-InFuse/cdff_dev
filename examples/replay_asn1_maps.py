import warnings
from cdff_dev import logloader, dataflowcontrol, envirevisualization, transformer
import cdff_envire


class Transformer(transformer.EnvireDFN):
    def __init__(self):
        super(Transformer, self).__init__()

    def _set_transform_with_cov(self, tf):  # TODO move to base
        """Update transformation in an EnviRe graph.

        Parameters
        ----------
        tf : TransformWithCovariance
            Transformation

        TODO
        """
        if self.graph_ is None:
            warnings.warn("EnviRe Graph is not initialized.")
            return

        if True:  # TODO
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
        self._set_transform_with_cov(data)


def main():
    app = envirevisualization.EnvireVisualizerApplication(
        frames={
            "dem_building.fusedMap": "LocalTerrainFrame",
            "pom.pose": "LocalTerrainFrame",
        },
        center_frame="LocalTerrainFrame"
    )

    stream_aliases = {
        "/dem_building/fusedMap": "dem_building.fusedMap",
        "/pom_pose": "pom.pose",
    }
    nodes = {"transformer": Transformer()}
    connections = (
        ("pom.pose", "transformer.pom_pose"),
    )
    dfc = dataflowcontrol.DataFlowControl(
        nodes=nodes, connections=connections,
        periods={"transformer": 1.0},
        stream_aliases=stream_aliases, verbose=2)
    dfc.setup()

    # Note that the logfiles are not in the repository because they are too
    # large. Ask Alexander Fabisch about it.
    folder = "/home/dfki.uni-bremen.de/afabisch/Projekte/ros_workspace/src/CDFF_ROS/rosbag2msgpack/logs_maps/"
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
