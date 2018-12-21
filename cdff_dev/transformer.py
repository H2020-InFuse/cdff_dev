import warnings
import msgpack
import cdff_envire
import cdff_types


# TODO at the moment, we can only handle RigidBodyStates!
#      other options: envire.Transform, cdff_types.TransformWithCov...
class EnvireDFN:
    """Data Fusion Node that has access to an EnviRe graph.

    This can be used as a base class for nodes that provide transformations.

    Parameters
    ----------
    verbose : int, optional (default: 0)
        Verbosity level

    Attributes
    ----------
    graph_ : EnvireGraph
        Graph that stores and provides transformations
    """
    def __init__(self, verbose=0):
        self.verbose = verbose

        self._graph = None
        self._configuration_file = None
        self._config = None
        self._timestamp = 0

    def _set_graph_(self, graph):
        self._graph = graph
        if graph is not None:
            self.initialize_graph(graph)
        if self._config is not None:
            self.initialize_graph_from_config(graph, self._config)

    def _get_graph_(self):
        return self._graph

    graph_ = property(_get_graph_, _set_graph_)

    def initialize_graph(self, graph):
        """Initialize graph.

        This method will be called after the graph object has been set.
        It can be used to add frames and static transformations.

        Parameters
        ----------
        graph : EnvireGraph
            New graph object
        """

    def initialize_graph_from_config(self, graph, config, prefix="config_"):
        """Initialize graph with static transformations from configuration.

        Parameters
        ----------
        graph : EnvireGraph
            EnviRe graph that will be initialized

        config : dict
            Each key is a tuple of names of the source and target frames.
            Each value is a dict with the entries 'translation' and
            'rotation'. The rotation is represented by a quaternion in
            scalar last convention (x, y, z, w).

        prefix : str, optional (default: 'config_')
            Frame names from the configuration file will be prefixed with this
            string.
        """
        for frames, transformation in config.items():
            source, target = frames
            t = cdff_envire.Transform(
                translation=cdff_envire.Vector3d(
                    *transformation["translation"]),
                orientation=cdff_envire.Quaterniond(
                    *transformation["rotation"])
            )
            graph.add_transform(prefix + source, prefix + target, t)

    def set_configuration_file(self, filename):
        self._configuration_file = filename

    def configure(self):
        """Read configuration file and initialize graph if it exists already."""
        if self._configuration_file is None:
            return

        if self._configuration_file.endswith(".msg"):
            with open(self._configuration_file, "rb") as f:
                self._config = msgpack.unpack(
                    f, encoding="utf8", use_list=False)
        else:
            raise NotImplementedError(
                "Don't know how to handle the configuration file format.")

        if self._graph is not None:
            self.initialize_graph_from_config(self._graph, self._config)

    def set_time(self, timestamp):
        self._timestamp = timestamp

    def _set_transform(self, rigid_body_state, frame_transformation=True):
        """Update transformation in an EnviRe graph.

        Parameters
        ----------
        rigid_body_state : RigidBodyState
            Transformation, must have source and target frame

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
            origin = rigid_body_state.source_frame
            target = rigid_body_state.target_frame
        else:
            origin = rigid_body_state.target_frame
            target = rigid_body_state.source_frame

        if not self.graph_.contains_frame(origin):
            self.graph_.add_frame(origin)
        if not self.graph_.contains_frame(target):
            self.graph_.add_frame(target)

        t = cdff_envire.TransformWithCovariance()
        t.translation.fromarray(rigid_body_state.pos.toarray())
        t.orientation.fromarray(rigid_body_state.orient.toarray())
        timestamp = cdff_envire.Time()
        timestamp.microseconds = rigid_body_state.timestamp.microseconds
        transform = cdff_envire.Transform(
            time=timestamp, transform_with_covariance=t)
        if self.graph_.contains_edge(origin, target):
            self.graph_.update_transform(origin, target, transform)
        else:
            self.graph_.add_transform(origin, target, transform)

    def _get_transform(self, origin, target, frame_transformation=True):
        """Get transformation from an EnviRe graph.

        Parameters
        ----------
        origin : str
            Source frame

        target : str
            Target frame

        frame_transformation : bool, optional (default: True)
            A frame transformation represents the transformation of a source
            frame to a target frame or tells us where a target frame is in
            the source frame. The opposite is a data transformation that
            transforms data from a source frame to a target frame.
        """
        if self.graph_ is None:
            warnings.warn("EnviRe graph is not initialized.")
            return

        rigid_body_state = cdff_types.RigidBodyState()
        if frame_transformation:
            rigid_body_state.target_frame = target
            rigid_body_state.source_frame = origin
        else:
            rigid_body_state.target_frame = origin
            rigid_body_state.source_frame = target

        try:
            envire_transform = self.graph_.get_transform(
                rigid_body_state.source_frame, rigid_body_state.target_frame)
            base_transform = envire_transform.transform
            rigid_body_state.pos.fromarray(base_transform.translation.toarray())
            rigid_body_state.orient.fromarray(
                base_transform.orientation.toarray())
            rigid_body_state.timestamp.microseconds = self._timestamp
        except RuntimeError as e:
            print("[EnvireGraph] ERROR: %s" % e)

        return rigid_body_state

    def process(self):
        pass