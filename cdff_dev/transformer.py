import warnings
import cdff_envire
import cdff_types


# TODO at the moment, we can only handle RigidBodyStates!
#      other options: envire.Transform, basetypes.TransformWithCov...
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
        self.graph_ = None
        self._timestamp = 0
        self.verbose = verbose

    def set_configuration_file(self):
        pass

    def configure(self):
        pass

    def set_time(self, timestamp):
        self._timestamp = timestamp

    def _set_transform(self, rigid_body_state, data_transformation=True):
        """Update transformation in an EnviRe graph.

        Parameters
        ----------
        rigid_body_state : RigidBodyState
            Transformation, must have source and target frame

        data_transformation : bool, optional (default: True)
            A data transformation transforms data from a source frame to a
            target frame. The opposite is a frame transformation that
            represents the transformation of a source frame to a target frame
            or tells us where a target frame is in the source frame.
        """
        if self.graph_ is None:
            warnings.warn("EnviRe Graph is not initialized.")
            return

        if data_transformation:
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

    def _get_transform(self, origin, target, data_transformation=True):
        """Get transformation from an EnviRe graph.

        Parameters
        ----------
        origin : str
            Source frame

        target : str
            Target frame

        data_transformation : bool, optional (default: True)
            A data transformation transforms data from a source frame to a
            target frame. The opposite is a frame transformation that
            represents the transformation of a source frame to a target frame
            or tells us where a target frame is in the source frame.
        """
        if self.graph_ is None:
            warnings.warn("EnviRe graph is not initialized.")
            return

        rigid_body_state = cdff_types.RigidBodyState()
        if data_transformation:
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