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

    def configure(self):
        pass

    def set_time(self, timestamp):
        self._timestamp = timestamp

    def _set_transform(self, rigid_body_state):
        """Update transformation in an EnviRe graph.

        Parameters
        ----------
        rigid_body_state : RigidBodyState
            Transformation, must have source and target frame
        """
        if self.graph_ is None:
            warnings.warn("EnviRe Graph is not initialized.")
            return

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

    def _get_transform(self, origin, target):
        rigid_body_state = cdff_types.RigidBodyState()
        rigid_body_state.target_frame = origin
        rigid_body_state.source_frame = target
        if self.graph_ is None:
            warnings.warn("EnviRe Graph is not initialized.")
            return

        try:
            envireTransform = self.graph_.get_transform(origin, target)
            baseTransform = envireTransform.transform
            rigid_body_state.pos.fromarray(baseTransform.translation.toarray())
            rigid_body_state.orient.fromarray(
                baseTransform.orientation.toarray())
            rigid_body_state.timestamp.microseconds = self._timestamp
        except RuntimeError as e:
            print("[EnvireGraph] ERROR: %s" % e)

        return rigid_body_state

    def process(self):
        pass