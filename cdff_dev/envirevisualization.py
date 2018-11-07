import sys
import warnings
from PyQt4.QtGui import QApplication
from . import dataflowcontrol, qtgui
import cdff_envire
import cdff_types


class EnvireVisualizerApplication:
    """Qt Application with EnviRe visualizer.

    Parameters
    ----------
    frames : dict
        Mapping from port names to frame names

    urdf_files : list, optional (default: [])
        URDF files that should be loaded

    center_frame : str, optional (default: first in lexical order)
        Frame that represents the center of visualization
    """
    def __init__(self, frames, urdf_files=(), center_frame=None):
        self.app = QApplication(sys.argv)
        self.visualization = EnvireVisualization(
            frames, urdf_files, center_frame)
        self.control_window = None

    def __del__(self):
        # Make sure to remove all items before visualizer is deleted,
        # otherwise the published events will result in a segfault!
        self.visualization.world_state_.remove_all_items()
        # Visualizer must be deleted before graph, otherwise
        # unsubscribing from events will result in a segfault!
        del self.visualization.visualizer

    def show_controls(self, iterator, dfc):
        """Show control window to replay log file.

        Parameters
        ----------
        iterator : Iterable
            Iterable object that yields log samples in the correct temporal
            order. The iterable returns in each step a quadrupel of
            (timestamp, stream_name, typename, sample).

        dfc : DataFlowControl
            Configured processing and data fusion logic
        """
        dfc.register_visualization(self.visualization)
        dfc.register_world_state(self.visualization.world_state_)
        self.control_window = qtgui.ReplayMainWindow(qtgui.Step, iterator, dfc)
        self.control_window.show()

    def exec_(self):
        """Start Qt application.

        Qt will take over the main thread until the main window is closed.
        """
        self.app.exec_()


class EnvireVisualization(dataflowcontrol.VisualizationBase):
    """EnviRe visualization.

    Parameters
    ----------
    frames : dict
        Mapping from port names to frame names

    urdf_files : list, optional (default: [])
        URDF files that should be loaded

    center_frame : str, optional (default: first in lexical order)
        Frame that represents the center of visualization
    """
    def __init__(self, frames, urdf_files=(), center_frame=None):
        self.world_state_ = WorldState(frames, urdf_files)
        self.visualizer = cdff_envire.EnvireVisualizer()
        if center_frame is None:
            center_frame = sorted(frames.values())[0]
        if not self.world_state_.graph_.contains_frame(center_frame):
            self.world_state_.graph_.add_frame(center_frame)
        self.visualizer.display_graph(self.world_state_.graph_, center_frame)
        self.visualizer.show()

    def report_node_output(self, port_name, sample, timestamp):
        self.world_state_.report_node_output(port_name, sample, timestamp)
        self.visualizer.redraw()


class WorldState:
    """Represents the estimated world state of the system based on log data.

    Parameters
    ----------
    frames : dict
        Mapping from port names to frame names

    urdf_files : list
        URDF files that should be loaded
    """
    def __init__(self, frames, urdf_files):
        self.frames = frames
        self.items = dict()
        self.graph_ = cdff_envire.EnvireGraph()
        self.urdf_models = []
        for filename in urdf_files:
            urdf_model = cdff_envire.EnvireURDFModel()
            urdf_model.load_urdf(self.graph_, filename, load_visuals=True)
            self.urdf_models.append(urdf_model)
        for frame in self.frames.values():
            if not self.graph_.contains_frame(frame):
                self.graph_.add_frame(frame)

    def __del__(self):
        if self.items:
            self.remove_all_items()
        del self.graph_

    def remove_all_items(self):
        """Remove all items from EnviRe graph.

        This has to be done manually before any attached visualizer is deleted.
        """
        for item in self.items.values():
            if item is not None:
                item.remove_from(self.graph_)
        self.items = {}

    def report_node_output(self, port_name, sample, timestamp):
        if sample is None:
            warnings.warn("No sample on the output port '%s'" % port_name)
            return
        elif port_name not in self.frames:
            warnings.warn("No frame registered for port '%s'" % port_name)
            return

        if port_name in self.items:
            if self.items[port_name] is None:
                return
            else:
                self.items[port_name].set_data(sample)
        else:
            item = EnvireItem(sample)
            try:
                item.add_to(self.graph_, self.frames[port_name])
            except TypeError as e:
                warnings.warn("Cannot store type '%s' in EnviRe graph. "
                              "Reason: %s" % (type(sample), e))
                self.items[port_name] = None
                return
            except ValueError as e:
                warnings.warn("Cannot store type '%s' in EnviRe graph. "
                              "Reason: %s" % (type(sample), e))
                self.items[port_name] = None
                return
            self.items[port_name] = item
        self.items[port_name].set_time(timestamp)


class EnvireItem:
    """Stores an EnviRe item with its corresponding content.

    NOTE: This is a hack that is required to identify the correct
    template type.
    """
    def __init__(self, sample):
        self.sample = sample
        self.item = cdff_envire.GenericItem()

    def set_data(self, sample):
        self.item.set_data(sample)
        self.sample = sample

    def set_time(self, timestamp):
        self.item.set_time(self.sample, timestamp)

    def add_to(self, graph, frame):
        graph.add_item_to_frame(frame, self.item, self.sample)

    def remove_from(self, graph):
        graph.remove_item_from_frame(self.item, self.sample)
