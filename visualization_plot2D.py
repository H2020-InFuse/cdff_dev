import glob
import matplotlib.pyplot as plt
from matplotlib import animation
import visualization_log_replay
import threading
from matplotlib.ticker import ScalarFormatter
#from tkinter import Tk
#from tkinter.filedialog import askopenfilename
from matplotlib.widgets import SpanSelector
import matplotlib.image as mpimg
from cdff_dev import dataflowcontrol


# open a tk dialogue box to select log file
#Tk().withdraw()
#log_file = askopenfilename()

class MatplotlibVisualizer:
    def __init__(self, dfc, log_files, stream_names):
        self.dfc = dfc
        self.log_files = log_files
        self.stream_names = stream_names

        self.vdh = visualization_log_replay.VisualizationDataHandler()

        # create figure, subplots
        self.fig = plt.figure()
        self.ax = plt.subplot(111)

        # set axis titles
        self.ax.set_ylabel("units")
        self.ax.set_xlabel("Time (seconds)")

        # format axis labels
        xfmt = ScalarFormatter(useMathText=True)
        self.ax.yaxis.set_major_locator(plt.MaxNLocator(10))
        self.ax.yaxis.set_major_formatter(xfmt)
        self.ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
        self.ax.xaxis.set_major_locator(plt.MaxNLocator(10))

        plt.tight_layout()

        blank = mpimg.imread("Blank.png")

        span = SpanSelector(
            self.ax, self.vdh.onselect, 'horizontal', useblit=True,
            rectprops=dict(alpha=0.5, facecolor='red'))

        """removes blinking from span selection, but data is not displayed during selection"""
        def onclick(event):
            self.anim.event_source.stop()
        #cid = self.fig.canvas.mpl_connect('button_press_event', onclick)

        # This is the only part that limits how many types of data
        # can be graphed at once
        line0, = self.ax.plot([], color="b", linewidth=0.5, alpha=0.75)
        line1, = self.ax.plot([], color="r", linewidth=0.5, alpha=0.75)
        line2, = self.ax.plot([], color="g", linewidth=0.5, alpha=0.75)
        line3, = self.ax.plot([], color="darkorange", linewidth=0.5, alpha=0.75)
        marker, = self.ax.plot([], color="k", marker="+")
        self.line = [line0, line1, line2, line3, marker]

        # Setting blit to False renders animation useless - only grey window displayed
        # Blitting may have some connection to removing blinking from animation
        self.anim = animation.FuncAnimation(
            self.fig, visualization_log_replay.animate,
            fargs=(self.line, self.ax, self.vdh), interval=0.0, blit=True)

    def exec_(self):
        # begin log replay on a separate thread in order to run concurrently
        thread = threading.Thread(
            target=visualization_log_replay.main,
            args=(self.dfc, self.vdh, self.log_files, self.stream_names))
        thread.start()

        # display plotting window
        try:
            plt.show()
        except AttributeError:
            pass
        plt.close()


nodes = {}
periods = {}
connections = []
dfc = dataflowcontrol.DataFlowControl(nodes, connections, periods)

log_files = [
    #["test/test_data/logs/frames.msg"],
    sorted(glob.glob("logs/open_day/open_day_xsens_imu_*.msg")),
    #sorted(glob.glob("logs/open_day/open_day_laser_filter_*.msg")),
    #sorted(glob.glob("logs/open_day/open_day_tilt_scan_*.msg")),
    #sorted(glob.glob("logs/open_day/open_day_dynamixel_*.msg")),
    #sorted(glob.glob("logs/open_day/open_day_velodyne_*.msg"))
]
stream_names = [
    #"/laser_filter.filtered_scans",
    #"/velodyne.laser_scans",
    #"/tilt_scan.pointcloud",
    #"/dynamixel.transforms",
    #"/camera1.frame"
    "/xsens_imu.calibrated_sensors"
]

vis = MatplotlibVisualizer(dfc, log_files, stream_names)
vis.exec_()