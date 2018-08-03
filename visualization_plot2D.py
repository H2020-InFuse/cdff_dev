import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import visualization_log_replay
import threading
from matplotlib.ticker import ScalarFormatter
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from matplotlib.widgets import SpanSelector
import matplotlib.image as mpimg

#TODO: better name
class Coordinates():
    """Instantiate and maintain basic coordinate aspects of the plot.
    These points are stored for use by other functions. 

    Attributes
    ---------
    span_selected : list
        stores the range of X-values selected by the user from onselect function

    data_min/data_max : float
        min/max value from Y-Axis data

    """

    def __init__(self):
        self.span_selected = None
        self.data_min = float("inf")
        self.data_max = -float("inf")

    def yrange_reset(self):
        self.data_min = float("inf")
        self.data_max = -float("inf")


def set_axis_limits(ax, vdh, time_list, data_lists, list_assigner, coords):
    """Control axis limits that change dynamically with data and user selections 
    """
    control_panel = vdh.control_panel
    
    # find highest and lowest data points out of all lists given
    for the_list in data_lists:
        coords.data_min = min(
            np.amin(the_list), coords.data_min)
        coords.data_max = max(
            np.amax(the_list), coords.data_max)

    # set x and y limits based on current highest and lowest data points
    if data_lists:
        margin = max(np.abs(coords.data_min),
                     coords.data_max) * 0.05
        ax.set_ylim(coords.data_min - margin,
                    coords.data_max + margin)

    if vdh.source_dict and vdh.control_panel.stream_reset:
        coords.yrange_reset()
        control_panel.stream_reset = False

    if vdh.control_panel.yrange_reset:
        coords.yrange_reset()
        control_panel.yrange_reset = False

    ax.set_xlim(time_list[0], np.amax(time_list))


def animate(i, line, ax, vdh, coords, camera_1, camera_2):
    """ Update and plot line data. Anything that needs to be
    updated must lie within this function. Repeatedly called 
    by FuncAnimation, incrementing i with each iteration. 
    """
    time_list = vdh.time_list
    list_assigner = vdh.list_assigner
    data_lists = list_assigner.get_data()
    data_labels = list_assigner.get_labels()

    # pass data from lists to lines
    if len(data_lists) > 0:
        set_axis_limits(ax, vdh,
                        time_list, data_lists, list_assigner, coords)

        try:
            for index, the_list in enumerate(data_lists):
                if len(the_list) != len(time_list):
                    raise ValueError(
                        "Y data and X data have different lengths: ", "X: ", time_list, "Y: ", the_list)
                x = time_list[0:i]
                y = the_list[0:i]
                line[index].set_data(x, y)
        except ValueError:
            pass

        try:
            if vdh.control_panel.remove_outlier:
                delete_last_line(vdh)
                print("line deleted")
        except AttributeError:
            pass

        try:
            camera_1.set_array(vdh.frame_camera1)
            camera_2.set_array(vdh.frame_camera2)
        except AttributeError:
            pass

        plt.legend(handles=[line0, line1, line2, line3],
                   labels=data_labels, fancybox=True, frameon=True, loc=(1))

        # The best solution for displaying "animated" tick labels.
        # A better solution would be to selectively only redraw these labels,
        # instead of the entire plot
        if i % 75 == 0:
            plt.draw()

    return line


vdh = visualization_log_replay.VisualizationDataHandler()
coords = Coordinates()

def convert_back_timestamp(timestamp):
    return timestamp*1000000 + vdh.first_timestamp

def onselect(xmin, xmax):
    """When a range is selected, prints selected range to file
    """
    file_ = open(vdh.control_panel.outlier_file, "a+")

    indmin, indmax = np.searchsorted(vdh.time_list, (xmin, xmax))
    indmax = min(len(vdh.time_list) - 1, indmax)

    coords.span_selected = vdh.time_list[indmin:indmax]

    #TODO: simplify file write - string join
    file_.write("[")
    try:
        for i, num in enumerate(coords.span_selected):
            if i != len(coords.span_selected) - 1:
                file_.write("%d, " % (convert_back_timestamp(num)))
            else:
                file_.write("%d]\n" % (convert_back_timestamp(num)))
    except TypeError:
        print("Data is not in list form")
        pass

    file_.close()


def onclick(event):
    anim.event_source.stop()


def delete_last_line(vdh):
    file_r = open(vdh.control_panel.outlier_file(), "r")
    lines = file_r.readlines()

    file_w = open(vdh.control_panel.outlier_file(), "w")
    file_w.writelines(lines[:-1])
    vdh.control_panel.remove_outlier = False


# open a tk dialogue box to select log file
Tk().withdraw()
log_file = askopenfilename()

# create figure, subplots
fig = plt.figure()
ax = plt.subplot2grid((2, 2), (0, 0), colspan=2)
ax1 = plt.subplot2grid((2, 2), (1, 0))
ax2 = plt.subplot2grid((2, 2), (1, 1))

# set axis titles
ax.set_ylabel("units")
ax.set_xlabel("Time (seconds)")

# format axis labels
xfmt = ScalarFormatter(useMathText=True)
ax.yaxis.set_major_locator(plt.MaxNLocator(10))
ax.yaxis.set_major_formatter(xfmt)
ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
ax.xaxis.set_major_locator(plt.MaxNLocator(10))

ax1.yaxis.set_major_locator(plt.NullLocator())
ax1.xaxis.set_major_locator(plt.NullLocator())
ax2.yaxis.set_major_locator(plt.NullLocator())
ax2.xaxis.set_major_locator(plt.NullLocator())

plt.tight_layout()

# This is the only part that limits how many types of data
# can be graphed at once
line0, = ax.plot([], color="b", linewidth=0.5, alpha=0.75)
line1, = ax.plot([], color="r", linewidth=0.5, alpha=0.75)
line2, = ax.plot([], color="g", linewidth=0.5, alpha=0.75)
line3, = ax.plot([], color="darkorange", linewidth=0.5, alpha=0.75)
marker, = ax.plot([], color="k", marker="+")
line = [line0, line1, line2, line3, marker]

blank = mpimg.imread("Blank.png")
camera_1 = ax1.imshow(blank, animated=True)
camera_2 = ax2.imshow(blank, animated=True)


span = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                    rectprops=dict(alpha=0.5, facecolor='red'))
 
"""removes blinking from span selection, but data is not displayed during selection"""
#cid = fig.canvas.mpl_connect('button_press_event', onclick)

# begin log replay on a separate thread in order to run concurrently
thread = threading.Thread(target=visualization_log_replay.main, args=(
    vdh, log_file))
thread.start()


# Setting blit to False renders animation useless - only grey window displayed
# Blitting may have some connection to removing blinking from animation
anim = animation.FuncAnimation(fig, animate, fargs=(
    line, ax, vdh, coords, camera_1, camera_2), interval=0.0, blit=True)

# display plotting window
try:
    plt.show()
except AttributeError:
    pass
plt.close()
