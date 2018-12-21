"""
=========
Plot DGPS
=========

We can plot the DGPS quickly to check if the experiments have been recorded
correctly.

Run this script, for example, with

    python examples/plot_dgps.py dgps_*InFuse*.msg

Conversion commands for pocolog files from SherpaTT:

    pocolog2msgpack -l dgps_Logger.log -o dgps_Logger.msg
    rock2infuse dgps_Logger.msg dgps_Logger_InFuse.msg
"""
import sys
import matplotlib.pyplot as plt
from cdff_dev import logloader, data_export


def main():
    if len(sys.argv) < 2:
        raise ValueError("At least one log file required.")

    plt.figure()
    plt.title("Ground Truth")
    plt.gca().set_aspect("equal")
    for filename in sys.argv[1:]:
        ground_truth_df = load_path(filename, "/dgps.imu_pose")

        plt.plot(ground_truth_df["pos.0"], ground_truth_df["pos.1"],
                 label="Log File '%s'" % filename)
        plt.legend()
    plt.show()


def load_path(logfile, stream_name):
    log = logloader.load_log(logfile)
    df = data_export.object2dataframe(log, stream_name, ["pos"], ["pos"])
    return df


if __name__ == "__main__":
    main()
