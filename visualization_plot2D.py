import visualization_log_replay


# open a tk dialogue box to select log file
#from tkinter import Tk
#from tkinter.filedialog import askopenfilename
#Tk().withdraw()
#log_file = askopenfilename()


from cdff_dev import dataflowcontrol
import glob

dfc = dataflowcontrol.DataFlowControl(nodes={}, periods={}, connections=[])

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

vis = visualization_log_replay.MatplotlibVisualizer(
    dfc, log_files, stream_names)
vis.exec_()