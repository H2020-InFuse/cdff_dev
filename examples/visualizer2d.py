from cdff_dev import dataflowcontrol, visualization2d
import glob


# open a tk dialogue box to select log file
#from tkinter import Tk
#from tkinter.filedialog import askopenfilename
#Tk().withdraw()
#log_file = askopenfilename()


dfc = dataflowcontrol.DataFlowControl(nodes={}, periods={}, connections=[])

logfiles = [
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
    "/xsens_imu.calibrated_sensors"
]

vis = visualization2d.MatplotlibVisualizerApplication()
vis.show_controls(dfc, logfiles, stream_names)
vis.exec_()