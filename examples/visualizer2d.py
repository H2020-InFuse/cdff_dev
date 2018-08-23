from cdff_dev import dataflowcontrol, visualization2d
import glob


# open a tk dialogue box to select log file
#from tkinter import Tk
#from tkinter.filedialog import askopenfilename
#Tk().withdraw()
#log_file = askopenfilename()


dfc = dataflowcontrol.DataFlowControl(nodes={}, periods={}, connections=[])

logfiles = [
    sorted(glob.glob("test/test_data/logs/xsens_imu_*.msg")),
]
stream_names = [
    "/xsens_imu.calibrated_sensors"
]

vis = visualization2d.MatplotlibVisualizerApplication()
vis.show_controls(dfc, logfiles, stream_names)
vis.exec_()