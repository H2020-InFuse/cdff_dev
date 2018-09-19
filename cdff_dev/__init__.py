__version__ = "0.0.0"
__description__ = "CDFF-Dev provides tools to develop, test, " \
    "visualize, and perform analysis on data fusion products."


import os
loglevel = os.environ.get("GLOG_minloglevel", default="3")
os.environ["GLOG_minloglevel"] = loglevel