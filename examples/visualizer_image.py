from cdff_dev import dataflowcontrol, visualization2d


dfc = dataflowcontrol.DataFlowControl(nodes={}, periods={}, connections=[])

logfiles = [["test/test_data/logs/frames.msg"]]
stream_names = ["/camera1.frame"]

vis = visualization2d.MatplotlibVisualizerApplication()
vis.show_controls(dfc, logfiles, stream_names,
                  image_stream_name=stream_names[0])
vis.exec_()