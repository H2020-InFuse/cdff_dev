from cdff_dev import dataflowcontrol, visualization2d


dfc = dataflowcontrol.DataFlowControl(nodes={}, periods={}, connections=[])

logfiles = [["test/test_data/logs/frames.msg"]]
stream_names = ["/camera1.frame"]

vis = visualization2d.MatplotlibVisualizerApplication()
vis.show_controls(dfc, logfiles, stream_names,
                  image_stream_names=stream_names, image_shape=(1024, 768, 3))
vis.exec_()