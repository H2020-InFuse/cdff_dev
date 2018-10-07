from cdff_dev import dataflowcontrol, logloader, visualization_qtgraph


def main():
    verbose = 2
    filename = "test/test_data/logs/frames2.msg"
    stream_names = ["/hcru0/pt_stereo_rect/left/image"]
    stream_aliases = {
        "/hcru0/pt_stereo_rect/left/image": "/hcru0/pt_stereo_rect/left.image"}
    log_iterator = logloader.replay_logfile(filename, stream_names)

    dfc = dataflowcontrol.DataFlowControl(
        nodes={}, connections=(), stream_aliases=stream_aliases,
        verbose=verbose)
    dfc.setup()

    app = visualization_qtgraph.ImageVisualizerApplication(
        "/hcru0/pt_stereo_rect/left.image")
    app.show_controls(log_iterator, dfc)
    app.exec_()

    dfc.node_statistics_.print_statistics()


if __name__ == "__main__":
    main()
