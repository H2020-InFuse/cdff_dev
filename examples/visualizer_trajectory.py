from cdff_dev import envirevisualization, dataflowcontrol


def data_generator():
    for i in range(100):
        sample = [i, i, i]
        typename = "Vector3d"
        timestamp = i
        stream_name = "data.point"
        yield timestamp, stream_name, typename, sample


def main():
    sample_iter = data_generator()

    frames = {"data.point": "center"}

    connections = (("data.point", "result.trajectory"),)
    dfc = dataflowcontrol.DataFlowControl({}, connections, {})
    dfc.setup()

    app = envirevisualization.EnvireVisualizerApplication(
        frames, [], center_frame="center")
    app.show_controls(sample_iter, dfc)
    app.exec_()


if __name__ == "__main__":
    main()
