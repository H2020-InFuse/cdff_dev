from . import logloader, typefromdict


def replay_and_process(dfc, log, stream_names):
    """Replay log file(s) and feed them to DataFlowControl.

    Parameters
    ----------
    dfc : DataFlowControl
        Configured processing components

    log : dict
        Log data

    stream_names : list of str
        Names of the streams that should be read from the logs
    """
    for timestamp, stream_name, typename, sample in logloader.replay(
            stream_names, log, verbose=0):
        obj = typefromdict.create_from_dict(typename, sample)
        dfc.process_sample(
            timestamp=timestamp, stream_name=stream_name, sample=obj)