from . import logloader, typefromdict


def replay_and_process(dfc, iterator):
    """Replay log file(s) and feed them to DataFlowControl.

    Parameters
    ----------
    dfc : DataFlowControl
        Configured processing components

    iterator : Iterable
        Iterable object that yields log samples in the correct temporal
        order. The iterable returns in each step a quadrupel of
        (timestamp, stream_name, typename, sample).
    """
    for timestamp, stream_name, typename, sample in iterator:
        obj = typefromdict.create_from_dict(typename, sample)
        dfc.process_sample(
            timestamp=timestamp, stream_name=stream_name, sample=obj)