import asyncio
from . import typefromdict


def replay_and_process(dfc, log_iterator):
    """Replay log file(s) and feed them to DataFlowControl.

    Parameters
    ----------
    dfc : DataFlowControl
        Configured processing components

    log_iterator : Iterable
        Iterable object that yields log samples in the correct temporal
        order. The iterable returns in each step a quadrupel of
        (timestamp, stream_name, typename, sample).
    """
    for timestamp, stream_name, typename, sample in log_iterator:
        obj = typefromdict.create_from_dict(typename, sample)
        dfc.process_sample(
            timestamp=timestamp, stream_name=stream_name, sample=obj)


async def _sample_reader(log_iterator, queue):
    for timestamp, stream_name, typename, sample in log_iterator:
        obj = typefromdict.create_from_dict(typename, sample)
        await queue.put((timestamp, stream_name, obj))

    await queue.put(None)


async def _process_sample(dfc, queue):
    while True:
        item = await queue.get()
        if item is None:
            break

        timestamp, stream_name, obj = item
        dfc.process_sample(
            timestamp=timestamp, stream_name=stream_name, sample=obj)


def replay_and_process_async(dfc, log_iterator, queue_size=10):
    """Replay log file(s) and feed them to DataFlowControl.

    This version does conversion to objects and data flow control processing
    in parallel.

    Parameters
    ----------
    dfc : DataFlowControl
        Configured processing components

    log_iterator : Iterable
        Iterable object that yields log samples in the correct temporal
        order. The iterable returns in each step a quadrupel of
        (timestamp, stream_name, typename, sample).

    queue_size : int, optional (default: 10)
        Maximum size of the queue between coroutines
    """
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue(loop=loop, maxsize=queue_size)
    producer_coro = _sample_reader(log_iterator, queue)
    consumer_coro = _process_sample(dfc, queue)
    loop.run_until_complete(asyncio.gather(producer_coro, consumer_coro))
    loop.close()
