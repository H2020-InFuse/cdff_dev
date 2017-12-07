import os
import sys
import shutil
from contextlib import contextmanager
from code_generator import render
from functools import partial


@contextmanager
def ensure_cleanup(folder):
    """Create temporary folder for test data that will be deleted."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    try:
        yield
    finally:
        shutil.rmtree(folder)


@contextmanager
def hidden_stream(fileno):
    """Hide output stream."""
    if fileno not in [1, 2]:
        raise ValueError("Expected fileno 1 or 2.")
    stream_name = ["stdout", "stderr"][fileno - 1]
    getattr(sys, stream_name).flush()
    oldstream_fno = os.dup(fileno)
    devnull = os.open(os.devnull, os.O_WRONLY)
    newstream = os.dup(fileno)
    os.dup2(devnull, fileno)
    os.close(devnull)
    setattr(sys, stream_name, os.fdopen(newstream, 'w'))
    try:
        yield
    finally:
        os.dup2(oldstream_fno, fileno)


hidden_stdout = partial(hidden_stream, fileno=1)
hidden_stderr = partial(hidden_stream, fileno=2)


def build_extension(folder, **kwargs):
    filename = os.path.join(folder, "python", "setup.py")
    print(filename)
    with open(filename, "w") as f:
        setup_py = render("setup.py", **kwargs)
        f.write(setup_py)
    cmd = "python3 %s build_ext --inplace" % filename
    with hidden_stdout():
        os.system(cmd)