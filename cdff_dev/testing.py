import os
import sys
import shutil
from contextlib import contextmanager
from functools import partial
from .code_generator import render


class EnsureCleanup():
    def __init__(self, folder):
        self.folder = folder
        self.created_folder = False
        self.filenames = []
        self.folders = []

    def __enter__(self):
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
            self.created_folder = True
        return self

    def add_files(self, filenames):
        self.filenames.extend(filenames)

    def add_folder(self, folder):
        self.folders.append(folder)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for filename in self.filenames:
            os.remove(filename)
        for folder in self.folders:
            shutil.rmtree(folder)
        if self.created_folder:
            shutil.rmtree(self.folder)


@contextmanager
def modified_pythonpath(additional_folder):
    old_path = sys.path
    sys.path = old_path + [additional_folder]
    try:
        yield
    finally:
        sys.path = old_path


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
    if "hide_stderr" in kwargs:
        hide_stderr = kwargs.pop("hide_stderr")
    else:
        hide_stderr = False
    filename = os.path.join(folder, "python", "setup.py")
    with open(filename, "w") as f:
        setup_py = render("setup.py", **kwargs)
        f.write(setup_py)
    cmd = "python3 %s build_ext --inplace" % filename
    with hidden_stdout():
        if hide_stderr:
            with hidden_stderr():
                os.system(cmd)
        else:
            os.system(cmd)
