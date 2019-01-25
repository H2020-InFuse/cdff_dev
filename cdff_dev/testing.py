import os
import sys
import shutil
import warnings
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
                exit_status = os.system(cmd)
        else:
            exit_status = os.system(cmd)
    if exit_status:
        raise Exception("Exit status '%s'" % exit_status)


# from scikit-klearn: https://scikit-learn.org
def assert_warns_message(warning_class, message, func, *args, **kw):
    # very important to avoid uncontrolled state propagation
    """Test that a certain warning occurs and with a certain message.
    Parameters
    ----------
    warning_class : the warning class
        The class to test for, e.g. UserWarning.
    message : str | callable
        The entire message or a substring to  test for. If callable,
        it takes a string as argument and will trigger an assertion error
        if it returns `False`.
    func : callable
        Calable object to trigger warnings.
    *args : the positional arguments to `func`.
    **kw : the keyword arguments to `func`.
    Returns
    -------
    result : the return value of `func`
    """
    clean_warning_registry()
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        warnings.simplefilter('ignore', PendingDeprecationWarning)
        warnings.simplefilter('ignore', ResourceWarning)
        # Trigger a warning.
        result = func(*args, **kw)
        # Verify some things
        if not len(w) > 0:
            raise AssertionError("No warning raised when calling %s"
                                 % func.__name__)

        found = [issubclass(warning.category, warning_class) for warning in w]
        if not any(found):
            raise AssertionError("No warning raised for %s with class "
                                 "%s"
                                 % (func.__name__, warning_class))

        message_found = False
        # Checks the message of all warnings belong to warning_class
        for index in [i for i, x in enumerate(found) if x]:
            # substring will match, the entire message with typo won't
            msg = w[index].message  # For Python 3 compatibility
            msg = str(msg.args[0] if hasattr(msg, 'args') else msg)
            if callable(message):  # add support for certain tests
                check_in_message = message
            else:
                check_in_message = lambda msg: message in msg

            if check_in_message(msg):
                message_found = True
                break

        if not message_found:
            raise AssertionError("Did not receive the message you expected "
                                 "('%s') for <%s>, got: '%s'"
                                 % (message, func.__name__, msg))

    return result


def clean_warning_registry():
    """Safe way to reset warnings """
    warnings.resetwarnings()
    reg = "__warningregistry__"
    for mod in list(sys.modules.values()):
        if hasattr(mod, reg):
            getattr(mod, reg).clear()
