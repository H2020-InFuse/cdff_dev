import os
import shutil
from contextlib import contextmanager


@contextmanager
def ensure_cleanup(folder):
    """Create temporary folder for test data that will be deleted."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    try:
        yield
    finally:
        shutil.rmtree(folder)
