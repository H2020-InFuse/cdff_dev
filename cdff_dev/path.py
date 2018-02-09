import os


def check_cdffpath(cdffpath="CDFF"):
    """Check if the provided path to CDFF is correct.

    If the path is not correct, an IOError is raised.

    Parameters
    ----------
    cdffpath : str, optional (default: 'CDFF')
        Path to CDFF
    """
    if not os.path.exists(cdffpath):
        raise IOError("Path '%s' does not exist." % cdffpath)

    path_to_types = os.path.join(cdffpath, "Common", "Types")
    if not os.path.exists(path_to_types):
        raise IOError("Path to types '%s' not found." % path_to_types)

    path_to_ctypes = os.path.join(path_to_types, "C")
    if not os.path.exists(path_to_ctypes):
        raise IOError("Path to C types (generated from ASN.1) not found.")
