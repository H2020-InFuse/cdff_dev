import os


TYPESDIR = os.path.join("Common", "Types")
CTYPESDIR = os.path.join(TYPESDIR, "C")


def load_cdffpath():
    if "CDFFPATH" in os.environ:
        cdffpath = os.environ.get("CDFFPATH")
    elif os.path.exists("cdffpath"):
        with open("cdffpath", "r") as f:
            cdffpath = f.read()
    else:
        cdffpath = input("Please enter the path to CDFF:")
        with open("cdffpath", "w") as f:
            f.write(cdffpath)

    cdffpath = cdffpath.strip()

    check_cdffpath(cdffpath)

    return os.path.abspath(cdffpath)


def check_cdffpath(cdffpath):
    """Check if the provided path to CDFF is correct.

    If the path is empty, does not exist, or does not contain the required
    data type subdirectories, an IOError is raised.

    Parameters
    ----------
    cdffpath : str
        Path to CDFF
    """
    if not os.path.exists(cdffpath):
        raise IOError("Path '%s' does not exist." % cdffpath)

    path_to_types = os.path.join(cdffpath, TYPESDIR)
    if not os.path.exists(path_to_types):
        raise IOError("Path to types '%s' not found." % path_to_types)

    path_to_ctypes = os.path.join(cdffpath, CTYPESDIR)
    if not os.path.exists(path_to_ctypes):
        raise IOError("Path to C types (generated from ASN.1) not found.")
