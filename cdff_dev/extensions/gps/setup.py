from subprocess import Popen, PIPE
import warnings
from setuptools import Extension


def get_extensions():
    extra_compile_args = [
        "-std=c++14",
        "-O3",
        "-Wno-cpp", "-Wno-unused-function"
    ]

    libraries = ["gdal", "proj"]
    try:
        include_dirs = _get_include_dirs(libraries)
        library_dirs = _get_library_dirs(libraries)
    except IOError as e:
        warnings.warn("Could not build extension gps, reason: %s" % e)
        return []

    return [Extension(
        "cdff_dev.extensions.gps.conversion",
        sources=["cdff_dev/extensions/gps/conversion.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        define_macros=[("NDEBUG",)],
        extra_compile_args=extra_compile_args
    )]


def _get_include_dirs(libraries):
    p = Popen(["pkg-config", "--cflags-only-I"] + libraries,
              stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    if err:
        raise IOError(err.decode("utf-8"))
    return [d[2:] for d in output.decode("utf-8").split() if d.startswith("-I")]


def _get_library_dirs(libraries):
    p = Popen(["pkg-config", "--libs-only-L"] + libraries,
              stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    if err:
        raise IOError(err.decode("utf-8"))
    return [d[2:] for d in output.decode("utf-8").split() if d.startswith("-L")]