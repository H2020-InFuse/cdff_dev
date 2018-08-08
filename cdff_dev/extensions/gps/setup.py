from subprocess import Popen, PIPE
import warnings


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration("gps", parent_package, top_path)

    extra_compile_args = [
        "-std=c++11",
        "-O3",
        # disable warnings caused by Cython using the deprecated
        # NumPy C-API
        "-Wno-cpp", "-Wno-unused-function"
    ]

    libraries = ["gdal", "proj"]
    try:
        include_dirs = get_include_dirs(libraries)
        library_dirs = get_library_dirs(libraries)
    except IOError as e:
        warnings.warn("Could not build extension gps, reason: %s" % e)
        return config

    config.add_extension(
        "conversion",
        sources=["conversion.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        define_macros=[("NDEBUG",)],
        extra_compile_args=extra_compile_args
    )

    return config


def get_include_dirs(libraries):
    p = Popen(["pkg-config", "--cflags-only-I"] + libraries,
              stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    if err:
        raise IOError(err.decode("utf-8"))
    # Remove -I
    return list(map(lambda dir: dir[2:].decode("utf-8"), output.split()))


def get_library_dirs(libraries):
    p = Popen(["pkg-config", "--libs-only-L"] + libraries,
              stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    if err:
        raise IOError(err.decode("utf-8"))
    # Remove -L
    return list(map(lambda dir: dir[2:].decode("utf-8"), output.split()))


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
