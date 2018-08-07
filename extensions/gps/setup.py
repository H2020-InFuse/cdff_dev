from subprocess import Popen, PIPE
import warnings
import os
from cdff_dev.path import load_cdffpath, CTYPESDIR


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration("gps", parent_package, top_path)

    cdffpath = load_cdffpath()
    ctypespath = os.path.join(cdffpath, CTYPESDIR)

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
        "cdff_gps",
        sources=["cdff_gps.pyx", "UTMConverter.cpp"],
        include_dirs=[".", ctypespath] + include_dirs,
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
        raise IOError(err)
    return list(map(lambda dir: dir[2:], output.split()))  # Remove -I


def get_library_dirs(libraries):
    p = Popen(["pkg-config", "--libs-only-L"] + libraries,
              stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    if err:
        raise IOError(err)
    return list(map(lambda dir: dir[2:], output.split()))  # Remove -L


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

"""
    extensions = [
        Extension(
            "cdff_gps",
            [
                "./cdff_gps.pyx",
                "./UTMConverter.cpp",
            ],
            include_dirs=[
                ".",
                "/home/dfki.uni-bremen.de/bwehbe/CDFF_dev/cdff/CDFF/Common/Types/C",
                "/home/dfki.uni-bremen.de/bwehbe/CDFF_dev/install/include/",
                "/usr/include/gdal/",
            ],
            define_macros=[
                ("NDEBUG",),
             ],
            extra_compile_args=[
                "-std=c++11",
                # disable warnings caused by Cython using the deprecated
                # NumPy C-API
                "-Wno-cpp", "-Wno-unused-function"
            ],
            library_dirs=[
                "/home/dfki.uni-bremen.de/bwehbe/CDFF_dev/install/lib/",
                "/user/include/gdal/",
            ],
            libraries=[
                "gdal",
                "proj",
            ],
            language="c++"
        )
    ]
    setup(ext_modules=cythonize(extensions))
"""