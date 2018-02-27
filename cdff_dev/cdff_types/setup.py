from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import get_config_vars
from Cython.Build import cythonize
import numpy
import os
from cdff_dev.path import check_cdffpath, CTYPESDIR


def strict_prototypes_workaround():
    # Workaround to remove '-Wstrict-prototypes' from compiler invocation
    opt = get_config_vars('OPT')[0]
    os.environ['OPT'] = " ".join(flag for flag in opt.split()
                                 if flag != '-Wstrict-prototypes')


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration("cdff_types", parent_package, top_path)

    cdffpath = "CDFF"  # TODO
    check_cdffpath(cdffpath)
    ctypespath = os.path.join(cdffpath, CTYPESDIR)

    pyx_filename = os.path.join("cdff_dev", "cdff_types", "cdff_types.pyx")
    cythonize(pyx_filename)

    config.add_extension(
        "",
        sources=["cdff_types.cpp"],
        include_dirs=[
            ".",
            numpy.get_include(),
            ctypespath
        ],
        library_dirs=[
            ctypespath
        ],
        libraries=[],
        define_macros=[("NDEBUG",)],
        extra_compile_args=[
            "-std=c++11",
            "-O3",
            # disable warnings caused by Cython using the deprecated
            # NumPy C-API
            "-Wno-cpp", "-Wno-unused-function"
        ]
    )

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    strict_prototypes_workaround()
    setup(**configuration(top_path='').todict())
