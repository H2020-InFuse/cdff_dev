from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import get_config_vars
from Cython.Build import cythonize
import os


def strict_prototypes_workaround():
    # Workaround to remove '-Wstrict-prototypes' from compiler invocation
    opt = get_config_vars('OPT')[0]
    os.environ['OPT'] = " ".join(flag for flag in opt.split()
                                 if flag != '-Wstrict-prototypes')


if __name__ == '__main__':
    strict_prototypes_workaround()

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
