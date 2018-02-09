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


if __name__ == '__main__':
    strict_prototypes_workaround()

    cdffpath = "../../../CDFF"  # TODO
    check_cdffpath(cdffpath)
    ctypespath = os.path.join(cdffpath, CTYPESDIR)

    extensions = [
        Extension(
            "cdff_types",
            [
                "cdff_types.pyx"
            ],
            include_dirs=[
                ".",
                numpy.get_include(),
                ctypespath
            ],
            library_dirs=[
                ctypespath
            ],
            libraries=[],
            define_macros=[
                ("NDEBUG",),
             ],
            extra_compile_args=[
                "-std=c++11",
                "-O3",
                # disable warnings caused by Cython using the deprecated
                # NumPy C-API
                "-Wno-cpp", "-Wno-unused-function"
            ],
            language="c++"
        )
    ]
    setup(
        name="cdff_types",
        ext_modules=cythonize(extensions),
        description="Python bindings for CDFF types",
        version="0.1",
        maintainer="Alexander Fabisch",
        maintainer_email="Alexander.Fabisch@dfki.de",
        packages=[""],
        package_dir={"": "."},
        package_data={
            "": ["_cdff_types.pxd", "cdff_types.pxd", "cdff_types.pyx"]
        }
    )
