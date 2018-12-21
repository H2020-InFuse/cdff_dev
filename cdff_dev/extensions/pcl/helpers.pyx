# distutils: language=c++
from cython.operator cimport dereference as deref
from libcpp.string cimport string
cimport cdff_types
cimport _cdff_types


cdef extern from "pcl_helper.hpp":
    void loadPLYFile(string filename, _cdff_types.asn1SccPointcloud* outoput)


def load_ply_file(str filename):
    """Load PLY file.

    Parameters
    ----------
    filename : str
        Name of the PLY file

    Returns
    -------
    pointcloud : Pointcloud
        Pointcloud object
    """
    cdef string value = filename.encode()
    cdef cdff_types.Pointcloud pointcloud = cdff_types.Pointcloud()
    loadPLYFile(value, pointcloud.thisptr)
    return pointcloud
