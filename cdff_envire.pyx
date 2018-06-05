# distutils: language = c++
from cython.operator cimport dereference as deref
from libc.stdint cimport int64_t
from libcpp.memory cimport shared_ptr as std_shared_ptr
cimport numpy as np
import numpy as np
cimport cdff_envire
cimport cdff_types
cimport _cdff_types
import os


np.import_array()  # must be here because we use the NumPy C API


cdef class Time:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_envire.Time()
        self.delete_thisptr = True

    def __str__(self):
        cdef bytes text = self.thisptr.toString(
            _cdff_envire.Resolution.Microseconds, "%Y%m%d-%H:%M:%S")
        return str("<time=%s>" % text.decode())

    def assign(self, Time other):
        self.thisptr.assign(deref(other.thisptr))

    def _get_microseconds(self):
        return self.thisptr.microseconds

    def _set_microseconds(self, int64_t microseconds):
        self.thisptr.microseconds = microseconds

    microseconds = property(_get_microseconds, _set_microseconds)

    def is_null(self):
        return self.thisptr.isNull()

    @staticmethod
    def now():
        cdef Time time = Time()
        time.thisptr[0] = _cdff_envire.now()
        return time

    def __richcmp__(Time self, Time other, int op):
        if op == 0:
            return deref(self.thisptr) < deref(other.thisptr)
        elif op == 1:
            return deref(self.thisptr) <= deref(other.thisptr)
        elif op == 2:
            return deref(self.thisptr) == deref(other.thisptr)
        elif op == 3:
            return deref(self.thisptr) != deref(other.thisptr)
        elif op == 4:
            return deref(self.thisptr) > deref(other.thisptr)
        elif op == 5:
            return deref(self.thisptr) >= deref(other.thisptr)
        else:
            raise ValueError("Unknown comparison operation %d" % op)

    def __add__(Time self, Time other):
        cdef Time time = Time()
        time.thisptr[0] = deref(self.thisptr) + deref(other.thisptr)
        return time

    def __iadd__(Time self, Time other):
        self.thisptr[0] = deref(self.thisptr) + deref(other.thisptr)
        return self

    def __sub__(Time self, Time other):
        cdef Time time = Time()
        time.thisptr[0] = deref(self.thisptr) - deref(other.thisptr)
        return time

    def __isub__(Time self, Time other):
        self.thisptr[0] = deref(self.thisptr) - deref(other.thisptr)
        return self

    def __floordiv__(Time self, int divider):
        cdef Time time = Time()
        time.thisptr[0] = deref(self.thisptr) / divider
        return time

    def __ifloordiv__(Time self, int divider):
        self.thisptr[0] = deref(self.thisptr) / divider
        return self

    def __mul__(Time self, double factor):
        cdef Time time = Time()
        time.thisptr[0] = deref(self.thisptr) * factor
        return time

    def __imul__(Time self, double factor):
        self.thisptr[0] = deref(self.thisptr) * factor
        return self


cdef class Vector3d:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self, double x=0.0, double y=0.0, double z=0.0):
        self.thisptr = new _cdff_envire.Vector3d(x, y, z)
        self.delete_thisptr = True

    def __str__(self):
        return str("[%.2f, %.2f, %.2f]" % (
                   self.thisptr.get(0), self.thisptr.get(1),
                   self.thisptr.get(2)))

    def assign(self, Vector3d other):
        self.thisptr.assign(deref(other.thisptr))

    def __array__(self, dtype=None):
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> 3
        return np.PyArray_SimpleNewFromData(
            1, shape, np.NPY_DOUBLE, <void*> self.thisptr.data())

    def __getitem__(self, int i):
        if i < 0 or i > 2:
            raise KeyError("index must be 0, 1 or 2 but was %d" % i)
        return self.thisptr.data()[i]

    def __setitem__(self, int i, double v):
        if i < 0 or i > 2:
            raise KeyError("index must be 0, 1 or 2 but was %d" % i)
        self.thisptr.data()[i] = v

    def _get_x(self):
        return self.thisptr.data()[0]

    def _set_x(self, double x):
        self.thisptr.data()[0] = x

    x = property(_get_x, _set_x)

    def _get_y(self):
        return self.thisptr.data()[1]

    def _set_y(self, double y):
        self.thisptr.data()[1] = y

    y = property(_get_y, _set_y)

    def _get_z(self):
        return self.thisptr.data()[2]

    def _set_z(self, double z):
        self.thisptr.data()[2] = z

    z = property(_get_z, _set_z)

    def norm(self):
        return self.thisptr.norm()

    def squared_norm(self):
        return self.thisptr.squaredNorm()

    def toarray(self):
        cdef np.ndarray[double, ndim=1] array = np.empty(3)
        cdef int i
        for i in range(3):
            array[i] = self.thisptr.data()[i]
        return array

    def fromarray(self, np.ndarray[double, ndim=1] array):
        cdef int i
        for i in range(3):
            self.thisptr.data()[i] = array[i]

    def __richcmp__(Vector3d self, Vector3d other, int op):
        if op == 0:
            raise NotImplementedError("<")
        elif op == 1:
            raise NotImplementedError("<=")
        elif op == 2:
            return deref(self.thisptr) == deref(other.thisptr)
        elif op == 3:
            return deref(self.thisptr) != deref(other.thisptr)
        elif op == 4:
            raise NotImplementedError(">")
        elif op == 5:
            raise NotImplementedError(">=")
        else:
            raise ValueError("Unknown comparison operation %d" % op)


cdef class Quaterniond:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self, double w=1.0, double x=0.0, double y=0.0, double z=0.0):
        self.thisptr = new _cdff_envire.Quaterniond(w, x, y, z)
        self.delete_thisptr = True

    def __str__(self):
        return str("[im=%.2f, real=(%.2f, %.2f, %.2f)]" % (
            self.thisptr.w(), self.thisptr.x(), self.thisptr.y(),
            self.thisptr.z()))

    def assign(self, Quaterniond other):
        self.thisptr.assign(deref(other.thisptr))

    def toarray(self):
        cdef np.ndarray[double, ndim=1] array = np.array([
            self.thisptr.w(), self.thisptr.x(), self.thisptr.y(),
            self.thisptr.z()])
        return array

    def fromarray(self, np.ndarray[double, ndim=1] array):
        self.thisptr[0] = _cdff_envire.Quaterniond(
            array[0], array[1], array[2], array[3])


cdef class TransformWithCovariance:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_envire.TransformWithCovariance()
        self.delete_thisptr = True

    def __str__(self):
        return str("(translation=%s, orientation=%s)" % (self.translation,
                                                         self.orientation))

    def assign(self, TransformWithCovariance other):
        self.thisptr.assign(deref(other.thisptr))

    def _get_translation(self):
        cdef Vector3d translation = Vector3d()
        del translation.thisptr
        translation.delete_thisptr = False
        translation.thisptr = &self.thisptr.translation
        return translation

    def _set_translation(self, Vector3d translation):
        self.thisptr.translation = deref(translation.thisptr)

    translation = property(_get_translation, _set_translation)

    def _get_orientation(self):
        cdef Quaterniond orientation = Quaterniond()
        del orientation.thisptr
        orientation.delete_thisptr = False
        orientation.thisptr = &self.thisptr.orientation
        return orientation

    def _set_orientation(self, Quaterniond orientation):
        self.thisptr.orientation = deref(orientation.thisptr)

    orientation = property(_get_orientation, _set_orientation)


cdef class Transform:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.delete_thisptr and self.thisptr != NULL:
            del self.thisptr

    def __init__(
            self, Time time=None,
            TransformWithCovariance transform_with_covariance=None,
            Vector3d translation=None, Quaterniond orientation=None):
        if time is not None:
            if transform_with_covariance is not None:
                self.thisptr = new _cdff_envire.Transform(
                    deref(time.thisptr), deref(transform_with_covariance.thisptr))
            else:
                self.thisptr = new _cdff_envire.Transform(deref(time.thisptr))
        elif transform_with_covariance is not None:
            self.thisptr = new _cdff_envire.Transform(
                    deref(transform_with_covariance.thisptr))
        elif translation is not None and orientation is not None:
            self.thisptr = new _cdff_envire.Transform(
                    deref(translation.thisptr), deref(orientation.thisptr))
        elif translation is not None:
            raise ValueError("Orientation is missing")
        elif orientation is not None:
            raise ValueError("Translation is missing")
        else:
            self.thisptr = new _cdff_envire.Transform()
        self.delete_thisptr = True

    def _get_transform(self):
        cdef TransformWithCovariance transform = TransformWithCovariance()
        del transform.thisptr
        transform.thisptr = &self.thisptr.transform
        transform.delete_thisptr = False
        return transform

    def _set_transform(self, TransformWithCovariance transform):
        self.thisptr.transform = deref(transform.thisptr)

    transform = property(_get_transform, _set_transform)

    def __str__(self):
        return self.thisptr.toString().decode()


ctypedef fused GenericType:
    cdff_types.LaserScan
    cdff_types.Pointcloud
    cdff_types.RigidBodyState
    cdff_types.JointState
    cdff_types.Joints


cdef class GenericItem:
    def __cinit__(self):
        self.thisptr = new _cdff_envire.GenericItem()

    def __dealloc__(self):
        if self.filled:
            raise RuntimeError("Item content must be deleted explicitely.")
        del self.thisptr

    def __init__(self):
        self.filled = False

    def initialize(self, GenericType content):
        if self.filled:
            raise RuntimeError("Item already contains data.")
        self.thisptr.initialize(content.thisptr)
        self.filled = True

    def set_data(self, GenericType content):
        if not self.filled:
            raise RuntimeError("Item does not have any content.")
        self.thisptr.setData(content.thisptr)

    def get_data(self, GenericType content):
        if not self.filled:
            raise RuntimeError("Item does not have any content.")
        content.thisptr[0] = self.thisptr.getItem(content.thisptr).get().getData()

    def delete_item(self, GenericType content):
        self.thisptr.deleteItem(content.thisptr)
        self.filled = False


cdef class EnvireGraph:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.delete_thisptr and self.thisptr != NULL:
            del self.thisptr

    def __init__(self, EnvireGraph other=None):
        if other is None:
            self.thisptr = new _cdff_envire.EnvireGraph()
        else:
            self.thisptr = new _cdff_envire.EnvireGraph(deref(other.thisptr))
        self.delete_thisptr = True

    def add_frame(self, str name):
        self.thisptr.addFrame(name.encode())

    def remove_frame(self, str name):
        self.thisptr.removeFrame(name.encode())

    def contains_frame(self, str name):
        return self.thisptr.containsFrame(name.encode())

    def contains_edge(self, str origin, str target):
        return self.thisptr.containsEdge(origin.encode(), target.encode())

    def add_transform(self, str origin, str target, Transform tf):
        self.thisptr.addTransform(origin.encode(), target.encode(),
                                  deref(tf.thisptr))

    def update_transform(self, str origin, str target, Transform tf):
        self.thisptr.updateTransform(origin.encode(), target.encode(),
                                     deref(tf.thisptr))

    def get_transform(self, str origin, str target):
        """Return a copy of the transform computed from the graph."""
        cdef Transform transform = Transform()
        transform.thisptr[0] = self.thisptr.getTransform(
            origin.encode(), target.encode())
        return transform

    def remove_transform(self, str origin, str target):
        self.thisptr.removeTransform(origin.encode(), target.encode())

    def num_vertices(self):
        return self.thisptr.num_vertices()

    def num_edges(self):
        return self.thisptr.num_edges()

    def save_to_file(self, str filename):
        self.thisptr.saveToFile(filename.encode())

    def load_from_file(self, str filename):
        self.thisptr.loadFromFile(filename.encode())

    def clear_frame(self, str name):
        self.thisptr.clearFrame(name.encode())

    def get_total_item_count(self, str name):
        return self.thisptr.getTotalItemCount(name.encode())

    def add_item_to_frame(self, str frame, GenericItem item, GenericType content):
        item.initialize(content)
        self.thisptr.addItemToFrame(
            frame.encode(), item.thisptr.getItem(content.thisptr))

    def remove_item_from_frame(self, GenericItem item, GenericType content):
        self.thisptr.removeItemFromFrame(item.thisptr.getItem(content.thisptr))
        item.delete_item(content)

    def get_item_count(self, str frame, GenericType item):
        return _cdff_envire.getItemCount(
            deref(self.thisptr), frame.encode(), item.thisptr)


cpdef load_urdf(EnvireGraph graph, str filename, bool load_frames=False, bool load_joints=False):
    if not os.path.exists(filename):
        raise IOError("File '%s' does not exist." % filename)

    _cdff_envire.loadURDF(deref(graph.thisptr), filename.encode(), load_frames, load_joints)


cdef class EnvireVisualizer:
    cdef _cdff_envire.EnvireVisualizerInterface* thisptr

    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        if self.thisptr != NULL:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_envire.EnvireVisualizerInterface()

    def display_graph(self, EnvireGraph graph, str base):
        self.thisptr.displayGraph(deref(graph.thisptr), base.encode())

    def redraw(self):
        self.thisptr.redraw()

    def show(self):
        self.thisptr.show()

    def start_redraw_thread(self):
        self.thisptr.startRedrawThread()

    def stop_redraw_thread(self):
        self.thisptr.stopRedrawThread()

    def lock_redraw_thread(self):
        self.thisptr.lockRedrawThread()

    def unlock_redraw_thread(self):
        self.thisptr.unlockRedrawThread()
