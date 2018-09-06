from libcpp.string cimport string
from libcpp cimport bool
from libc.stdint cimport int64_t
from libcpp.memory cimport shared_ptr as std_shared_ptr


# Type definitions

cdef extern from "boost/shared_ptr.hpp" namespace "boost":
    cdef cppclass shared_ptr[T]:
        shared_ptr()
        shared_ptr(T*)
        T* get()


cdef extern from "base/Time.hpp" namespace "base":
    cdef cppclass Time:
        Time()

        int64_t microseconds

        Time& assign "operator="(Time&)

        bool operator<(Time)
        bool operator>(Time)
        bool operator==(Time)
        bool operator!=(Time)
        bool operator>=(Time)
        bool operator<=(Time)
        Time operator-(Time)
        Time operator+(Time)
        Time operator/(int)
        Time operator*(double)

        bool isNull()
        string toString(Resolution, string)


cdef extern from "base/Time.hpp" namespace "base::Time":
    cdef enum Resolution:
        Seconds
        Milliseconds
        Microseconds


cdef extern from "base/Time.hpp" namespace "base::Time":
    Time now()


cdef extern from "base/Eigen.hpp" namespace "base":
    cdef cppclass Vector3d:
        Vector3d()
        Vector3d(double, double, double)
        Vector3d(Vector3d&)
        double* data()
        int rows()
        Vector3d& assign "operator="(Vector3d&)
        bool operator==(Vector3d&)
        bool operator!=(Vector3d&)
        double& get "operator()"(int row)
        double x()
        double y()
        double z()
        double norm()
        double squaredNorm()

    cdef cppclass Quaterniond:
        Quaterniond()
        Quaterniond(double, double, double, double)
        Quaterniond& assign "operator="(Quaterniond&)
        double w()
        double x()
        double y()
        double z()


cdef extern from "base/TransformWithCovariance.hpp" namespace "base":
    cdef cppclass TransformWithCovariance:
        TransformWithCovariance()
        TransformWithCovariance& assign "operator="(TransformWithCovariance&)
        Vector3d translation
        Quaterniond orientation


# EnviRe core

cdef extern from "envire_core/items/Transform.hpp" namespace "envire::core":
    cdef cppclass Transform:
        Transform()
        Transform(Time)
        Transform(Time, TransformWithCovariance)
        Transform(TransformWithCovariance)
        Transform(Vector3d, Quaterniond)
        string toString()

        Time time
        TransformWithCovariance transform


cdef extern from "envire_core/items/ItemBase.hpp" namespace "envire::core":
    cdef cppclass ItemBase:
        #void setID(uuid)
        #uuid getID()
        string getIDString()


cdef extern from "envire_core/items/Item.hpp" namespace "envire::core":
    cdef cppclass Item[_ItemData](ItemBase):
        Item()
        Item(_ItemData)
        void setTime(Time)
        Time getTime()
        void setFrame(string)
        string getFrame()
        void setData(_ItemData)
        _ItemData getData()


cdef extern from "envire_core/graph/EnvireGraph.hpp" namespace "envire::core":
    cdef cppclass EnvireGraph:
        cppclass ItemIterator[T]:
            T operator*()
            ItemIterator& next "operator++"()
            ItemIterator& last "operator--"()
        EnvireGraph()
        EnvireGraph(const EnvireGraph&)
        void addFrame(string) except +
        void removeFrame(string) except +
        bool containsFrame(string)
        #void disconnectFrame(string)
        bool containsEdge(string, string)
        void addTransform(string, string, Transform) except +
        void updateTransform(string, string, Transform)
        Transform getTransform(string, string) except +
        void removeTransform(string, string)

        unsigned num_vertices()
        unsigned num_edges()

        void saveToFile(string)
        void loadFromFile(string)

        void clearFrame(string)
        bool containsItems[T](string)
        unsigned getItemCount[T](string)
        unsigned getTotalItemCount(string)
        #unsigned getItemCount(string)

        void addItemToFrame[T](string, T) except +
        void removeItemFromFrame[T](T) except +
        ItemIterator[T] getItem[T](string, int) except +


cdef extern from "envire_helper.hpp":
    cdef cppclass GenericItem:
        GenericItem()
        void initialize[_ItemData](_ItemData* content) except +
        void setData[_ItemData](_ItemData* content)
        void setTime[_ItemData](_ItemData* content, int64_t timestamp)
        shared_ptr[Item[_ItemData]] getItem[_ItemData](_ItemData* content)
    void addItemToFrame[_ItemData](
        EnvireGraph& graph, const string& frame, _ItemData* contentPtr) except +
    unsigned getItemCount[_ItemData](
        EnvireGraph& graph, const string& frame, _ItemData* contentPtr) except +
    void loadURDF(EnvireGraph& graph, const string& filename, bool load_frames,
                  bool load_joints, bool load_visuals) except +


# EnviRe visualizer

cdef extern from "envire_visualizer/EnvireVisualizerInterface.hpp":
    cdef cppclass EnvireVisualizerInterface:
        EnvireVisualizerInterface()
        void displayGraph(EnvireGraph graph, string base)
        void redraw()
        void show()
