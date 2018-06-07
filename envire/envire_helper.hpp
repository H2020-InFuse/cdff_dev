#include <string>
#include <stdexcept>
#include <cstdint>
#include <envire_core/items/Item.hpp>
#include <envire_core/graph/EnvireGraph.hpp>
#include <envire_urdf/GraphLoader.hpp>
#include <urdf_parser/urdf_parser.h>
#include <envire_core/plugin/Plugin.hpp>
#include <envire_core/items/Item.hpp>


template <class _ItemData>
unsigned getItemCount(
    envire::core::EnvireGraph& graph, const envire::core::FrameId& frame,
    _ItemData* contentPtr)
{
    return graph.getItemCount<typename envire::core::Item<_ItemData> >(frame);
}

#include <iostream>
class GenericItem
{
    void* ptr;
public:
    GenericItem() : ptr(0)
    {
    }

    ~GenericItem()
    {
        if(ptr)
            throw std::runtime_error("Item content has not been deleted.");
    }

    template <class _ItemData>
    void initialize(_ItemData* content)
    {
        typename envire::core::Item<_ItemData>::Ptr* typedPtr =
            new typename envire::core::Item<_ItemData>::Ptr(
                new envire::core::Item<_ItemData>(*content));
        ptr = (void*) typedPtr;
    }

    template <class _ItemData>
    typename envire::core::Item<_ItemData>::Ptr getItem(_ItemData* content)
    {
        return *((typename envire::core::Item<_ItemData>::Ptr *) ptr);
    }

    template <class _ItemData>
    void setData(_ItemData* content)
    {
        typename envire::core::Item<_ItemData>::Ptr* typedPtr = (typename envire::core::Item<_ItemData>::Ptr *) ptr;
        (*typedPtr)->setData(*content);
        (*typedPtr)->contentsChanged();
    }

    template <class _ItemData>
    void setTime(_ItemData* content, int64_t timestamp)
    {
        typename envire::core::Item<_ItemData>::Ptr* typedPtr = (typename envire::core::Item<_ItemData>::Ptr *) ptr;
        base::Time time;
        time.microseconds = timestamp;
        (*typedPtr)->setTime(time);
    }

    template <class _ItemData>
    void deleteItem(_ItemData* content)
    {
        delete (typename envire::core::Item<_ItemData>::Ptr *) ptr;
        ptr = 0;
    }
};

template <typename T>
struct NoDeleter
{
    void operator()(T* p) const
    {
    }
};

void loadURDF(envire::core::EnvireGraph& graph, const std::string& filename,
              bool load_frames=false, bool load_joints=false)
{
    std::shared_ptr<envire::core::EnvireGraph> ptr(&graph, NoDeleter<envire::core::EnvireGraph>());
    std::shared_ptr<urdf::ModelInterface> model = urdf::parseURDFFile(filename);

    // TODO
    // everything is attached to iniPose, which is initialized with the current
    // timestamp, we should at least allow the user to do something else
    envire::urdf::GraphLoader loader(ptr);
    loader.loadStructure(*model);
    if(load_frames)
        loader.loadFrames(*model);
    if(load_joints)
        loader.loadJoints(*model);
}

#define ENVIRE_REGISTER_MULTI_METADATA( _classname, _datatype, _variable ) \
static envire::core::MetadataInitializer _metadataInit##_variable(typeid(_classname), #_datatype, #_classname);
#define ENVIRE_REGISTER_MULTI_ITEM_WITHOUT_SERIALIZATION( _datatype, _variable ) \
CLASS_LOADER_REGISTER_CLASS(envire::core::Item<_datatype>, envire::core::ItemBase) \
ENVIRE_REGISTER_MULTI_METADATA(envire::core::Item<_datatype>, _datatype, _variable)


#include <base/samples/RigidBodyState.hpp>
ENVIRE_REGISTER_MULTI_ITEM_WITHOUT_SERIALIZATION(base::samples::RigidBodyState, 1)
#include <base/samples/Pointcloud.hpp>
ENVIRE_REGISTER_MULTI_ITEM_WITHOUT_SERIALIZATION(base::samples::Pointcloud, 2)
#include <base/samples/LaserScan.hpp>
ENVIRE_REGISTER_MULTI_ITEM_WITHOUT_SERIALIZATION(base::samples::LaserScan, 3)
#include <base/samples/Joints.hpp>
ENVIRE_REGISTER_MULTI_ITEM_WITHOUT_SERIALIZATION(base::samples::Joints, 4)
#include <urdf_model/joint.h>
ENVIRE_REGISTER_MULTI_ITEM_WITHOUT_SERIALIZATION(urdf::Joint, 5)
#include <urdf_model/link.h>
ENVIRE_REGISTER_MULTI_ITEM_WITHOUT_SERIALIZATION(urdf::Link, 6)
