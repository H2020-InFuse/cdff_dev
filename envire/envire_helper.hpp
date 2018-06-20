#include <string>
#include <stdexcept>
#include <cstdint>
#include <envire_core/items/Item.hpp>
#include <envire_core/graph/EnvireGraph.hpp>
#include <envire_urdf/GraphLoader.hpp>
#include <urdf_parser/urdf_parser.h>
#include <envire_core/plugin/Plugin.hpp>
#include <envire_core/plugin/ClassLoader.hpp>
#include <envire_core/items/Item.hpp>
#include <envire_core/util/Demangle.hpp>
#include <boost/pointer_cast.hpp>

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
    //void* ptr;
    envire::core::ItemBase::Ptr item;
public:
    GenericItem()
    {
    }

    ~GenericItem()
    {
        // if(ptr)
        //     throw std::runtime_error("Item content has not been deleted.");
    }

    template <class _ItemData>
    void initialize(_ItemData* content)
    {
        
        std::string name = envire::core::demangleTypeName(typeid(*content));
        std::string itemname = "envire::core::Item<"+name+">";

        envire::core::ClassLoader* loader = envire::core::ClassLoader::getInstance();
            if(loader->hasClass(itemname))
            {
                if (loader->createEnvireItem(itemname, item))
                {
                    printf("created %s\n",itemname.c_str()); 
                    
                    // //typename envire::core::Item<_ItemData>::Ptr* typedPtr = (typename envire::core::Item<_ItemData>::Ptr *) ptr;
                    typename envire::core::Item<_ItemData>::Ptr typedPtr = boost::dynamic_pointer_cast< typename envire::core::Item<_ItemData> >(item);
                    // ptr = (void*) &typedPtr;
                    typedPtr->setData(*content);
                    typedPtr->contentsChanged();

                }
            }else{
                printf("no class with name %s found in plugins\n",name.c_str());
            }
        //     //ok
        //     envire::core::Item<_ItemData> *ritem = new envire::core::Item<_ItemData>(*content);
        //     //ok
        //     typename envire::core::Item<_ItemData>::Ptr* typedPtr = new typename envire::core::Item<_ItemData>::Ptr(ritem);
        //     //nope
        //     envire::core::ItemBase* base = dynamic_cast<envire::core::ItemBase*>(ritem);
        //     //nope
        //     item = boost::dynamic_pointer_cast< envire::core::ItemBase > (*typedPtr);
        //     ptr = (void*) typedPtr;

        // std::cout << __PRETTY_FUNCTION__ << "\n\t" << (*typedPtr)->getRawData() << " " << item->getRawData() << " " << base->getRawData() << " " << ritem->getRawData() << std::endl;
        // std::cout << __PRETTY_FUNCTION__ << "\n\t" << (*typedPtr)->getID() << " " << item->getID() << std::endl;

    }

    template <class _ItemData>
    typename envire::core::Item<_ItemData>::Ptr getItem(_ItemData* content)
    {
        //return *((typename envire::core::Item<_ItemData>::Ptr *) ptr);
        return boost::dynamic_pointer_cast< typename envire::core::Item<_ItemData> >(item);
    }

    template <class _ItemData>
    void setData(_ItemData* content)
    {
        //typename envire::core::Item<_ItemData>::Ptr* typedPtr = (typename envire::core::Item<_ItemData>::Ptr *) ptr;
        typename envire::core::Item<_ItemData>::Ptr typedPtr = boost::dynamic_pointer_cast< typename envire::core::Item<_ItemData> >(item);
        typedPtr->setData(*content);
        typedPtr->contentsChanged();
    }

    template <class _ItemData>
    void setTime(_ItemData* content, int64_t timestamp)
    {
        //assert(boost::dynamic_pointer_cast< typename envire::core::Item<_ItemData> >(item) != 0);
        //typename envire::core::Item<_ItemData>::Ptr* typedPtr = (typename envire::core::Item<_ItemData>::Ptr *) ptr;
        typename envire::core::Item<_ItemData>::Ptr typedPtr = boost::dynamic_pointer_cast< typename envire::core::Item<_ItemData> >(item);
        base::Time time;
        time.microseconds = timestamp;
        typedPtr->setTime(time);
    }

    template <class _ItemData>
    std::string getID()
    {
        return item->getID();
    }

    template <class _ItemData>
    void deleteItem(_ItemData* content)
    {
        // delete (typename envire::core::Item<_ItemData>::Ptr *) ptr;
        // ptr = 0;
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

