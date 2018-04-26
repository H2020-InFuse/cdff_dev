#include <string>
#include <stdexcept>
#include <envire_core/items/Item.hpp>
#include <envire_core/graph/EnvireGraph.hpp>


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
    void saveItem(_ItemData* content)
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
    void deleteItem(_ItemData* content)
    {
        delete (typename envire::core::Item<_ItemData>::Ptr *) ptr;
        ptr = 0;
    }
};