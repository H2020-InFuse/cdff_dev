#include <string>
#include <envire_core/items/Item.hpp>
#include <envire_core/graph/EnvireGraph.hpp>


template <class _ItemData>
void addItemToFrame(
    envire::core::EnvireGraph& graph, const envire::core::FrameId& frame,
    _ItemData* contentPtr)
{
    typename envire::core::Item<_ItemData>::Ptr item = typename envire::core::Item<_ItemData>::Ptr(new envire::core::Item<_ItemData>(*contentPtr));
    graph.addItemToFrame(frame, item);
}


template <class _ItemData>
unsigned getItemCount(
    envire::core::EnvireGraph& graph, const envire::core::FrameId& frame,
    _ItemData* contentPtr)
{
    return graph.getItemCount<typename envire::core::Item<_ItemData> >(frame);
}