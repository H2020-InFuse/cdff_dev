#include "Square.hpp"

namespace dfn_ci {

Square::Square()
{
}

Square::~Square()
{
}

bool Square::configure() 
{
    // TODO Fill in configure functionality here
    return true;
}


bool Square::process() {
    outx_squared = inx * inx;
    return true;
}

}