#include "Square.hpp"

namespace dfn_ci {

Square::Square()
{
}

Square::~Square()
{
}

void Square::configure() 
{
    // TODO Fill in configure functionality here
}


void Square::process() {
    outx_squared = inx * inx;
}

}