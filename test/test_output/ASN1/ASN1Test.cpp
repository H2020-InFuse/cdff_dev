#include "ASN1Test.hpp"
#include <Time.h>
#include <Eigen.h>

namespace dfn_ci {

ASN1Test::ASN1Test()
{
}

ASN1Test::~ASN1Test()
{
}

void ASN1Test::configure()
{
    // TODO Fill in configure functionality here
}


void ASN1Test::process() {
    Vector3d someVector;
    someVector.nCount = 3;
    someVector.arr[0] = incurrenttime.microseconds + 1;
    someVector.arr[1] = incurrenttime.microseconds + 2;
    someVector.arr[2] = incurrenttime.microseconds + 3;
    outsomevector = someVector;
}

}
