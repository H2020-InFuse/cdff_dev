#include "ASN1Test.hpp"

#include <Time.h>
#include <Eigen.h>

namespace dfn_ci
{

ASN1Test::ASN1Test()
{
}

ASN1Test::~ASN1Test()
{
}

void ASN1Test::configure()
{
}

void ASN1Test::process()
{
    asn1SccVector3d someVector;

    someVector.nCount = 3;
    someVector.arr[0] = inCurrentTime.microseconds + 1;
    someVector.arr[1] = inCurrentTime.microseconds + 2;
    someVector.arr[2] = inCurrentTime.microseconds + 3;

    outSomeVector = someVector;
}

}
