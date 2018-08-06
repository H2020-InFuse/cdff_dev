#include "ASN1Test.hpp"

#include <Time.h>
#include <Eigen.h>

namespace CDFF
{
namespace DFN
{
namespace ASN1Test
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

    someVector.arr[0] = inCurrentTime.microseconds + 1;
    someVector.arr[1] = inCurrentTime.microseconds + 2;
    someVector.arr[2] = inCurrentTime.microseconds + 3;

    outSomeVector = someVector;
}

}
}
}
