#include "ASN1Test.hpp"
#include <Time.h>

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
    Time sometime;
    sometime.microseconds = incurrenttime.microseconds + 1;
    sometime.usecPerSec = 1000000;
    outnexttime = sometime;
}

}
