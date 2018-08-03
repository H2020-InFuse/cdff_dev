#include "UTMConverter.hpp"
#include <iostream>
#include <ogr_spatialref.h>

using namespace std;
using namespace gps;

UTMConverter::UTMConverter()
    : utm_zone(32)
    , utm_north(true)
    , coTransform(NULL)
{
    origin.nCount = 3;
    for(int i=0;i<3;++i){
        origin.arr[i] = 0.0;
    }
    createCoTransform();
}

void UTMConverter::createCoTransform()
{
    OGRSpatialReference oSourceSRS;
    OGRSpatialReference oTargetSRS;

    oSourceSRS.SetWellKnownGeogCS("WGS84");
    oTargetSRS.SetWellKnownGeogCS("WGS84");
    oTargetSRS.SetUTM(this->utm_zone, this->utm_north);

    OGRCoordinateTransformation* newTransform =
        OGRCreateCoordinateTransformation(&oSourceSRS, &oTargetSRS);
    if (newTransform == NULL)
        throw runtime_error("Failed to initialize CoordinateTransform");

    delete coTransform;
    coTransform = newTransform;
}

void UTMConverter::setUTMZone(int zone)
{
    this->utm_zone = zone;
    createCoTransform();
}

void UTMConverter::setUTMNorth(bool north)
{
    this->utm_north = north;
    createCoTransform();
}

int UTMConverter::getUTMZone() const
{
    return this->utm_zone;
}

bool UTMConverter::getUTMNorth() const
{
    return this->utm_north;
}

asn1SccPosition UTMConverter::getNWUOrigin() const
{
    return this->origin;
}

void UTMConverter::setNWUOrigin(asn1SccPosition origin)
{
    this->origin = origin;
}

asn1SccRigidBodyState UTMConverter::convertToUTM(const asn1SccTime &time, const double &longitude, const double &latitude, const double &altitude) const
{
    asn1SccRigidBodyState position;
    position.timestamp = time;

    double northing = latitude;
    double easting  = longitude;
    double alt = altitude;

    coTransform->Transform(1, &easting, &northing, &alt);

    position.pos.arr[0] = easting;
    position.pos.arr[1] = northing;
    position.pos.arr[2] = alt;
    return position;
}

asn1SccRigidBodyState UTMConverter::convertToNWU(const asn1SccTime &time, const double &longitude, const double &latitude, const double &altitude) const
{
    return convertToNWU(convertToUTM(time, longitude, latitude, altitude));
}

asn1SccRigidBodyState UTMConverter::convertToNWU(const asn1SccRigidBodyState &utm) const
{
    asn1SccRigidBodyState position = utm;
    double easting  = position.pos.arr[0];
    double northing = position.pos.arr[1];
    position.pos.arr[0] = northing;
    position.pos.arr[1] = 1000000 - easting;
    for(int i=0;i<3;++i){
        position.pos.arr[i] -= origin.arr[i];
    }
    return position;
}
