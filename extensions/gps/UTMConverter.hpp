#ifndef _GPS_UTMCONVERTER_HPP_
#define _GPS_UTMCONVERTER_HPP_

#include <RigidBodyState.h>
#include <Pose.h>
#include <time.h>

class OGRCoordinateTransformation;

namespace gps
{
    class UTMConverter
    {
        private:
            int utm_zone;
            bool utm_north;
            asn1SccPosition origin;
            OGRCoordinateTransformation *coTransform;

            void createCoTransform();

            /** Convert a UTM-converted GPS solution into NWU coordinates (Rock's convention)
             */
            asn1SccRigidBodyState convertToNWU(const asn1SccRigidBodyState &utm) const;

        public:
            UTMConverter();

            /** Sets the UTM zone
             */
            void setUTMZone(int zone);

            /** Sets whether we're north or south of the equator
             */
            void setUTMNorth(bool north);
            
            /** Get the UTM zone */
            int getUTMZone() const;

            /** Get whether we're north or south of the equator */
            bool getUTMNorth() const;

            /** Set a position that will be removed from the computed UTM
             * solution
             */
            void setNWUOrigin(asn1SccPosition origin);

            /** Returns the current origin position in UTM coordinates
             */
            asn1SccPosition getNWUOrigin() const;

            /** Convert a GPS solution into UTM coordinates
             *
             * The returned RBS will has all its fields invalidated (only the
             * timestamp updated) if there is no solution
             */
            asn1SccRigidBodyState convertToUTM(const asn1SccTime &time, const double &longitude, const double &latitude, const double &altitude) const;

            /** Convert a GPS solution into NWU coordinates (Rock's convention)
             *
             * The returned RBS will has all its fields invalidated (only the
             * timestamp updated) if there is no solution
             */
            asn1SccRigidBodyState convertToNWU(const asn1SccTime &time, const double &longitude, const double &latitude, const double &altitude) const;

    };

} // end namespace gps

#endif // _GPS_UTMCONVERTER_HPP_
