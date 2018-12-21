#include <pcl/io/ply_io.h>
#include <pcl/console/print.h>
#include <boost/shared_ptr.hpp>
#include <string>
#include <Pointcloud.h>
#include <PointCloud.hpp>
#include <PclPointCloudToPointCloudConverter.hpp>


void loadPLYFile(std::string filename, asn1SccPointcloud* output)
{
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    Converters::PclPointCloudToPointCloudConverter pointCloudConverter;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >();
    pcl::io::loadPLYFile(filename, *pclCloud);
    const asn1SccPointcloud* result = pointCloudConverter.Convert(pclCloud);
    *output = *result;
    delete result;
}