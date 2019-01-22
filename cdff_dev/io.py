import numpy as np
try:
    import gdal
except ImportError:
    raise ImportError("Please install python3.5-gdal with apt.")
import cdff_types


class GeoTiffMap:
    """Map loaded from GeoTIFF.

    Parameters
    ----------
    filename : str
        Path to map file

    verbose : int, optional (default: 0)
        Verbosity level
    """
    def __init__(self, filename, verbose=0):
        self.ds = gdal.Open(filename)
        geo_transform = self.ds.GetGeoTransform()
        self.origin_x, self.origin_y = geo_transform[0], geo_transform[3]
        self.col_scale, self.row_scale = geo_transform[1], geo_transform[5]
        self.band = self.ds.GetRasterBand(1)
        self.undefined = self.band.GetNoDataValue()
        self.map_data = self.band.ReadAsArray().astype(np.float32)
        self.rows, self.cols = self.map_data.shape
        self.min_height = self.map_data[self.map_data != self.undefined].min()
        self.max_height = self.map_data[self.map_data != self.undefined].max()

        if verbose:
            print('RasterCount: ', self.ds.RasterCount)
            print('undefined', self.undefined)
            print('col_scale: ', self.col_scale)
            print('row_scale: ', self.row_scale)
            print('origin_x, origin_y:', self.origin_x, self.origin_y)
            print('rows, cols: ', self.rows, self.cols)
            print('min_height: ', self.min_height)
            print('max_height: ', self.max_height)

        if self.col_scale != -self.row_scale:
            raise ValueError(
                "Maps with different row and column scale cannot be handled")
        assert geo_transform[2] == 0.0
        assert geo_transform[4] == 0.0

    def slice(self, map_slice_rows=(0, 1080), map_slice_cols=(0, 1920)):
        """Slice map object from GeoTIFF.

        Parameters
        ----------
        map_slice_rows : pair of int, optional (default: (0, 1080))
            Row slice from original map

        map_slice_cols : pair of int, optional (default: (0, 1920))
            Column slice from original map

        Returns
        -------
        origin : pair of float
            x and y position with respect to map's origin

        map : Map
            Slice of full map
        """
        map = cdff_types.Map()
        map.metadata.scale = self.col_scale
        map.metadata.err_values[0].type = "error_UNDEFINED"
        map.metadata.err_values[0].value = self.undefined
        # maximum rows: 1080
        map.data.rows = map_slice_cols[1] - map_slice_cols[0]
        # maximum cols: 1920
        map.data.cols = map_slice_rows[1] - map_slice_rows[0]
        # maximum channels: 4
        map.data.channels = 1
        # only for dtype == np.float32
        map.data.row_size = map.data.cols * 4
        map.data.depth = "depth_32F"

        data = map.data.array_reference()
        data[:, :, 0] = self.map_data[map_slice_rows[0]:map_slice_rows[1],
                                      map_slice_cols[0]:map_slice_cols[1]].T
        origin = (map_slice_rows[0] * self.row_scale,
                  map_slice_cols[0] * self.col_scale)
        return origin, map
