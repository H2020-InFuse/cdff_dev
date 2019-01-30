"""
============
Plot GeoTIFF
============

The GeoTIFF image is used to store ground truth digital elevation maps.
In this example, we will display a simple GeoTIFF map.

Examples:

    python examples/plot_geotiff.py 2018_11_27_xalucadunes_dsm.tif 7000 8080 5500 7420
    python examples/plot_geotiff.py 2018_11_28_og3hires_dsm.tif 5000 6000 5000 6000
    python examples/plot_geotiff.py 2018_11_30_sherpashort_dsm.tif 6540 7540 2340 3340
"""
import matplotlib.pyplot as plt
import numpy as np
import sys
from cdff_dev import io


def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "test/test_data/maps/res1.tif"
    if len(sys.argv) > 5:
        x_min, x_max, y_min, y_max = map(int, sys.argv[2:6])
    else:
        x_min, x_max, y_min, y_max = 250, 500, 600, 800

    gtm = io.GeoTiffMap(filename, verbose=1)

    m = gtm.downsample(10)
    m = m.data.array_reference().squeeze().copy().T
    m[m == gtm.undefined] = np.nan
    plt.subplot(121)
    plt.title("downsampled full map")
    plt.imshow(m)
    plt.xticks(())
    plt.yticks(())
    plt.gray()

    _, m = gtm.slice((x_min, x_max), (y_min, y_max))
    m = m.data.array_reference().squeeze().copy().T
    m[m == gtm.undefined] = np.nan
    plt.subplot(122)
    plt.title("slice of map: (%d, %d) x (%d, %d)"
              % (x_min, x_max, y_min, y_max))
    plt.imshow(m)
    plt.xticks(())
    plt.yticks(())
    plt.gray()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
