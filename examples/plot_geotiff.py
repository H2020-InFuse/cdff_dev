import sys
import matplotlib.pyplot as plt
from cdff_dev import io


def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "test/test_data/maps/res1.tif"
    gtm = io.GeoTiffMap(filename, verbose=1)
    origin, m = gtm.slice((250, 500), (600, 800))
    m.data.array_reference()[m.data.array_reference() == gtm.undefined] = 0.0
    plt.imshow(m.data.array_reference().squeeze())
    plt.gray()
    plt.show()


if __name__ == "__main__":
    main()
