from cdff_dev.dfns.stereodegradation import StereoDegradation
import cdff_types


def test_stereodegradation():
    dfn = StereoDegradation()
    dfn.configure()

    original_image_pair = cdff_types.FramePair()
    dfn.originalImagePairInput(original_image_pair)
    dfn.process()
    degraded_image_pair = cdff_types.FramePair()
    dfn.degradedImagePairOutput(degraded_image_pair)