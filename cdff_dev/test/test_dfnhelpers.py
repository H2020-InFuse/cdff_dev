import numpy as np
from cdff_dev import dfnhelpers
import cdff_types
from nose.tools import assert_equal, assert_true, assert_false, assert_is_none
from numpy.testing import assert_array_equal


def test_dfn_base_is_dfn():
    dfn = dfnhelpers.DFNBase()
    assert_true(dfnhelpers.isdfn(dfn))

    dfn.set_configuration_file("")
    dfn.configure()
    dfn.process()


def test_lambda_dfn():
    dfn = dfnhelpers.LambdaDFN(
        lambda x: x ** 2, input_port="x", output_port="y")
    assert_true(dfnhelpers.isdfn(dfn))
    dfn.set_configuration_file("")
    dfn.configure()
    dfn.xInput(5)
    dfn.process()
    y = dfn.yOutput()
    assert_equal(y, 25)


def test_lambda_dfn_without_input():
    dfn = dfnhelpers.LambdaDFN(
        lambda x: x ** 2, input_port="x", output_port="y")
    dfn.set_configuration_file("")
    dfn.configure()
    dfn.process()
    y = dfn.yOutput()
    assert_is_none(y)


def test_merge_without_config():
    merger = dfnhelpers.MergeFramePairDFN()
    merger.configure()

    random_state = np.random.RandomState(0)

    left = cdff_types.Frame()
    left.data.channels = 1
    left.data.cols = 2
    left.data.rows = 3
    left_image = left.array_reference()
    left_image[:, :, :] = random_state.randint(
        0, 256, (3, 2, 1), dtype=np.uint8)

    right = cdff_types.Frame()
    right.data.channels = 1
    right.data.cols = 2
    right.data.rows = 3
    right_image = right.array_reference()
    right_image[:, :, :] = random_state.randint(
        0, 256, (3, 2, 1), dtype=np.uint8)

    merger.leftImageInput(left)
    merger.rightImageInput(right)
    merger.process()
    pair = merger.pairOutput()

    assert_equal(pair.left.data.channels, left.data.channels)
    assert_equal(pair.left.data.cols, left.data.cols)
    assert_equal(pair.left.data.rows, left.data.rows)
    assert_array_equal(pair.left.array_reference(), left_image)

    assert_equal(pair.right.data.channels, right.data.channels)
    assert_equal(pair.right.data.cols, right.data.cols)
    assert_equal(pair.right.data.rows, right.data.rows)
    assert_array_equal(pair.right.array_reference(), right_image)


def test_merge_with_config():
    merger = dfnhelpers.MergeFramePairDFN(
        left_camera_info_stream="/hcru1/pt_stereo_rect/left/camera_info",
        right_camera_info_stream="/hcru1/pt_stereo_rect/right/camera_info",
        left_is_main_camera=True
    )
    merger.set_configuration_file("test/test_data/camera_config.msg")
    merger.configure()

    left = cdff_types.Frame()
    right = cdff_types.Frame()

    merger.leftImageInput(left)
    merger.rightImageInput(right)
    merger.process()
    pair = merger.pairOutput()

    dist_coeffs = pair.left.intrinsic.dist_coeffs.toarray()
    assert_array_equal(dist_coeffs, np.array([-0.188116, 0.227419, 0, 0, 0]))
    camera_matrix = pair.left.intrinsic.camera_matrix
    assert_array_equal(
        camera_matrix,
        np.array([[867.356, 0, 516], [0, 867.356, 386], [0, 0, 1]]))
    assert_equal(pair.left.intrinsic.camera_model, "cam_PINHOLE")

    dist_coeffs = pair.right.intrinsic.dist_coeffs.toarray()
    assert_array_equal(dist_coeffs, np.array([-0.183758, 0.221971, 0, 0, 0]))
    camera_matrix = pair.right.intrinsic.camera_matrix
    assert_array_equal(
        camera_matrix,
        np.array([[867.356, 0, 516], [0, 867.356, 386], [0, 0, 1]]))
    assert_equal(pair.right.intrinsic.camera_model, "cam_PINHOLE")


def test_is_dfn():
    class NoDFN:
        def set_configuration_file(self, filename):
            pass
    class DFN:
        def set_configuration_file(self, filename):
            pass
        def configure(self):
            pass
        def process(self):
            pass

    assert_false(dfnhelpers.isdfn(NoDFN))
    assert_true(dfnhelpers.isdfn(DFN))


def test_is_dfpc():
    class NoDFPC:
        def set_configuration_file(self, filename):
            pass
    class DFPC:
        def set_configuration_file(self, filename):
            pass
        def setup(self):
            pass
        def run(self):
            pass

    assert_false(dfnhelpers.isdfpc(NoDFPC))
    assert_true(dfnhelpers.isdfpc(DFPC))


def test_dfn_adapter():
    class DFPC:
        def set_configuration_file(self, filename):
            self.filename = filename
        def setup(self):
            self.configured = True
        def run(self):
            self.executed = True
        def aInput(self, data):
            self.a = data
        def bOutput(self):
            return self.a

    assert_true(dfnhelpers.isdfpc(DFPC))
    DFPCDFN = dfnhelpers.create_dfn_from_dfpc(DFPC)
    assert_true(dfnhelpers.isdfn(DFPCDFN))
    dfn = DFPCDFN()
    dfn.set_configuration_file("testfile")
    assert_equal(dfn.dfpc.filename, "testfile")
    dfn.configure()
    assert_true(dfn.dfpc.configured)
    dfn.aInput(5)
    dfn.process()
    assert_true(dfn.dfpc.executed)
    b = dfn.bOutput()
    assert_equal(b, 5)


class SquareDFPC:
    def __init__(self):
        self.x = 0.0

    def set_configuration_file(self, filename):
        pass

    def setup(self):
        pass

    def xInput(self, x):
        self.x = x

    def run(self):
        self.y = self.x ** 2

    def yOutput(self):
        return self.y


def test_wrap_dfpc():
    dfpc = SquareDFPC()
    dfn = dfnhelpers.wrap_dfpc_as_dfn(dfpc)
    assert_true(dfnhelpers.isdfn(dfn))
    dfn.configure()
    dfn.xInput(5)
    dfn.process()
    sq = dfn.yOutput()
    assert_equal(sq, 25)
