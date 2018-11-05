import os
from cdff_dev import path, logloader, typefromdict, testing
from cdff_dev.extensions.pcl import helpers
from cdff_dev.dfpcs.reconstruction3d import DenseRegistrationFromStereo
from cdff_dev.dfpcs.pointcloudmodellocalisation import FeaturesMatching3D
from nose.tools import assert_true, assert_false


def test_reconstruction3d():
    configpath = os.path.join(
        path.load_cdffpath(),
        "Tests/ConfigurationFiles/DFPCs/Reconstruction3D/"
        "DfpcDenseRegistrationFromStereo_DlrHcru.yaml")
    assert_true(os.path.exists(configpath))

    log = logloader.load_log("test/test_data/logs/frames2.msg")
    img1 = typefromdict.create_from_dict(
        "Frame", log["/hcru0/pt_stereo_rect/left/image"][0])
    img2 = typefromdict.create_from_dict(
        "Frame", log["/hcru0/pt_stereo_rect/left/image"][1])

    dfpc = DenseRegistrationFromStereo()
    dfpc.set_configuration_file(configpath)
    with testing.hidden_stdout():
        dfpc.setup()

    dfpc.leftImageInput(img1)
    dfpc.rightImageInput(img2)
    with testing.hidden_stdout():
        dfpc.run()
    assert_false(dfpc.successOutput())
    dfpc.poseOutput()
    dfpc.pointCloudOutput()


def test_pointcloud_model_localisation():
    configpath = os.path.join(
        path.load_cdffpath(),
        "Tests/ConfigurationFiles/DFPCs/PointCloudModelLocalisation/"
        "DfpcFeaturesMatching3D_conf01.yaml")
    assert_true(os.path.exists(configpath))

    pc_original = helpers.load_ply_file(
        "test/test_data/pointclouds/cube.ply")
    pc_transformed = helpers.load_ply_file(
        "test/test_data/pointclouds/cube.ply")

    dfpc = FeaturesMatching3D()
    dfpc.set_configuration_file(configpath)
    with testing.hidden_stdout():
        dfpc.setup()

    dfpc.sceneInput(pc_transformed)
    dfpc.modelInput(pc_original)
    dfpc.computeModelFeaturesInput(True)
    with testing.hidden_stdout():
        dfpc.run()
    assert_false(dfpc.successOutput())
    dfpc.poseOutput()
