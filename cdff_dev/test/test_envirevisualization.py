from cdff_dev import envirevisualization
from nose.tools import assert_true


def test_world_state():
    world_state = envirevisualization.WorldState(
        {"a": "test1", "b": "test2"},
        ["test/test_data/model.urdf"])
    assert_true(world_state.graph_.contains_frame("test1"))
    assert_true(world_state.graph_.contains_frame("body"))