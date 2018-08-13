import cdff_envire
import cdff_types
import os
import re
import numpy as np
from nose.tools import assert_equal, assert_not_equal, \
    assert_false, assert_true, assert_greater, \
    assert_regexp_matches, assert_raises_regexp, assert_almost_equal, \
    assert_is_not_none
from numpy.testing import assert_array_almost_equal


def test_get_set_microseconds():
    t = cdff_envire.Time()
    m = 1000023
    t.microseconds = m
    assert_equal(t.microseconds, m)


def test_time_str():
    t = cdff_envire.Time.now()
    assert_regexp_matches(str(t), "<time=\d{8}-\d{2}:\d{2}:\d{2}:\d{6}>")


def test_no_overflow():
    t = cdff_envire.Time.now()
    mus = t.microseconds
    assert_greater(mus, 0)
    t.microseconds = mus


def test_time_operators():
    t1 = cdff_envire.Time()
    t1.microseconds = 0
    t2 = cdff_envire.Time()
    t2.microseconds = 1
    t3 = cdff_envire.Time()
    t3.microseconds = 1
    t4 = cdff_envire.Time()
    t4.microseconds = 2

    assert_true(t1 < t2 < t4)
    assert_true(t4 > t2 > t1)
    assert_true(t1 != t2)
    assert_false(t2 != t3)
    assert_true(t2 == t3)
    assert_false(t1 == t2)
    assert_true(t2 >= t3)
    assert_false(t2 > t3)
    assert_true(t2 <= t3)
    assert_false(t2 < t3)

    assert_true(t2 + t2 == t4)
    assert_true(t4 - t2 == t2)
    assert_true(t4 // 2 == t2)
    assert_true(t2 * 2 == t4)

    t5 = cdff_envire.Time()
    t5.microseconds = 10
    t5 //= 2
    assert_equal(t5.microseconds, 5)
    t5 -= t2 * 4
    assert_equal(t5, t2)
    t5 *= 2
    assert_equal(t5, t4)
    t5 += t1
    assert_equal(t5, t4)


def test_time_assign():
    t1 = cdff_envire.Time()
    t2 = cdff_envire.Time.now()
    assert_not_equal(t1, t2)
    t1.assign(t2)
    assert_equal(t1, t2)


def test_vector3d_ctor():
    v = cdff_envire.Vector3d(1.0, 2.0, 3.0)
    assert_equal(str(v), "[1.00, 2.00, 3.00]")


def test_vector3d_fromarray():
    v = cdff_envire.Vector3d()
    a = np.array([-2.32, 2.42, 54.23])
    v.fromarray(a)
    assert_equal(v[0], a[0])
    assert_equal(v[1], a[1])
    assert_equal(v[2], a[2])


def test_vector3d_as_ndarray():
    random_state = np.random.RandomState(843)
    r = random_state.randn(3, 3)
    v = cdff_envire.Vector3d(3.23, 2.24, 3.63)
    rv = r.dot(np.asarray(v))
    rv2 = r.dot(v.toarray())
    assert_array_almost_equal(rv, rv2)


def test_vector3d_get_set_data():
    v = cdff_envire.Vector3d(1.0, 2.0, 3.0)
    v.x = 5.0
    v.y = 6.0
    v.z = 7.0
    assert_equal(v.x, 5.0)
    assert_equal(v.y, 6.0)
    assert_equal(v.z, 7.0)


def test_vector3d_array_access():
    v = cdff_envire.Vector3d(1.0, 2.0, 3.0)
    assert_equal(v[0], 1.0)
    v[1] = 4.0
    assert_equal(v[1], 4.0)
    assert_raises_regexp(KeyError, "index must be", lambda i: v[i], -1)
    def assign(i):
        v[i] = 5.0
    assert_raises_regexp(KeyError, "index must be", assign, 3)


def test_vector3d_assign():
    obj1 = cdff_envire.Vector3d()
    obj2 = cdff_envire.Vector3d(1.0, 2.0, 3.0)
    assert_not_equal(obj1, obj2)
    obj1.assign(obj2)
    assert_equal(obj1, obj2)


def test_norms():
    v = cdff_envire.Vector3d(1.0, 2.0, 3.0)
    assert_almost_equal(v.norm(), 3.741657387)
    assert_equal(v.squared_norm(), 14.0)


def test_quaterniond_ctor():
    q = cdff_envire.Quaterniond(w=1.0, x=0.0, y=0.0, z=0.0)
    assert_equal(str(q), "[real=1.00, im=(0.00, 0.00, 0.00)]")


def test_transform_with_cov_ctor():
    cdff_envire.TransformWithCovariance()


def test_transform_set_get_translation():
    p = cdff_envire.TransformWithCovariance()
    t = cdff_envire.Vector3d(1.0, 2.0, 3.0)
    p.translation = t
    assert_equal(str(p.translation), "[1.00, 2.00, 3.00]")


def test_transform_set_get_orientation():
    p = cdff_envire.TransformWithCovariance()
    o = cdff_envire.Quaterniond(w=1.0, x=0.0, y=0.0, z=0.0)
    p.orientation = o
    assert_equal(str(p.orientation), "[real=1.00, im=(0.00, 0.00, 0.00)]")


def test_default_transform_ctor():
    cdff_envire.Transform()


def test_transform_time_transform_with_covariance_ctor():
    t = cdff_envire.Time()

    p = cdff_envire.TransformWithCovariance()
    p.translation = cdff_envire.Vector3d(1.0, 2.0, 3.0)
    p.orientation = cdff_envire.Quaterniond(w=1.0, x=0.0, y=0.0, z=0.0)

    transform = cdff_envire.Transform(time=t, transform_with_covariance=p)
    result = re.match(
        r"19\d\d\d\d\d\d-\d\d:\d\d:\d\d\nt: \(1.00 2.00 3.00\)\nr: \(1.00 0.00 0.00 0.00\)",
        str(transform), re.MULTILINE)
    assert_is_not_none(result)


def test_envire_graph_add_frame():
    g = cdff_envire.EnvireGraph()
    g.add_frame("test")
    assert_true(g.contains_frame("test"))


def test_envire_graph_add_frame_twice():
    g = cdff_envire.EnvireGraph()
    g.add_frame("test")
    assert_raises_regexp(
        RuntimeError, "Frame test already exists",
        g.add_frame, "test")


def test_envire_graph_add_get_transform():
    g = cdff_envire.EnvireGraph()
    g.add_frame("1")
    g.add_frame("2")

    t = cdff_envire.Time()
    p = cdff_envire.TransformWithCovariance()
    p.translation = cdff_envire.Vector3d(1.0, 2.0, 3.0)
    p.orientation = cdff_envire.Quaterniond(1.0, 0.0, 0.0, 0.0)
    transform = cdff_envire.Transform(time=t, transform_with_covariance=p)

    g.add_transform("1", "2", transform)
    assert_true(g.contains_edge("1", "2"))
    t2 = g.get_transform("1", "2")
    assert_equal(str(transform), str(t2))
    g.remove_transform("1", "2")
    assert_false(g.contains_edge("1", "2"))


def test_envire_graph_add_transform_twice():
    g = cdff_envire.EnvireGraph()
    g.add_frame("1")
    g.add_frame("2")
    g.add_transform("1", "2", cdff_envire.Transform())
    assert_raises_regexp(
        RuntimeError, "Edge .* already exists",
        g.add_transform, "1", "2", cdff_envire.Transform())


def test_envire_graph_get_missing_transform():
    g = cdff_envire.EnvireGraph()
    g.add_frame("1")
    g.add_frame("2")
    assert_raises_regexp(
        RuntimeError, "Transform .* doesn't exist",
        g.get_transform, "1", "2")


def test_envire_graph_num_vertices():
    g = cdff_envire.EnvireGraph()
    for i in range(4):
        g.add_frame(str(i))
    assert_equal(g.num_vertices(), 4)


def test_envire_graph_num_edges():
    g = cdff_envire.EnvireGraph()
    for i in range(5):
        g.add_frame(str(i))
    for i, j in zip(range(4), range(1, 5)):
        g.add_transform(str(i), str(j), cdff_envire.Transform())
    # The inverse transforms will be added to the graph, too!
    assert_equal(g.num_edges(), 2 * 4)


def test_envire_graph_copy_ctor():
    g = cdff_envire.EnvireGraph()
    g.add_frame("1")
    g.add_frame("2")
    g2 = cdff_envire.EnvireGraph(g)
    assert_true(g2.contains_frame("1"))
    assert_true(g2.contains_frame("2"))


def test_envire_graph_update_transform():
    g = cdff_envire.EnvireGraph()
    g.add_frame("1")
    g.add_frame("2")

    t = cdff_envire.Time()
    p = cdff_envire.TransformWithCovariance()
    p.translation = cdff_envire.Vector3d(1.0, 2.0, 3.0)
    p.orientation = cdff_envire.Quaterniond(1.0, 0.0, 0.0, 0.0)
    transform = cdff_envire.Transform(time=t, transform_with_covariance=p)

    g.add_transform("1", "2", transform)
    transform.transform.translation = cdff_envire.Vector3d(2.0, 2.0, 3.0)
    g.update_transform("1", "2", transform)
    t2 = g.get_transform("1", "2")
    assert_equal(str(transform), str(t2))


def test_envire_graph_compute_transform():
    g = cdff_envire.EnvireGraph()

    g.add_frame("A")
    g.add_frame("B")
    g.add_frame("C")

    A2B = cdff_envire.Transform(translation=cdff_envire.Vector3d(1.0, 0.0, 1.0),
                                orientation=cdff_envire.Quaterniond())
    g.add_transform("A", "B", A2B)
    B2C = cdff_envire.Transform(translation=cdff_envire.Vector3d(0.0, 1.0, 0.0),
                                orientation=cdff_envire.Quaterniond())
    g.add_transform("B", "C", B2C)

    C2A = g.get_transform("C", "A")
    assert_equal(C2A.transform.translation.x, -1.0)
    assert_equal(C2A.transform.translation.y, -1.0)
    assert_equal(C2A.transform.translation.z, -1.0)


def test_envire_graph_save_load():
    filename = "graph_test.envire"
    try:
        g = cdff_envire.EnvireGraph()
        g.add_frame("A")
        g.add_frame("B")
        g.add_transform("A", "B", cdff_envire.Transform())
        g.save_to_file(filename)

        g2 = cdff_envire.EnvireGraph()
        g2.load_from_file(filename)
        assert_true(g2.contains_frame("A"))
        assert_true(g2.contains_frame("B"))
        assert_true(g2.contains_edge("A", "B"))
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def test_make_item():
    item = cdff_envire.GenericItem()
    content = cdff_types.RigidBodyState()
    content.source_frame = "A"
    content.target_frame = "B"
    item.initialize(content)
    content2 = cdff_types.RigidBodyState()
    item.get_data(content2)
    assert_equal(content2.source_frame, "A")
    assert_equal(content2.target_frame, "B")


def test_envire_graph_add_remove_frame():
    g = cdff_envire.EnvireGraph()
    g.add_frame("test")
    assert_true(g.contains_frame("test"))
    g.remove_frame("test")
    assert_false(g.contains_frame("test"))


def test_envire_graph_add_item():
    g = cdff_envire.EnvireGraph()
    g.add_frame("test")
    item = cdff_envire.GenericItem()
    content = cdff_types.RigidBodyState()
    g.add_item_to_frame("test", item, content)
    assert_equal(g.get_item_count("test", content), 1)
    assert_equal(g.get_total_item_count("test"), 1)
    g.remove_item_from_frame(item, content)


def test_envire_urdf_file_does_not_exist():
    g = cdff_envire.EnvireGraph()
    assert_raises_regexp(
        IOError, "File .* does not exist", cdff_envire.load_urdf,
        g, "does_not_exist.urdf")


def test_envire_urdf_load():
    g = cdff_envire.EnvireGraph()
    cdff_envire.load_urdf(g, "test/test_data/model.urdf")
    assert_true(g.contains_frame("body"))
    assert_true(g.contains_frame("lower_dynamixel"))
    assert_true(g.contains_edge("body", "lower_dynamixel"))


def test_envire_load_meshes():
    g = cdff_envire.EnvireGraph()
    cdff_envire.load_urdf(
        g, "test/test_data/urdf_sherpa_meshes_LQ/sherpa_tt.urdf",
        load_visuals=True)
    assert_equal(g.num_vertices(), 45)
    assert_equal(g.num_edges(), 2 * 44)
    assert_true(g.contains_frame("body"))
    assert_equal(g.get_total_item_count("body"), 1)
    assert_true(g.contains_frame("outer_leg_rear_right"))
    assert_equal(g.get_total_item_count("outer_leg_rear_right"), 1)
