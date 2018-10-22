import os
import glob
from cdff_dev import loggermsgpack, logloader
from nose.tools import assert_equal, assert_not_in


def test_log_all():
    output_prefix = "/tmp/test_loggermsgpack_all"
    try:
        logger = loggermsgpack.MsgPackLogger(output_prefix, max_samples=3)
        logger.report_node_output("A.port", 5, 0)
        logger.report_node_output("B.port", 6, 1)
        logger.report_node_output("C.port", 7, 2)
        logger.report_node_output("A.port", 8, 3)
        logger.report_node_output("B.port", 9, 4)
        logger.report_node_output("C.port", 10, 5)

        filename = "%s_000000000.msg" % output_prefix
        log = logloader.load_log(filename)
        assert_equal(log["A.port"][0], 5)
        assert_equal(log["A.port.meta"]["type"], "int")
        assert_equal(log["A.port.meta"]["timestamps"][0], 0)
        assert_equal(log["B.port"][0], 6)
        assert_equal(log["B.port.meta"]["timestamps"][0], 1)
        assert_equal(log["C.port"][0], 7)
        assert_equal(log["C.port.meta"]["timestamps"][0], 2)

        filename = "%s_000000001.msg" % output_prefix
        log = logloader.load_log(filename)
        assert_equal(log["A.port"][0], 8)
        assert_equal(log["A.port.meta"]["timestamps"][0], 3)
        assert_equal(log["B.port"][0], 9)
        assert_equal(log["B.port.meta"]["timestamps"][0], 4)
        assert_equal(log["C.port"][0], 10)
        assert_equal(log["C.port.meta"]["timestamps"][0], 5)
    finally:
        filenames = glob.glob("%s_*.msg" % output_prefix)
        for filename in filenames:
            os.remove(filename)


def test_only_one_stream():
    output_prefix = "/tmp/test_loggermsgpack_one"
    try:
        logger = loggermsgpack.MsgPackLogger(
            output_prefix, max_samples=1, stream_names="A.port")
        logger.report_node_output("A.port", 5, 0)
        logger.report_node_output("B.port", 6, 1)
        logger.report_node_output("C.port", 7, 2)
        logger.report_node_output("A.port", 8, 3)
        logger.report_node_output("B.port", 9, 4)
        logger.report_node_output("C.port", 10, 5)

        filename = "%s_000000000.msg" % output_prefix
        log = logloader.load_log(filename)
        assert_equal(log["A.port"][0], 5)
        assert_equal(log["A.port.meta"]["type"], "int")
        assert_equal(log["A.port.meta"]["timestamps"][0], 0)
        assert_not_in("B.port", log)
        assert_not_in("B.port.meta", log)
        assert_not_in("C.port", log)
        assert_not_in("C.port.meta", log)

        filename = "%s_000000001.msg" % output_prefix
        log = logloader.load_log(filename)
        assert_equal(log["A.port"][0], 8)
        assert_equal(log["A.port.meta"]["type"], "int")
        assert_equal(log["A.port.meta"]["timestamps"][0], 3)
        assert_not_in("B.port", log)
        assert_not_in("B.port.meta", log)
        assert_not_in("C.port", log)
        assert_not_in("C.port.meta", log)
    finally:
        filenames = glob.glob("%s_*.msg" % output_prefix)
        for filename in filenames:
            os.remove(filename)
