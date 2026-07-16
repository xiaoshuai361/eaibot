# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

"""ROS-free Python 2/3 smoke check for the tagless grasp helpers."""

import math
import os
import sys

import cv2
import numpy as np


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.normpath(os.path.join(TESTS_DIR, os.pardir, "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from block_detector_protocol import read_message, write_message
from block_grasp_vision import (
    compute_link_targets,
    deproject_pixel,
    find_block_quadrilateral,
    rotate_vector_by_quaternion,
    sample_depth_m,
)


def assert_close(actual, expected, tolerance=1e-6):
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    if not np.allclose(actual_array, expected_array, rtol=0.0, atol=tolerance):
        raise AssertionError("%r != %r" % (actual, expected))


def protocol_pipe_smoke():
    request_read_fd, request_write_fd = os.pipe()
    response_read_fd, response_write_fd = os.pipe()
    request_reader = os.fdopen(request_read_fd, "rb")
    request_writer = os.fdopen(request_write_fd, "wb")
    response_reader = os.fdopen(response_read_fd, "rb")
    response_writer = os.fdopen(response_write_fd, "wb")
    try:
        request = {
            "id": 7,
            "image_path": u"/tmp/无标签物块.png",
            "target": u"Fire extinguishing device",
        }
        write_message(request_writer, request)
        received_request = read_message(request_reader)
        if received_request != request:
            raise AssertionError("request pipe roundtrip changed the JSON payload")

        response = {
            "id": received_request["id"],
            "ok": True,
            "class_id": 1,
            "class_name": u"Fire extinguishing device",
            "confidence": 0.98,
            "box": [85.0, 65.0, 135.0, 135.0],
            "target": received_request["target"],
        }
        write_message(response_writer, response)
        received_response = read_message(response_reader)
        if received_response != response:
            raise AssertionError("response pipe roundtrip changed the JSON payload")
    finally:
        request_reader.close()
        request_writer.close()
        response_reader.close()
        response_writer.close()


def vision_geometry_smoke():
    image = np.zeros((200, 240, 3), dtype=np.uint8)
    image[40:161, 60:181] = 255
    image[75:126, 95:146] = 30
    localization = find_block_quadrilateral(
        image_bgr=image,
        detector_box=(90.0, 70.0, 150.0, 130.0),
        roi_margin=1.0,
        min_area_pixels=1000.0,
        max_aspect_error=0.20,
        min_rectangularity=0.85,
        ambiguity_ratio=0.95,
    )
    assert_close(localization["center"], (120.0, 100.0), tolerance=1.0)

    depth = np.full((200, 240), 1000, dtype=np.uint16)
    depth[99, 119] = 0
    depth_m, quality = sample_depth_m(
        depth,
        localization["center"],
        "16UC1",
        2,
        0.20,
        2.00,
        0.80,
        0.01,
    )
    assert_close(depth_m, 1.0)
    if quality["valid_ratio"] < 0.80:
        raise AssertionError("depth valid ratio was not enforced")

    camera_point = deproject_pixel(
        localization["center"][0],
        localization["center"][1],
        depth_m,
        500.0,
        500.0,
        120.0,
        100.0,
    )
    assert_close(camera_point, (0.0, 0.0, 1.0))

    half_angle = math.pi / 4.0
    rotated_tcp = rotate_vector_by_quaternion(
        (0.10, 0.0, 0.0),
        (0.0, 0.0, math.sin(half_angle), math.cos(half_angle)),
    )
    assert_close(rotated_tcp, (0.0, 0.10, 0.0))
    contact, precontact = compute_link_targets(
        (0.30, 0.20, 0.50), rotated_tcp, 0.05
    )
    assert_close(contact, (0.30, 0.10, 0.50))
    assert_close(precontact, (0.30, 0.05, 0.50))


def main():
    protocol_pipe_smoke()
    vision_geometry_smoke()
    print("OK: Python 2/3 protocol, OpenCV depth, quadrilateral, and TCP geometry smoke")
    return 0


if __name__ == "__main__":
    sys.exit(main())
