# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

"""ROS-free Python 2/3 smoke check for the monocular tagless helpers."""

import math
import os
import sys

import numpy as np


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.normpath(os.path.join(TESTS_DIR, os.pardir, "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from block_mono_vision import (
    box_geometry,
    deproject_pixel_to_camera_mm,
    estimate_distance_mm,
    is_detection_usable,
    stable_median_observation,
)


def assert_close(actual, expected, tolerance=1e-6):
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    if not np.allclose(actual_array, expected_array, rtol=0.0, atol=tolerance):
        raise AssertionError("%r != %r" % (actual, expected))


def monocular_geometry_smoke():
    detection = {
        "confidence": 0.92,
        "box": [90.0, 70.0, 150.0, 130.0],
    }
    rules = {
        "confidence_min": 0.70,
        "box_width_min_px": 30.0,
        "box_aspect_ratio_min": 0.75,
        "box_aspect_ratio_max": 1.30,
    }
    usable, reason = is_detection_usable(detection, rules)
    if not usable:
        raise AssertionError(reason)

    geometry = box_geometry(detection["box"])
    assert_close((geometry["u"], geometry["v"], geometry["w"]), (120.0, 100.0, 60.0))

    observations = [
        {"u": 120.0, "v": 100.0, "w": 60.0, "h": 60.0, "confidence": 0.91},
        {"u": 121.0, "v": 101.0, "w": 60.5, "h": 60.0, "confidence": 0.92},
        {"u": 119.0, "v": 99.0, "w": 59.5, "h": 60.0, "confidence": 0.90},
    ]
    stable = stable_median_observation(observations, 3, 2.0, 0.03)
    assert_close((stable["u"], stable["v"], stable["w"]), (120.0, 100.0, 60.0))

    z_mm = estimate_distance_mm(
        "theory",
        stable["w"],
        500.0,
        30.0,
        "fire",
        {},
    )
    assert_close(z_mm, 250.0)
    camera_point = deproject_pixel_to_camera_mm(
        stable["u"], stable["v"], z_mm, 500.0, 500.0, 120.0, 100.0)
    assert_close(camera_point, (0.0, 0.0, 250.0))


def main():
    monocular_geometry_smoke()
    print("OK: Python 2/3 monocular block geometry smoke")
    return 0


if __name__ == "__main__":
    sys.exit(main())
