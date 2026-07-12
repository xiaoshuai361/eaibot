#!/usr/bin/env python3
# coding=utf-8

import importlib.util
import math
import os
import sys
import threading
import types
import unittest

import cv2
import numpy as np


def load_line_cy():
    rospy = types.ModuleType("rospy")
    rospy.is_shutdown = lambda: False
    sys.modules.setdefault("rospy", rospy)

    geometry_msgs = types.ModuleType("geometry_msgs")
    geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")

    class Twist(object):
        pass

    geometry_msgs_msg.Twist = Twist
    geometry_msgs.msg = geometry_msgs_msg
    sys.modules.setdefault("geometry_msgs", geometry_msgs)
    sys.modules.setdefault("geometry_msgs.msg", geometry_msgs_msg)

    path = os.path.join(os.path.dirname(__file__), "line_cy.py")
    spec = importlib.util.spec_from_file_location("line_cy_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


line_cy = load_line_cy()


def fill_rotated_rect(binary, center, size, angle, value=255):
    polygon = cv2.boxPoints((center, size, angle)).astype(np.int32)
    cv2.fillConvexPoly(binary, polygon, value)
    return polygon


class StoplineGeometryTests(unittest.TestCase):
    def test_long_edge_angle_is_zero_for_horizontal_rectangle(self):
        points = cv2.boxPoints(((100.0, 80.0), (120.0, 18.0), 0.0))
        self.assertAlmostEqual(line_cy.long_edge_angle_deg(points), 0.0, delta=0.2)

    def test_long_edge_angle_keeps_diagonal_sign(self):
        points = cv2.boxPoints(((100.0, 80.0), (120.0, 18.0), 17.0))
        self.assertTrue(math.isclose(line_cy.long_edge_angle_deg(points), 17.0, abs_tol=0.3))


class CrosswalkMaskTests(unittest.TestCase):
    def test_rotated_crosswalk_polygons_are_removed_but_lane_edge_remains(self):
        binary = np.zeros((180, 260), dtype=np.uint8)
        cv2.line(binary, (18, 0), (18, 179), 255, 5)

        stop_polygon = cv2.boxPoints(((145.0, 125.0), (150.0, 15.0), 12.0)).astype(np.int32)
        stripe_polygons = [
            cv2.boxPoints(((85.0 + index * 32.0, 72.0), (14.0, 48.0), 0.0)).astype(np.int32)
            for index in range(4)
        ]
        cv2.fillConvexPoly(binary, stop_polygon, 255)
        for polygon in stripe_polygons:
            cv2.fillConvexPoly(binary, polygon, 255)

        follower = line_cy.LaneFollower.__new__(line_cy.LaneFollower)
        cleaned = follower.suppress_crosswalk_regions(
            binary,
            {
                "stop_polygon": stop_polygon.tolist(),
                "stripe_polygons": [polygon.tolist() for polygon in stripe_polygons],
            },
        )

        self.assertEqual(int(cleaned[125, 145]), 0)
        self.assertEqual(int(cleaned[72, 85]), 0)
        self.assertEqual(int(cleaned[90, 18]), 255)

    def test_detector_output_removes_complete_crosswalk_group(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(binary, (25, 0), (25, 479), 255, 5)
        stop_polygon = cv2.boxPoints(((320.0, 345.0), (390.0, 20.0), 12.0)).astype(np.int32)
        cv2.fillConvexPoly(binary, stop_polygon, 255)
        stripe_centers = [(190, 225), (255, 225), (320, 225), (385, 225), (450, 225)]
        for center in stripe_centers:
            polygon = cv2.boxPoints(((float(center[0]), float(center[1])), (28.0, 100.0), 0.0)).astype(np.int32)
            cv2.fillConvexPoly(binary, polygon, 255)

        vision = line_cy.LineVision()
        result = vision.detect_stopline_before_crosswalk(binary)
        follower = line_cy.LaneFollower.__new__(line_cy.LaneFollower)
        cleaned = follower.suppress_crosswalk_regions(binary, result)

        self.assertTrue(result["candidate"])
        self.assertEqual(len(result["stripe_polygons"]), len(stripe_centers))
        self.assertEqual(int(cleaned[345, 320]), 0)
        for x, y in stripe_centers:
            self.assertEqual(int(cleaned[y, x]), 0)
        self.assertEqual(int(cleaned[120, 25]), 255)


class CrosswalkMisclassificationTests(unittest.TestCase):
    def test_thin_long_lane_fragments_do_not_form_crosswalk_stripes(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        for center in [(120.0, 240.0), (260.0, 238.0), (400.0, 242.0)]:
            fill_rotated_rect(binary, center, (9.0, 90.0), -58.0)

        result = line_cy.LineVision().detect_stopline_before_crosswalk(binary)

        self.assertFalse(result["candidate"])
        self.assertEqual(result["stripe_polygons"], [])

    def test_grouped_short_thick_bars_are_crosswalk_stripes(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        for x in [180, 245, 310, 375, 440]:
            fill_rotated_rect(binary, (float(x), 220.0), (32.0, 95.0), 4.0)

        result = line_cy.LineVision().detect_stopline_before_crosswalk(binary)

        self.assertFalse(result["candidate"])
        self.assertEqual(len(result["stripe_polygons"]), 5)

    def test_perspective_crosswalk_group_keeps_right_side_bars(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        bars = [
            ((220.0, 240.0), (32.0, 105.0), -8.0),
            ((290.0, 240.0), (35.0, 108.0), -6.0),
            ((360.0, 240.0), (38.0, 112.0), -4.0),
            ((465.0, 245.0), (70.0, 145.0), 4.0),
            ((550.0, 250.0), (75.0, 150.0), 9.0),
        ]
        for center, size, angle in bars:
            fill_rotated_rect(binary, center, size, angle)

        result = line_cy.LineVision().detect_stopline_before_crosswalk(binary)

        self.assertFalse(result["candidate"])
        self.assertEqual(len(result["stripe_polygons"]), len(bars))

    def test_lane_scan_ignores_sparse_crosswalk_bars_when_left_edge_is_continuous(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(binary, (65, 430), (250, 120), 255, 12)
        for center in [(440.0, 250.0), (520.0, 250.0), (600.0, 250.0)]:
            fill_rotated_rect(binary, center, (42.0, 120.0), 6.0)

        vision = line_cy.LineVision()
        kalman = cv2.KalmanFilter(2, 1)
        kalman.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
        kalman.measurementMatrix = np.array([[1, 0]], np.float32)
        kalman.processNoiseCov = np.eye(2, dtype=np.float32) * 1e-4
        kalman.measurementNoiseCov = np.array([[1]], np.float32) * 1e-1
        kalman.statePost = np.array([[320], [0]], np.float32)

        deviation, centers, failed_count, debug = vision.scan(
            binary, kalman, 320, 0, 360.0, "normal", None
        )

        self.assertLess(centers[-1][0], 360)
        self.assertEqual(debug["dominant"], "left_single")
        self.assertEqual(failed_count, 0)

    def test_diagonal_lane_with_stripes_is_not_a_stopline(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        fill_rotated_rect(binary, (250.0, 285.0), (450.0, 16.0), -35.0)
        for x in [235, 300, 365, 430, 495]:
            fill_rotated_rect(binary, (float(x), 210.0), (32.0, 92.0), -6.0)

        result = line_cy.LineVision().detect_stopline_before_crosswalk(binary)

        self.assertFalse(result["candidate"])
        self.assertIsNone(result["stop_polygon"])


class AlignmentControlTests(unittest.TestCase):
    def test_horizontal_stopline_requires_no_rotation(self):
        angular = line_cy.alignment_angular(0.0, 0.025, 0.08, 0.35, 1.0)
        self.assertEqual(angular, 0.0)

    def test_opposite_angles_produce_opposite_rotation(self):
        positive = line_cy.alignment_angular(8.0, 0.025, 0.08, 0.35, 1.0)
        negative = line_cy.alignment_angular(-8.0, 0.025, 0.08, 0.35, 1.0)
        self.assertAlmostEqual(positive, -negative)

    def test_rotation_obeys_minimum_and_maximum_limits(self):
        small = line_cy.alignment_angular(0.5, 0.025, 0.08, 0.35, 1.0)
        large = line_cy.alignment_angular(40.0, 0.025, 0.08, 0.35, 1.0)
        self.assertAlmostEqual(abs(small), 0.08)
        self.assertAlmostEqual(abs(large), 0.35)

    def test_direction_sign_can_be_reversed_on_real_vehicle(self):
        normal = line_cy.alignment_angular(8.0, 0.025, 0.08, 0.35, 1.0)
        reversed_direction = line_cy.alignment_angular(8.0, 0.025, 0.08, 0.35, -1.0)
        self.assertAlmostEqual(normal, -reversed_direction)


class CrosswalkStateTests(unittest.TestCase):
    def test_stable_candidate_starts_low_speed_approach_before_old_trigger(self):
        state = line_cy.crosswalk_next_state("FOLLOW_LINE", True, True, 0.50, 0.82, 0, 6)
        self.assertEqual(state, "APPROACH_CROSSWALK")

    def test_stopline_must_reach_bottom_before_alignment(self):
        far_state = line_cy.crosswalk_next_state("APPROACH_CROSSWALK", True, True, 0.75, 0.82, 0, 6)
        near_state = line_cy.crosswalk_next_state("APPROACH_CROSSWALK", True, True, 0.84, 0.82, 0, 6)
        self.assertEqual(far_state, "APPROACH_CROSSWALK")
        self.assertEqual(near_state, "ALIGN_STOPLINE")

    def test_lost_candidate_is_safe_in_each_phase(self):
        approaching = line_cy.crosswalk_next_state("APPROACH_CROSSWALK", False, False, 0.0, 0.82, 7, 6)
        aligning = line_cy.crosswalk_next_state("ALIGN_STOPLINE", False, False, 0.0, 0.82, 7, 6)
        self.assertEqual(approaching, "CROSSWALK_WAIT")
        self.assertEqual(aligning, "CROSSWALK_WAIT")

    def test_alignment_timeout_enters_wait_immediately(self):
        state = line_cy.crosswalk_next_state(
            "ALIGN_STOPLINE", True, True, 0.85, 0.82, 0, 6, timed_out=True
        )
        self.assertEqual(state, "CROSSWALK_WAIT")


class RuntimeSafetyTests(unittest.TestCase):
    def test_detect_only_disables_motion_even_when_dry_run_is_false(self):
        self.assertFalse(line_cy.motion_commands_enabled(False, True))
        self.assertFalse(line_cy.motion_commands_enabled(True, False))
        self.assertTrue(line_cy.motion_commands_enabled(False, False))

    def test_requested_approach_speed_caps_search_speed(self):
        self.assertAlmostEqual(line_cy.capped_speed(0.08, 0.06), 0.06)
        self.assertAlmostEqual(line_cy.capped_speed(0.08, None), 0.08)

    def test_camera_reader_does_not_return_same_frame_twice(self):
        reader = line_cy.CameraReader.__new__(line_cy.CameraReader)
        reader.lock = threading.Lock()
        reader.latest_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        reader.latest_ok = True
        reader.latest_seq = 1
        reader.last_read_seq = 0

        first_ok, _ = reader.read(timeout=0.01)
        second_ok, _ = reader.read(timeout=0.01)

        self.assertTrue(first_ok)
        self.assertFalse(second_ok)


if __name__ == "__main__":
    unittest.main(verbosity=2)
