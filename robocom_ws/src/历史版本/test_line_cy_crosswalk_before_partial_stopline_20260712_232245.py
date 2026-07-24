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

    def test_loose_stripes_are_removed_only_in_maneuver_memory_window(self):
        binary = np.zeros((180, 260), dtype=np.uint8)
        loose_polygon = cv2.boxPoints(((120.0, 88.0), (28.0, 85.0), 4.0)).astype(np.int32)
        cv2.fillConvexPoly(binary, loose_polygon, 255)

        follower = line_cy.LaneFollower.__new__(line_cy.LaneFollower)
        result = {"loose_stripe_polygons": [loose_polygon.tolist()], "stripe_polygons": []}
        normal_cleaned = follower.suppress_crosswalk_regions(binary, result)
        memory_cleaned = follower.suppress_crosswalk_regions(binary, result, include_loose=True)

        self.assertEqual(int(normal_cleaned[88, 120]), 255)
        self.assertEqual(int(memory_cleaned[88, 120]), 0)


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

    def test_standalone_near_stopline_tracks_alignment_without_triggering_entry(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        fill_rotated_rect(binary, (320.0, 370.0), (420.0, 20.0), -3.0)

        result = line_cy.LineVision().detect_stopline_before_crosswalk(binary)

        self.assertFalse(result["candidate"])
        self.assertIsNone(result["stop_polygon"])
        self.assertIsNotNone(result["tracking_stop_polygon"])
        self.assertGreaterEqual(result["tracking_confidence"], line_cy.CROSSWALK_TRACK_CONFIDENCE)
        self.assertGreater(result["tracking_stop_bottom_y"], 0)

    def test_steep_but_not_vertical_stopline_can_still_track_alignment(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        angle = max(1.0, line_cy.STOP_MAX_ANGLE_DEG - 5.0)
        fill_rotated_rect(binary, (320.0, 370.0), (420.0, 20.0), angle)

        result = line_cy.LineVision().detect_stopline_before_crosswalk(binary)

        self.assertFalse(result["candidate"])
        self.assertIsNone(result["stop_polygon"])
        self.assertIsNotNone(result["tracking_stop_polygon"])
        self.assertLessEqual(abs(result["tracking_stop_angle_deg"]), line_cy.STOP_MAX_ANGLE_DEG)

    def test_tracking_only_stopline_does_not_get_masked_in_normal_follow(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        polygon = fill_rotated_rect(binary, (320.0, 370.0), (420.0, 20.0), -3.0)

        result = line_cy.LineVision().detect_stopline_before_crosswalk(binary)
        follower = line_cy.LaneFollower.__new__(line_cy.LaneFollower)
        cleaned = follower.suppress_crosswalk_regions(binary, result)

        self.assertFalse(result["candidate"])
        self.assertIsNone(result["stop_polygon"])
        self.assertIsNotNone(result["tracking_stop_polygon"])
        self.assertEqual(int(cleaned[370, 320]), 255)
        self.assertTrue(np.any(cleaned[polygon[:, 1], polygon[:, 0]] == 255))

    def test_rejected_stop_group_does_not_mask_lane_like_horizontal_edge(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        edge_polygon = fill_rotated_rect(binary, (320.0, 110.0), (390.0, 20.0), 0.0)
        for x in [190, 255, 320, 385, 450]:
            fill_rotated_rect(binary, (float(x), 180.0), (28.0, 100.0), 0.0)

        result = line_cy.LineVision().detect_stopline_before_crosswalk(binary)
        follower = line_cy.LaneFollower.__new__(line_cy.LaneFollower)
        cleaned = follower.suppress_crosswalk_regions(binary, result)

        self.assertFalse(result["candidate"])
        self.assertIsNone(result["stop_polygon"])
        self.assertIsNotNone(result["tracking_stop_polygon"])
        self.assertEqual(int(cleaned[110, 320]), 255)
        self.assertTrue(np.any(cleaned[edge_polygon[:, 1], edge_polygon[:, 0]] == 255))

    def test_curved_outer_lane_edge_is_not_tracked_as_stopline(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        points = []
        for t in np.linspace(0.0, 1.0, 80):
            x = 80.0 + 420.0 * t
            y = 90.0 + 30.0 * (2.0 * t - 1.0) ** 2
            points.append([int(round(x)), int(round(y))])
        cv2.polylines(binary, [np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)], False, 255, 14)
        for x in [190, 255, 320, 385, 450]:
            fill_rotated_rect(binary, (float(x), 220.0), (28.0, 100.0), 0.0)

        result = line_cy.LineVision().detect_stopline_before_crosswalk(binary)

        self.assertFalse(result["candidate"])
        self.assertIsNone(result["stop_polygon"])
        self.assertIsNone(result.get("tracking_stop_polygon"))

    def test_lane_lock_prevents_previous_lane_edge_from_becoming_stopline(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        fill_rotated_rect(binary, (320.0, 260.0), (390.0, 20.0), 0.0)
        for x in [190, 255, 320, 385, 450]:
            fill_rotated_rect(binary, (float(x), 180.0), (28.0, 100.0), 0.0)
        lock_mask = np.zeros_like(binary)
        cv2.line(lock_mask, (125, 260), (515, 260), 255, 35)

        unlocked = line_cy.LineVision().detect_stopline_before_crosswalk(binary)
        locked = line_cy.LineVision().detect_stopline_before_crosswalk(binary, lock_mask)

        self.assertTrue(unlocked["candidate"])
        self.assertFalse(locked["candidate"])
        self.assertIsNone(locked["stop_polygon"])
        self.assertIsNone(locked.get("tracking_stop_polygon"))

    def test_stopline_crossing_two_locked_lane_edges_is_still_detected(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        fill_rotated_rect(binary, (320.0, 300.0), (390.0, 20.0), 0.0)
        for x in [190, 255, 320, 385, 450]:
            fill_rotated_rect(binary, (float(x), 210.0), (28.0, 100.0), 0.0)

        lock_mask = np.zeros_like(binary)
        cv2.line(lock_mask, (150, 430), (245, 100), 255, 70)
        cv2.line(lock_mask, (490, 430), (395, 100), 255, 70)

        result = line_cy.LineVision().detect_stopline_before_crosswalk(binary, lock_mask)

        self.assertTrue(result["candidate"])
        self.assertIsNotNone(result["stop_polygon"])

    def test_stopline_connected_to_lane_edge_is_extracted_below_crosswalk(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        for x in [220, 285, 350, 415, 480]:
            fill_rotated_rect(binary, (float(x), 175.0), (30.0, 92.0), 8.0)
        cv2.line(binary, (145, 285), (520, 345), 255, 18)
        cv2.line(binary, (145, 285), (25, 455), 255, 18)

        result = line_cy.LineVision().detect_stopline_before_crosswalk(binary)
        follower = line_cy.LaneFollower.__new__(line_cy.LaneFollower)
        cleaned = follower.suppress_crosswalk_regions(binary, result)

        self.assertTrue(result["candidate"])
        self.assertIsNotNone(result["stop_polygon"])
        self.assertEqual(int(cleaned[315, 330]), 0)

    def test_first_visible_stripe_with_connected_stopline_triggers_early_entry(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        stripe = fill_rotated_rect(binary, (360.0, 180.0), (32.0, 100.0), 8.0)
        cv2.line(binary, (135, 285), (525, 340), 255, 18)
        cv2.line(binary, (135, 285), (25, 455), 255, 18)

        result = line_cy.LineVision().detect_stopline_before_crosswalk(binary)

        self.assertTrue(result["candidate"])
        self.assertIsNotNone(result["stop_polygon"])
        self.assertEqual(len(result["stripe_polygons"]), 1)
        self.assertTrue(np.any(np.asarray(result["stripe_polygons"]) == stripe[0, 0]))

    def test_near_vertical_lane_edge_is_not_a_stopline(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        fill_rotated_rect(binary, (310.0, 300.0), (420.0, 18.0), 82.0)
        for x in [210, 275, 340, 405, 470]:
            fill_rotated_rect(binary, (float(x), 210.0), (32.0, 95.0), 4.0)

        result = line_cy.LineVision().detect_stopline_before_crosswalk(binary)

        self.assertFalse(result["candidate"])
        self.assertIsNone(result["stop_polygon"])

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

    def test_single_left_lane_keeps_side_after_crossing_image_center(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(binary, (95, 430), (470, 130), 255, 14)

        vision = line_cy.LineVision()
        kalman = cv2.KalmanFilter(2, 1)
        kalman.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
        kalman.measurementMatrix = np.array([[1, 0]], np.float32)
        kalman.processNoiseCov = np.eye(2, dtype=np.float32) * 1e-4
        kalman.measurementNoiseCov = np.array([[1]], np.float32) * 1e-1
        kalman.statePost = np.array([[320], [0]], np.float32)

        deviation, centers, failed_count, debug = vision.scan(
            binary, kalman, 320, 0, 280.0, "normal", None
        )

        self.assertEqual(debug["dominant"], "left_single")
        self.assertEqual(debug["right_single_rows"], 0)
        self.assertGreater(debug["raw_mid"], 320)
        self.assertEqual(failed_count, 0)

    def test_single_side_hint_prevents_center_crossing_from_flipping_side(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(binary, (360, 430), (500, 130), 255, 14)

        vision = line_cy.LineVision()
        kalman = cv2.KalmanFilter(2, 1)
        kalman.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
        kalman.measurementMatrix = np.array([[1, 0]], np.float32)
        kalman.processNoiseCov = np.eye(2, dtype=np.float32) * 1e-4
        kalman.measurementNoiseCov = np.array([[1]], np.float32) * 1e-1
        kalman.statePost = np.array([[320], [0]], np.float32)

        deviation, centers, failed_count, debug = vision.scan(
            binary, kalman, 320, 0, 260.0, "normal", "left"
        )

        self.assertEqual(debug["dominant"], "left_single")
        self.assertEqual(debug["right_single_rows"], 0)
        self.assertGreater(debug["raw_mid"], 430)

    def test_diagonal_lane_with_stripes_is_not_a_stopline(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        fill_rotated_rect(binary, (250.0, 285.0), (450.0, 16.0), -35.0)
        for x in [235, 300, 365, 430, 495]:
            fill_rotated_rect(binary, (float(x), 210.0), (32.0, 92.0), -6.0)

        result = line_cy.LineVision().detect_stopline_before_crosswalk(binary)

        self.assertFalse(result["candidate"])
        self.assertIsNone(result["stop_polygon"])

    def test_dual_lane_row_keeps_only_two_active_edges(self):
        vision = line_cy.LineVision()
        groups = [
            (40, 52, 46.0, 13),
            (205, 217, 211.0, 13),
            (398, 410, 404.0, 13),
        ]

        center, valid, ignored, kind, measured_width, left, right, ref_edge = vision.row_center(
            groups, 640, 360.0, 320.0, "normal", None
        )

        self.assertEqual(kind, "dual")
        self.assertEqual(valid, [groups[0], groups[2]])
        self.assertIn(groups[1], ignored)
        self.assertAlmostEqual(center, (groups[0][1] + groups[2][0]) * 0.5)

    def test_outer_lines_are_ignored_when_inner_lane_pair_exists(self):
        vision = line_cy.LineVision()
        left_outer = (20, 32, 26.0, 13)
        left_lane = (175, 187, 181.0, 13)
        right_lane = (515, 527, 521.0, 13)

        center, valid, ignored, kind, measured_width, left, right, ref_edge = vision.row_center(
            [left_outer, left_lane, right_lane], 640, 340.0, 320.0, "normal", None
        )

        self.assertEqual(kind, "dual")
        self.assertEqual(valid, [left_lane, right_lane])
        self.assertIn(left_outer, ignored)

        left_lane = (100, 112, 106.0, 13)
        right_lane = (440, 452, 446.0, 13)
        right_outer = (610, 622, 616.0, 13)
        center, valid, ignored, kind, measured_width, left, right, ref_edge = vision.row_center(
            [left_lane, right_lane, right_outer], 640, 340.0, 320.0, "normal", None
        )

        self.assertEqual(kind, "dual")
        self.assertEqual(valid, [left_lane, right_lane])
        self.assertIn(right_outer, ignored)

    def test_outer_frame_side_lane_pair_is_not_downgraded_to_virtual_line(self):
        vision = line_cy.LineVision()
        entries = [
            {"center": 360.0, "y": 360, "weight": 2.5, "kind": "dual", "left_edge": 195.0, "right_edge": 620.0,
             "single_left": None, "single_right": None},
            {"center": 345.0, "y": 310, "weight": 2.0, "kind": "dual", "left_edge": 205.0, "right_edge": 520.0,
             "single_left": None, "single_right": None},
            {"center": 365.0, "y": 260, "weight": 1.5, "kind": "dual", "left_edge": 215.0, "right_edge": 635.0,
             "single_left": None, "single_right": None},
        ]
        candidates = [(item["center"], item["y"], item["weight"], item["kind"]) for item in entries]

        fixed_candidates, fixed_entries = vision._fix_discontinuous_pairs(candidates, entries, 640, 320.0)

        self.assertTrue(all(item["kind"] == "dual" for item in fixed_entries))
        self.assertTrue(all(item[3] == "dual" for item in fixed_candidates))

    def test_left_only_mode_ignores_other_intersection_lines(self):
        vision = line_cy.LineVision()
        left_lane = (85, 99, 92.0, 15)
        branch_line = (290, 306, 298.0, 17)
        right_line = (520, 536, 528.0, 17)

        center, valid, ignored, kind, measured_width, left, right, ref_edge = vision.row_center(
            [left_lane, branch_line, right_line], 640, 330.0, 320.0, "left_only", None
        )

        self.assertEqual(kind, "left_ref_single")
        self.assertEqual(valid, [left_lane])
        self.assertIn(branch_line, ignored)
        self.assertIn(right_line, ignored)
        self.assertAlmostEqual(center, left_lane[1] + 330.0 * line_cy.SINGLE_CENTER_FACTOR)

    def test_right_only_mode_ignores_other_intersection_lines(self):
        vision = line_cy.LineVision()
        left_line = (70, 86, 78.0, 17)
        branch_line = (310, 326, 318.0, 17)
        right_lane = (545, 559, 552.0, 15)

        center, valid, ignored, kind, measured_width, left, right, ref_edge = vision.row_center(
            [left_line, branch_line, right_lane], 640, 330.0, 320.0, "right_only", None
        )

        self.assertEqual(kind, "right_ref_single")
        self.assertEqual(valid, [right_lane])
        self.assertIn(left_line, ignored)
        self.assertIn(branch_line, ignored)
        self.assertAlmostEqual(center, right_lane[0] - 330.0 * line_cy.SINGLE_CENTER_FACTOR)


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

    def test_pid_gain_changes_smoothly_at_curve_threshold(self):
        before = line_cy.blended_pid_gains(99, 0.0028, 0.0008, 0.03, 0.002, 100, 80)
        after = line_cy.blended_pid_gains(100, 0.0028, 0.0008, 0.03, 0.002, 100, 80)

        self.assertLess(abs(after[0] - before[0]), 0.001)
        self.assertGreater(after[0], before[0])

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
    def test_default_roi_is_expanded_for_intersection_view(self):
        self.assertLessEqual(line_cy.ROI_TOP_RATIO, 0.24)
        self.assertGreaterEqual(line_cy.ROI_BOTTOM_RATIO, 0.76)

    def test_straight_maneuver_uses_normal_follow_mode(self):
        follower = line_cy.LaneFollower.__new__(line_cy.LaneFollower)
        follower.left_turn_bias = 0.12
        follower.right_turn_bias = 0.12
        follower.straight_bias = 0.0

        self.assertEqual(follower.maneuver_mode("straight"), ("normal", 0.0))

    def test_left_and_right_maneuvers_keep_side_follow_modes(self):
        follower = line_cy.LaneFollower.__new__(line_cy.LaneFollower)
        follower.left_turn_bias = 0.12
        follower.right_turn_bias = 0.12
        follower.straight_bias = 0.0

        self.assertEqual(follower.maneuver_mode("left"), ("left", 0.12))
        self.assertEqual(follower.maneuver_mode("right"), ("right", -0.12))

    def test_initial_intersection_segment_should_not_force_right_follow(self):
        follower = line_cy.LaneFollower.__new__(line_cy.LaneFollower)
        follower.enter_intersection_straight_time = 0.6
        follower.straight_bias = 0.0
        mode, bias, allow_single = follower.maneuver_follow_choice("left", 0.2, "left", 0.12)

        self.assertEqual(mode, "normal")
        self.assertEqual(bias, 0.0)
        self.assertFalse(allow_single)

    def test_strict_maneuver_locks_commanded_turn_side_after_entry_segment(self):
        follower = line_cy.LaneFollower.__new__(line_cy.LaneFollower)
        follower.enter_intersection_straight_time = 0.6
        follower.maneuver_straight_follow_side = "auto"

        self.assertEqual(follower.locked_maneuver_mode("left", 0.7, None, "left"), "left_only")
        self.assertEqual(follower.locked_maneuver_mode("right", 0.7, None, "right"), "right_only")
        self.assertEqual(follower.locked_maneuver_mode("left", 0.2, None, "left"), "normal")

    def test_strict_maneuver_stays_normal_until_track_is_acquired(self):
        follower = line_cy.LaneFollower.__new__(line_cy.LaneFollower)
        follower.enter_intersection_straight_time = 0.6

        self.assertEqual(follower.locked_maneuver_mode("straight", 0.7, None, None), "normal")

    def test_straight_maneuver_can_choose_visible_locked_side(self):
        follower = line_cy.LaneFollower.__new__(line_cy.LaneFollower)
        follower.vision = line_cy.LineVision()
        follower.maneuver_straight_follow_side = "auto"
        binary = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(binary, (520, 430), (600, 130), 255, 14)

        self.assertEqual(follower.locked_maneuver_side("straight", binary), "right")

        follower.maneuver_straight_follow_side = "left"
        self.assertEqual(follower.locked_maneuver_side("straight", binary), "left")

    def test_maneuver_track_locks_only_the_remaining_continuous_side(self):
        follower = line_cy.LaneFollower.__new__(line_cy.LaneFollower)
        follower.vision = line_cy.LineVision()
        follower.maneuver_strict_band_ratio = 0.14
        binary = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(binary, (520, 430), (600, 130), 255, 14)
        cv2.line(binary, (80, 430), (80, 360), 255, 14)

        side, points = follower.acquire_maneuver_track(binary)

        self.assertEqual(side, "right")
        self.assertGreaterEqual(len(points), 3)

    def test_maneuver_track_mask_removes_everything_outside_locked_band(self):
        follower = line_cy.LaneFollower.__new__(line_cy.LaneFollower)
        follower.vision = line_cy.LineVision()
        follower.maneuver_strict_band_ratio = 0.14
        clean = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(clean, (520, 430), (600, 130), 255, 14)
        side, points = follower.acquire_maneuver_track(clean)
        binary = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(binary, (520, 430), (600, 130), 255, 14)
        cv2.line(binary, (80, 420), (300, 120), 255, 18)
        cv2.line(binary, (100, 250), (500, 250), 255, 18)

        filtered, _ = follower.filter_maneuver_track(binary, side, points)

        self.assertEqual(side, "right")
        self.assertEqual(int(filtered[250, 200]), 0)
        self.assertGreater(np.count_nonzero(filtered[:, 480:]), 0)

    def test_maneuver_track_waits_when_both_sides_are_continuous(self):
        follower = line_cy.LaneFollower.__new__(line_cy.LaneFollower)
        follower.vision = line_cy.LineVision()
        follower.maneuver_strict_band_ratio = 0.14
        binary = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(binary, (120, 430), (40, 130), 255, 14)
        cv2.line(binary, (520, 430), (600, 130), 255, 14)

        side, points = follower.acquire_maneuver_track(binary)

        self.assertIsNone(side)
        self.assertEqual(points, [])

    def test_maneuver_track_requires_three_same_side_frames_to_lock(self):
        pending, hits, locked = line_cy.confirm_maneuver_side(None, 0, None, "right", 3)
        pending, hits, locked = line_cy.confirm_maneuver_side(pending, hits, locked, "right", 3)
        self.assertIsNone(locked)

        pending, hits, locked = line_cy.confirm_maneuver_side(pending, hits, locked, "right", 3)
        self.assertEqual(locked, "right")

        _, _, still_locked = line_cy.confirm_maneuver_side("left", 9, locked, "left", 3)
        self.assertEqual(still_locked, "right")

    def test_exit_crosswalk_counts_only_after_entry_has_cleared(self):
        follower = line_cy.LaneFollower.__new__(line_cy.LaneFollower)

        self.assertEqual(follower.update_exit_crosswalk_hits(False, 0, True), 0)
        self.assertEqual(follower.update_exit_crosswalk_hits(True, 0, True), 1)
        self.assertEqual(follower.update_exit_crosswalk_hits(True, 1, True), 2)
        self.assertEqual(follower.update_exit_crosswalk_hits(True, 2, False), 1)

    def test_exit_crosswalk_requires_a_near_horizontal_bar(self):
        follower = line_cy.LaneFollower.__new__(line_cy.LaneFollower)
        follower.crosswalk_track_confidence = 0.52
        follower.exit_crosswalk_y_ratio = 0.68
        far = {
            "candidate": True,
            "stop_bottom_y": 220,
            "tracking_confidence": 0.9,
            "tracking_stop_bottom_y": 220,
        }
        near = dict(far, stop_bottom_y=350)

        self.assertFalse(follower.maneuver_exit_visible(far, 480))
        self.assertTrue(follower.maneuver_exit_visible(near, 480))

    def test_crosswalk_mask_result_combines_memory_and_current_loose_stripes(self):
        follower = line_cy.LaneFollower.__new__(line_cy.LaneFollower)
        current = {"stripe_polygons": [], "loose_stripe_polygons": [[[1, 1], [2, 1], [2, 2], [1, 2]]]}
        remembered = {"stripe_polygons": [[[3, 3], [4, 3], [4, 4], [3, 4]]], "loose_stripe_polygons": []}

        merged = follower.crosswalk_mask_result(current, remembered)

        self.assertEqual(len(merged["stripe_polygons"]), 1)
        self.assertEqual(len(merged["loose_stripe_polygons"]), 1)

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
