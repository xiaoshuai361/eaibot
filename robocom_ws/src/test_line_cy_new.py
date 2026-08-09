#!/usr/bin/env python3
# coding=utf-8

import importlib.util
import os
import shutil
import sys
import tempfile
import threading
import types
import unittest

import cv2
import numpy as np


def load_module(filename="line_cy_new.py", module_name="line_cy_new_under_test"):
    rospy = types.ModuleType("rospy")
    rospy.is_shutdown = lambda: False
    sys.modules.setdefault("rospy", rospy)
    geometry_msgs = types.ModuleType("geometry_msgs")
    geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msgs_msg.Twist = type("Twist", (), {})
    geometry_msgs.msg = geometry_msgs_msg
    sys.modules.setdefault("geometry_msgs", geometry_msgs)
    sys.modules.setdefault("geometry_msgs.msg", geometry_msgs_msg)
    path = os.path.join(os.path.dirname(__file__), filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


line_new = load_module()
line_task = load_module("line_cy_task.py", "line_cy_task_under_test")


def rotated_rect(binary, center, size, angle):
    polygon = cv2.boxPoints((center, size, angle)).astype(np.int32)
    cv2.fillConvexPoly(binary, polygon, 255)
    return polygon


class LaneGeometryTests(unittest.TestCase):
    def test_lane_center_weights_near_rows_more_than_far_rows(self):
        detector = line_new.LaneDetector(
            roi_top=0.2, roi_bottom=0.9, center_near_weight=3.0
        )
        points = [(100.0, 100), (300.0, 400)]

        weighted = detector._weighted_center_x(points, 480)

        self.assertGreater(weighted, 200.0)
        self.assertLess(weighted, 300.0)

    def test_lane_center_weight_does_not_move_straight_center(self):
        detector = line_new.LaneDetector(center_near_weight=3.0)
        points = [(320.0, 100), (320.0, 250), (320.0, 430)]

        self.assertEqual(detector._weighted_center_x(points, 480), 320.0)

    def test_positive_follow_bias_moves_control_target_right(self):
        self.assertEqual(line_new.control_target_x(300.0, 35.0), 335.0)

    def test_small_error_uses_normal_pd_in_both_directions(self):
        limit = line_new.LARGE_ERROR_THRESHOLD_PIXELS

        self.assertEqual(line_new.pd_gains(limit - 1),
                         (line_new.KP, line_new.KD, "small"))
        self.assertEqual(line_new.pd_gains(-limit + 1),
                         (line_new.KP, line_new.KD, "small"))

    def test_large_error_uses_fast_pd_in_both_directions(self):
        limit = line_new.LARGE_ERROR_THRESHOLD_PIXELS

        self.assertEqual(line_new.pd_gains(limit),
                         (line_new.LARGE_ERROR_KP,
                          line_new.LARGE_ERROR_KD, "large"))
        self.assertEqual(line_new.pd_gains(-limit),
                         (line_new.LARGE_ERROR_KP,
                          line_new.LARGE_ERROR_KD, "large"))

    def test_control_passes_small_error_pd_to_pid(self):
        follower = line_new.LaneFollower.__new__(line_new.LaneFollower)
        calls = []
        follower.pid = types.SimpleNamespace(
            update=lambda deviation, kp, kd: calls.append(
                (deviation, kp, kd)
            ) or 0.0
        )
        follower.last_angular = 0.0
        follower.last_control_target = None
        follower.publish = lambda speed, angular: None

        follower._control(280.0, 640, 0.16)

        self.assertEqual(calls[-1], (-40.0, line_new.KP, line_new.KD))

    def test_control_passes_large_error_pd_to_pid(self):
        follower = line_new.LaneFollower.__new__(line_new.LaneFollower)
        calls = []
        follower.pid = types.SimpleNamespace(
            update=lambda deviation, kp, kd: calls.append(
                (deviation, kp, kd)
            ) or 0.0
        )
        follower.last_angular = 0.0
        follower.last_control_target = None
        follower.publish = lambda speed, angular: None

        follower._control(
            320.0 + line_new.LARGE_ERROR_THRESHOLD_PIXELS, 640, 0.16
        )

        self.assertEqual(calls[-1],
                         (line_new.LARGE_ERROR_THRESHOLD_PIXELS,
                          line_new.LARGE_ERROR_KP,
                          line_new.LARGE_ERROR_KD))

    def test_center_out_segments_ignore_external_black_objects(self):
        row = np.zeros(640, dtype=np.uint8)
        row[20:45] = row[170:185] = row[455:470] = row[590:620] = 255

        left, right = line_new.center_out_segments(row, 320, min_width=4)

        self.assertEqual(left, (170, 184))
        self.assertEqual(right, (455, 469))

    def test_dual_lane_center_uses_inner_edges(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(binary, (170, 430), (220, 120), 255, 13)
        cv2.line(binary, (470, 430), (420, 120), 255, 13)

        result = line_new.LaneDetector().observe(binary, 300.0)

        self.assertGreaterEqual(result.dual_rows, 4)
        self.assertAlmostEqual(result.center_x, 320.0, delta=12.0)
        self.assertTrue(result.valid)

    def test_single_right_lane_reconstructs_center_to_its_left(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(binary, (500, 430), (440, 120), 255, 13)

        result = line_new.LaneDetector().observe(binary, 300.0, follow_side="right")

        self.assertTrue(result.valid)
        self.assertEqual(result.follow_side, "right")
        self.assertLess(result.center_x, 400.0)
        self.assertEqual(len(result.center_points), len(result.right_points))
        self.assertEqual(len(result.virtual_left_points), len(result.right_points))
        self.assertTrue(all(vx < rx for (vx, _), (rx, _) in zip(
            result.virtual_left_points, result.right_points
        )))

    def test_fill_width_can_be_tuned_independently_from_learned_lane_width(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(binary, (500, 430), (440, 120), 255, 13)
        detector = line_new.LaneDetector(fill_width=240.0)

        result = detector.observe(binary, lane_width=320.0, follow_side="right")
        mean_right = np.mean([x for x, _ in result.right_points])

        self.assertAlmostEqual(result.center_x, mean_right - 120.0, delta=0.1)
        for (center_x, y), (right_x, right_y) in zip(result.center_points,
                                                     result.right_points):
            self.assertEqual(y, right_y)
            self.assertAlmostEqual(center_x, right_x - 120.0, delta=0.1)

    def test_left_and_right_fill_widths_are_independent(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(binary, (140, 430), (200, 120), 255, 13)
        cv2.line(binary, (500, 430), (440, 120), 255, 13)

        for module in (line_new, line_task):
            detector = module.LaneDetector(
                fill_width=300.0,
                left_fill_width=360.0,
                right_fill_width=240.0,
            )
            left = detector.observe(binary, 320.0, follow_side="left")
            right = detector.observe(binary, 320.0, follow_side="right")
            mean_left = np.mean([x for x, _ in left.left_points])
            mean_right = np.mean([x for x, _ in right.right_points])

            self.assertAlmostEqual(left.center_x, mean_left + 180.0, delta=0.1)
            self.assertAlmostEqual(right.center_x, mean_right - 120.0, delta=0.1)

    def test_single_left_lane_draw_data_contains_virtual_right_edge_and_centers(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(binary, (140, 430), (220, 120), 255, 13)

        result = line_new.LaneDetector(fill_width=300.0).observe(
            binary, lane_width=320.0, follow_side="left"
        )

        self.assertEqual(len(result.virtual_right_points), len(result.left_points))
        self.assertEqual(len(result.center_points), len(result.left_points))
        for (left_x, y), (virtual_x, virtual_y), (center_x, center_y) in zip(
                result.left_points, result.virtual_right_points, result.center_points):
            self.assertEqual((y, y), (virtual_y, center_y))
            self.assertAlmostEqual(virtual_x, left_x + 300.0, delta=0.1)
            self.assertAlmostEqual(center_x, left_x + 150.0, delta=0.1)

    def test_mixed_dual_and_single_rows_all_produce_center_points(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        rows = np.linspace(int(480 * line_new.ROI_BOTTOM),
                           int(480 * line_new.ROI_TOP),
                           line_new.SCAN_ROWS).astype(np.int32)
        for index, y in enumerate(rows):
            cv2.rectangle(binary, (120 + index * 3, y - 3),
                          (132 + index * 3, y + 3), 255, -1)
            if index < 3:
                cv2.rectangle(binary, (490 - index * 3, y - 3),
                              (502 - index * 3, y + 3), 255, -1)

        result = line_new.LaneDetector(fill_width=360.0).observe(
            binary, lane_width=360.0
        )

        self.assertEqual(result.dual_rows, 3)
        self.assertEqual(len(result.center_points), line_new.SCAN_ROWS)
        self.assertEqual(len(result.virtual_right_points), line_new.SCAN_ROWS - 3)

    def test_invalid_dual_width_uses_continuous_single_side_center(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        rows = np.linspace(int(480 * line_new.ROI_BOTTOM),
                           int(480 * line_new.ROI_TOP),
                           line_new.SCAN_ROWS).astype(np.int32)
        for y in rows:
            cv2.rectangle(binary, (2, y - 3), (14, y + 3), 255, -1)
            cv2.rectangle(binary, (625, y - 3), (637, y + 3), 255, -1)

        result = line_new.LaneDetector(fill_width=300.0).observe(
            binary, lane_width=300.0, center_hint=245.0
        )

        self.assertEqual(result.dual_rows, 0)
        self.assertEqual(len(result.center_points), line_new.SCAN_ROWS)
        self.assertTrue(all(center_x < 300 for center_x, _ in result.center_points))

    def test_virtual_edge_is_clipped_only_for_display(self):
        points = [(700.0, 200), (-80.0, 300), (420.0, 400)]

        clipped = line_new.clip_points_for_display(points, width=640)

        self.assertEqual(clipped, [(639, 200), (0, 300), (420, 400)])

    def test_left_turn_uses_left_edge_and_right_turn_uses_right_edge(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(binary, (140, 430), (210, 120), 255, 13)
        cv2.line(binary, (500, 430), (430, 120), 255, 13)
        detector = line_new.LaneDetector()

        left_turn = detector.observe(binary, 300.0, follow_side="left")
        right_turn = detector.observe(binary, 300.0, follow_side="right")

        self.assertEqual(left_turn.follow_side, "left")
        self.assertEqual(right_turn.follow_side, "right")
        self.assertEqual(left_turn.right_points, [])
        self.assertEqual(right_turn.left_points, [])
        self.assertGreater(left_turn.center_x, 250.0)
        self.assertLess(right_turn.center_x, 390.0)

    def test_left_turn_uses_learned_perspective_when_filling_right_edge(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(binary, (0, 441), (180, 96), 255, 13)
        detector = line_new.LaneDetector(fill_width=620.0)
        # At the lower ROI the half-width is 310 px; toward the horizon it narrows.
        left_to_center = (230.0 / 345.0, 16.0)

        fixed = detector.observe(binary, 620.0, follow_side="left")
        perspective = detector.observe(
            binary, 620.0, follow_side="left",
            side_center_transform=left_to_center,
        )

        self.assertGreater(fixed.center_x, 320.0)
        self.assertLess(perspective.center_x, 320.0)
        self.assertEqual(perspective.follow_side, "left")
        self.assertTrue(all(vx > lx for (vx, _), (lx, _) in zip(
            perspective.virtual_right_points, perspective.left_points
        )))

    def test_right_turn_uses_learned_perspective_when_filling_left_edge(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(binary, (639, 441), (459, 96), 255, 13)
        detector = line_new.LaneDetector(fill_width=620.0)
        right_to_center = (-230.0 / 345.0, -16.0)

        fixed = detector.observe(binary, 620.0, follow_side="right")
        perspective = detector.observe(
            binary, 620.0, follow_side="right",
            side_center_transform=right_to_center,
        )

        self.assertLess(fixed.center_x, 320.0)
        self.assertGreater(perspective.center_x, 320.0)
        self.assertEqual(perspective.follow_side, "right")
        self.assertTrue(all(vx < rx for (vx, _), (rx, _) in zip(
            perspective.virtual_left_points, perspective.right_points
        )))

class CrosswalkTests(unittest.TestCase):
    def test_near_large_stripes_are_kept_as_candidates(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        for x in [220, 310, 400]:
            rotated_rect(binary, (float(x), 315.0), (45.0, 250.0), 8.0)

        candidates = line_new.CrosswalkDetector()._stripe_candidates(binary)

        self.assertGreaterEqual(len(candidates), 3)

    def test_stripe_group_keeps_gradual_perspective_size_change(self):
        candidates = []
        for index, (x, long_side, short_side) in enumerate([
                (250, 35, 12), (310, 55, 17), (375, 85, 25),
                (445, 130, 37), (520, 195, 54)]):
            candidates.append({
                "center": (float(x), 160.0 + index * 8.0),
                "long": float(long_side), "short": float(short_side),
                "bottom": int(190 + index * 10), "angle": 82.0 - index * 2.0,
                "polygon": [],
            })

        group = line_new.CrosswalkDetector()._stripe_group(
            candidates, (480, 640)
        )

        self.assertEqual(len(group), 5)

    def test_strong_stripe_group_recovers_bar_from_wrong_lane_crop(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        for x in [370, 425, 480, 535, 590]:
            rotated_rect(binary, (float(x), 210.0), (32.0, 105.0), 5.0)
        cv2.line(binary, (330, 310), (630, 325), 255, 18)
        wrong_right_lane = [(350, 430), (355, 360), (360, 290),
                            (365, 220), (370, 150)]

        result = line_new.CrosswalkDetector().detect(
            binary, lane_points=[[], wrong_right_lane]
        )

        self.assertTrue(result.candidate)
        self.assertGreaterEqual(len(result.stripe_polygons), 3)
        self.assertIsNotNone(result.stop_polygon)

    def test_elements_outside_single_left_lane_are_ignored(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        for x in [70, 140, 210]:
            rotated_rect(binary, (float(x), 180.0), (32.0, 100.0), 5.0)
        cv2.line(binary, (20, 300), (250, 300), 255, 18)
        left = [(280, 430), (280, 360), (280, 290), (280, 220), (280, 150)]

        result = line_new.CrosswalkDetector().detect(
            binary, lane_points=[left, []]
        )

        self.assertFalse(result.candidate)
        self.assertEqual(result.stripe_polygons, [])
        self.assertIsNone(result.stop_polygon)

    def test_elements_inside_single_left_lane_are_kept(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        for x in [360, 430, 500]:
            rotated_rect(binary, (float(x), 180.0), (32.0, 100.0), 5.0)
        cv2.line(binary, (300, 300), (620, 300), 255, 18)
        left = [(280, 430), (280, 360), (280, 290), (280, 220), (280, 150)]

        result = line_new.CrosswalkDetector().detect(
            binary, lane_points=[left, []]
        )

        self.assertTrue(result.candidate)
        self.assertEqual(len(result.stripe_polygons), 3)
        self.assertIsNotNone(result.stop_polygon)

    def test_hough_bar_polygon_is_centered_on_actual_thick_bar(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(binary, (100, 300), (540, 320), 255, 20)
        detector = line_new.CrosswalkDetector()

        polygon = np.asarray(
            detector._bar_polygon(binary, (105, 291, 535, 311)), dtype=np.float32
        )

        self.assertAlmostEqual(float(np.mean(polygon[:, 1])), 310.0, delta=2.5)
        self.assertGreater(float(np.max(polygon[:, 1]) - np.min(polygon[:, 1])), 18.0)

    def test_collinear_hough_fragments_are_merged_into_full_stop_bar(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(binary, (80, 310), (270, 320), 255, 18)
        cv2.line(binary, (335, 323), (560, 335), 255, 18)
        detector = line_new.CrosswalkDetector()

        bars = detector._hough_bars(binary, stripes=[])

        self.assertTrue(bars)
        self.assertGreater(max(bar["length"] for bar in bars), 450.0)

    def test_bar_matches_stripes_at_their_top_ends(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(binary, (80, 200), (560, 200), 255, 16)
        stripes = []
        for x in (220.0, 380.0):
            polygon = np.rint(cv2.boxPoints(
                ((x, 270.0), (30.0, 150.0), 0.0)
            )).astype(np.int32)
            stripes.append({
                "center": (x, 270.0),
                "bottom": int(np.max(polygon[:, 1])),
                "angle": 90.0,
                "polygon": polygon.tolist(),
            })

        bars = line_new.CrosswalkDetector()._hough_bars(binary, stripes)

        self.assertTrue(bars)
        self.assertGreaterEqual(
            max(len(bar["matched"]) for bar in bars), 2
        )

    def test_strong_crosswalk_bar_survives_contaminated_lane_model(self):
        detector = line_new.CrosswalkDetector()
        angle = 30.0
        center_x, center_y = 320.0, 240.0
        slope = 1.0 / np.tan(np.radians(angle))
        same_line = line_new.LineModel(
            slope, center_x - slope * center_y, 6, 0.0
        )
        polygon = np.rint(cv2.boxPoints(
            ((center_x, center_y), (360.0, 18.0), angle)
        )).astype(np.int32)
        strong_bar = {
            "center": (center_x, center_y),
            "angle": angle,
            "polygon": polygon.tolist(),
            "matched": [
                {"center": (220.0, 260.0)},
                {"center": (420.0, 260.0)},
            ],
        }
        plain_boundary = dict(strong_bar, matched=[])

        self.assertTrue(detector._bar_matches_lane(
            strong_bar, [same_line], 640
        ))
        self.assertFalse(detector._bar_matches_lane(
            strong_bar, [same_line], 640,
            allow_strong_override=True,
        ))
        self.assertTrue(detector._bar_matches_lane(
            plain_boundary, [same_line], 640,
            allow_strong_override=True,
        ))

    def test_detached_stop_bar_wins_over_stripe_end_edge(self):
        detector = line_new.CrosswalkDetector()
        stripes = []
        for x in (220.0, 420.0):
            polygon = np.rint(cv2.boxPoints(
                ((x, 180.0), (30.0, 100.0), 0.0)
            )).astype(np.int32)
            stripes.append({
                "center": (x, 180.0), "short": 30.0,
                "angle": 90.0, "polygon": polygon.tolist(),
            })

        def bar_at(y):
            polygon = np.rint(cv2.boxPoints(
                ((320.0, float(y)), (360.0, 18.0), 0.0)
            )).astype(np.int32)
            return {
                "center": (320.0, float(y)), "angle": 0.0,
                "length": 360.0, "bottom": int(np.max(polygon[:, 1])),
                "polygon": polygon.tolist(), "matched": stripes,
            }

        stripe_end_edge = bar_at(230.0)
        detached_stop_bar = bar_at(300.0)
        edge_lane = line_new.LineModel(100.0, 320.0 - 100.0 * 230.0, 6, 0.0)
        stop_lane = line_new.LineModel(100.0, 320.0 - 100.0 * 300.0, 6, 0.0)

        self.assertTrue(detector._bar_matches_lane(
            stripe_end_edge, [edge_lane], 640,
        ))
        self.assertFalse(detector._bar_matches_lane(
            detached_stop_bar, [stop_lane], 640,
        ))

    def test_detector_recovers_detached_stop_bar_from_lane_points(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        for x, y in [(220, 155), (290, 175), (360, 195), (430, 215)]:
            rotated_rect(binary, (float(x), float(y)), (32.0, 105.0), 15.0)
        cv2.line(binary, (120, 280), (560, 470), 255, 18)
        lane_points = [(int(x), int(y)) for x, y in zip(
            np.linspace(120, 560, 9), np.linspace(280, 470, 9)
        )]

        result = line_new.CrosswalkDetector().detect(
            binary, lane_points=[lane_points],
        )

        self.assertTrue(result.candidate)
        self.assertIsNotNone(result.stop_polygon)
        self.assertGreater(np.mean(np.asarray(result.stop_polygon)[:, 1]), 300.0)

    def test_fewer_than_required_stripes_do_not_confirm_stop_bar(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        stripe_count = max(0, line_new.STRIPE_STRONG_COUNT - 1)
        for x in [360, 440][:stripe_count]:
            rotated_rect(binary, (float(x), 180.0), (32.0, 100.0), 5.0)
        cv2.line(binary, (260, 300), (560, 310), 255, 18)

        result = line_new.CrosswalkDetector().detect(binary)

        self.assertFalse(result.candidate)
        self.assertIsNone(result.stop_polygon)
        self.assertIsNotNone(result.tracking_polygon)

    def test_stable_bar_without_stripes_is_confirmed_over_time(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(binary, (160, 300), (500, 315), 255, 18)
        detector = line_new.CrosswalkDetector()

        results = [detector.detect(binary) for _ in range(
            line_new.BAR_ONLY_STABLE_FRAMES
        )]

        self.assertTrue(all(not result.candidate for result in results[:-1]))
        self.assertTrue(results[-1].candidate)
        self.assertIsNotNone(results[-1].stop_polygon)

    def test_diagonal_line_without_stripes_never_confirms_stop_bar(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(binary, (90, 130), (550, 395), 255, 18)
        detector = line_new.CrosswalkDetector()

        results = [detector.detect(binary) for _ in range(
            line_new.BAR_ONLY_STABLE_FRAMES + line_new.STOP_STABLE_FRAMES
        )]

        self.assertTrue(all(not result.candidate for result in results))
        self.assertTrue(all(result.stop_polygon is None for result in results))

    def test_bar_only_confirmation_resets_when_position_jumps(self):
        first = np.zeros((480, 640), dtype=np.uint8)
        second = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(first, (160, 210), (500, 225), 255, 18)
        cv2.line(second, (160, 350), (500, 365), 255, 18)
        detector = line_new.CrosswalkDetector()
        for _ in range(line_new.BAR_ONLY_STABLE_FRAMES - 1):
            self.assertFalse(detector.detect(first).candidate)

        result = detector.detect(second)

        self.assertFalse(result.candidate)

    def test_bar_tracking_smooths_position_but_keeps_current_length(self):
        detector = line_new.CrosswalkDetector()
        previous = {
            "center": (320.0, 300.0), "length": 260.0, "angle": 2.0,
            "polygon": cv2.boxPoints(((320.0, 300.0), (260.0, 18.0), 2.0)).tolist(),
            "bottom": 312, "matched": [],
        }
        current = {
            "center": (330.0, 312.0), "length": 360.0, "angle": 6.0,
            "polygon": cv2.boxPoints(((330.0, 312.0), (360.0, 18.0), 6.0)).tolist(),
            "bottom": 340, "matched": [],
        }

        smoothed = detector._smooth_bar(current, previous)

        self.assertGreater(smoothed["center"][1], previous["center"][1])
        self.assertLess(smoothed["center"][1], current["center"][1])
        self.assertLess(current["center"][1] - smoothed["center"][1],
                        smoothed["center"][1] - previous["center"][1])
        self.assertAlmostEqual(smoothed["length"], current["length"])

    def test_confirmed_bar_stays_candidate_when_stripes_temporarily_disappear(self):
        confirmed = np.zeros((480, 640), dtype=np.uint8)
        for x in [260, 360]:
            rotated_rect(confirmed, (float(x), 180.0), (32.0, 100.0), 5.0)
        cv2.line(confirmed, (150, 300), (500, 315), 255, 18)
        bar_only = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(bar_only, (150, 320), (500, 335), 255, 18)
        detector = line_new.CrosswalkDetector()

        first = detector.detect(confirmed)
        self.assertTrue(first.candidate)
        self.assertTrue(detector.lock_current_bar())
        tracked = detector.detect(bar_only)

        self.assertTrue(tracked.candidate)
        self.assertIsNotNone(tracked.stop_polygon)
        self.assertEqual(len(tracked.stripe_polygons), 0)

    def test_locked_bar_does_not_jump_to_distant_hough_line(self):
        target = np.zeros((480, 640), dtype=np.uint8)
        for x in [260, 360]:
            rotated_rect(target, (float(x), 160.0), (32.0, 90.0), 5.0)
        cv2.line(target, (150, 270), (500, 285), 255, 18)
        distractor = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(distractor, (80, 410), (560, 410), 255, 18)
        detector = line_new.CrosswalkDetector()

        first = detector.detect(target)
        self.assertTrue(first.candidate)
        self.assertTrue(detector.lock_current_bar())
        tracked = detector.detect(distractor)

        self.assertTrue(tracked.candidate)
        self.assertLess(tracked.tracking_bottom, 350)

    def test_locked_bar_cannot_drift_away_from_original_anchor(self):
        detector = line_new.CrosswalkDetector()
        anchor = {
            "center": (320.0, 410.0), "length": 360.0, "angle": 0.0,
            "polygon": cv2.boxPoints(
                ((320.0, 410.0), (360.0, 18.0), 0.0)
            ).tolist(),
            "bottom": 419, "matched": [],
        }
        detector.last_bar = anchor
        self.assertTrue(detector.lock_current_bar())
        # Simulate a tracker that walked toward a different line over several frames.
        detector.last_bar = dict(anchor, center=(320.0, 300.0), bottom=309)
        distractor = dict(anchor, center=(320.0, 260.0), bottom=269)

        selected = detector._select_bar([distractor], (480, 640))

        self.assertIsNone(selected)

    def test_partial_stop_bar_below_three_stripes_is_detected(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        for x in [410, 470, 530]:
            rotated_rect(binary, (float(x), 190.0), (34.0, 105.0), 8.0)
        cv2.line(binary, (350, 275), (635, 439), 255, 18)

        result = line_new.CrosswalkDetector().detect(binary)

        self.assertTrue(result.candidate)
        self.assertIsNotNone(result.stop_polygon)
        self.assertEqual(len(result.stripe_polygons), 3)

    def test_stop_bar_above_stripe_tops_is_detected(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        for x in [250, 330, 410]:
            rotated_rect(binary, (float(x), 310.0), (32.0, 100.0), 0.0)
        cv2.line(binary, (150, 225), (500, 225), 255, 18)

        result = line_new.CrosswalkDetector().detect(binary)

        self.assertTrue(result.candidate)
        self.assertIsNotNone(result.stop_polygon)
        self.assertEqual(len(result.stripe_polygons), 3)

    def test_side_line_near_crosswalk_cannot_confirm_stop_bar(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        for x in [500, 555, 610]:
            rotated_rect(binary, (float(x), 310.0), (32.0, 100.0), 0.0)
        cv2.line(binary, (455, 225), (639, 205), 255, 18)
        detector = line_new.CrosswalkDetector()

        results = [detector.detect(binary) for _ in range(4)]

        self.assertTrue(any(result.tracking_polygon is not None
                            for result in results))
        self.assertTrue(all(not result.candidate for result in results))
        self.assertTrue(all(result.stop_polygon is None for result in results))

    def test_side_bar_with_one_stripe_is_available_as_exit_only_evidence(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        # Near the image edge, perspective can make the only visible zebra mark
        # horizontal, so it will not geometrically match the stop bar.
        rotated_rect(binary, (570.0, 220.0), (100.0, 32.0), 0.0)
        cv2.line(binary, (500, 320), (639, 320), 255, 18)

        result = line_new.CrosswalkDetector().detect(binary)
        visible, bottom = line_new.exit_bar_evidence(result)

        self.assertFalse(result.candidate)
        self.assertTrue(visible)
        self.assertGreater(bottom, 320)

    def test_unmatched_tracking_line_is_not_exit_evidence_without_stripes(self):
        result = line_new.CrosswalkResult()
        result.tracking_polygon = [[500, 310], [639, 310],
                                   [639, 330], [500, 330]]
        result.tracking_bottom = 330

        visible, bottom = line_new.exit_bar_evidence(result)

        self.assertFalse(visible)
        self.assertEqual(bottom, 0)

    def test_diagonal_lane_without_stripes_is_not_stop_bar(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(binary, (80, 430), (560, 100), 255, 18)

        result = line_new.CrosswalkDetector().detect(binary)

        self.assertFalse(result.candidate)

    def test_lane_lock_rejects_lane_edge_that_matches_crosswalk_geometry(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        for x in [380, 450, 520]:
            rotated_rect(binary, (float(x), 175.0), (32.0, 100.0), 8.0)
        cv2.line(binary, (170, 430), (565, 255), 255, 18)
        lane_points = [(int(x), int(y)) for x, y in zip(
            np.linspace(170, 565, 9), np.linspace(430, 255, 9)
        )]

        unlocked = line_new.CrosswalkDetector().detect(binary)
        locked = line_new.CrosswalkDetector().detect(binary, lane_points=lane_points)

        self.assertTrue(unlocked.candidate)
        self.assertFalse(locked.candidate)
        self.assertIsNone(locked.stop_polygon)

    def test_two_current_lane_points_are_enough_to_reject_lane_as_stop_bar(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        for x in [380, 450, 520]:
            rotated_rect(binary, (float(x), 175.0), (32.0, 100.0), 8.0)
        cv2.line(binary, (170, 430), (565, 255), 255, 18)
        sparse_lane = [(170, 430), (565, 255)]

        result = line_new.CrosswalkDetector().detect(
            binary, lane_points=[sparse_lane]
        )

        self.assertFalse(result.candidate)
        self.assertIsNone(result.stop_polygon)

    def test_real_stop_bar_crossing_locked_lane_edges_is_kept(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        for x in [250, 320, 390]:
            rotated_rect(binary, (float(x), 180.0), (32.0, 100.0), 5.0)
        cv2.line(binary, (170, 300), (500, 320), 255, 18)
        left = [(180, 430), (195, 360), (210, 290), (225, 220), (240, 150)]
        right = [(500, 430), (485, 360), (470, 290), (455, 220), (440, 150)]

        result = line_new.CrosswalkDetector().detect(binary, lane_points=[left, right])

        self.assertTrue(result.candidate)
        self.assertIsNotNone(result.stop_polygon)

    def test_current_frame_lane_points_do_not_hide_real_stop_bar(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        for x in [250, 320, 390]:
            rotated_rect(binary, (float(x), 180.0), (32.0, 100.0), 5.0)
        cv2.line(binary, (170, 300), (500, 320), 255, 18)
        cv2.line(binary, (180, 430), (240, 150), 255, 13)
        cv2.line(binary, (500, 430), (440, 150), 255, 13)
        left, right = line_new.LaneDetector().points(binary)

        result = line_new.CrosswalkDetector().detect(
            binary, lane_points=[left, right]
        )

        self.assertTrue(result.candidate)
        self.assertIsNotNone(result.stop_polygon)


class BridgeTests(unittest.TestCase):
    def test_bridge_reports_the_single_side_selected_by_existing_logic(self):
        left_points = [
            (180, 150), (185, 220), (190, 290), (195, 360), (200, 430)
        ]
        right_points = [
            (480, 150), (485, 220), (490, 290), (495, 360), (500, 430)
        ]
        for module in (line_new, line_task):
            bridge = module.DualLineBridge(300.0, fill_width=300.0)
            bridge.update(left_points, [], target_y=380)
            self.assertEqual(bridge.selected_side, "left")

            bridge.reset()
            bridge.update([], right_points, target_y=380)
            self.assertEqual(bridge.selected_side, "right")

    def test_new_right_model_cannot_extrapolate_across_vehicle_center(self):
        bridge = line_new.DualLineBridge(lane_width=300.0)
        zebra_edge = [(340, 300), (400, 340), (460, 380), (520, 420)]

        center, left_model, right_model = bridge.update(
            [], zebra_edge, target_y=380, frame_width=640,
            validation_top_y=96,
        )

        self.assertIsNone(center)
        self.assertIsNone(left_model)
        self.assertIsNone(right_model)

    def test_new_left_model_cannot_extrapolate_across_vehicle_center(self):
        bridge = line_new.DualLineBridge(lane_width=300.0)
        zebra_edge = [(300, 300), (240, 340), (180, 380), (120, 420)]

        center, left_model, right_model = bridge.update(
            zebra_edge, [], target_y=380, frame_width=640,
            validation_top_y=96,
        )

        self.assertIsNone(center)
        self.assertIsNone(left_model)
        self.assertIsNone(right_model)

    def test_new_model_rejects_zebra_edge_even_if_it_stays_on_right(self):
        bridge = line_new.DualLineBridge(lane_width=300.0)
        zebra_edge = [(350, 100), (420, 180), (490, 260), (560, 340),
                      (630, 420)]

        center, _, right_model = bridge.update(
            [], zebra_edge, target_y=380, frame_width=640,
            validation_top_y=96,
        )

        self.assertIsNone(center)
        self.assertIsNone(right_model)

    def test_locked_model_selects_outer_straight_line_over_nearer_curve(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(binary, (500, 430), (470, 120), 255, 13)
        curve = np.asarray([(360, 430), (365, 360), (385, 290), (420, 220)], np.int32)
        cv2.polylines(binary, [curve.reshape(-1, 1, 2)], False, 255, 13)
        model = line_new.LineModel(30.0 / 310.0, 458.0, 6, 1.0)

        points = line_new.LaneDetector().points_near_model(binary, model)

        self.assertGreaterEqual(len(points), 6)
        self.assertTrue(all(abs(x - model.x_at(y)) < 25 for x, y in points))

    def test_locked_left_model_selects_outer_straight_line_over_nearer_curve(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(binary, (140, 430), (170, 120), 255, 13)
        curve = np.asarray([(280, 430), (275, 360), (255, 290), (220, 220)], np.int32)
        cv2.polylines(binary, [curve.reshape(-1, 1, 2)], False, 255, 13)
        model = line_new.LineModel(-30.0 / 310.0, 182.0, 6, 1.0)

        points = line_new.LaneDetector().points_near_model(binary, model)

        self.assertGreaterEqual(len(points), 6)
        self.assertTrue(all(abs(x - model.x_at(y)) < 25 for x, y in points))

    def test_ransac_keeps_straight_right_edge_and_rejects_inward_curve(self):
        straight = [(500 + int(0.12 * (y - 300)), y) for y in [130, 180, 230, 280, 330, 380, 430]]
        curved = [(390, 280), (365, 330), (350, 380), (345, 430)]

        model = line_new.fit_line_ransac(straight + curved, residual=10.0)

        self.assertIsNotNone(model)
        self.assertGreaterEqual(model.inlier_count, 6)
        self.assertAlmostEqual(model.x_at(330), 503.0, delta=12.0)

    def test_dual_bridge_projects_center_from_both_straight_edges(self):
        left = [(180, 150), (185, 220), (190, 290), (195, 360), (200, 430)]
        right = [(480, 150), (485, 220), (490, 290), (495, 360), (500, 430)]
        bridge = line_new.DualLineBridge(lane_width=300.0)

        center, left_model, right_model = bridge.update(left, right, target_y=380)

        self.assertIsNotNone(left_model)
        self.assertIsNotNone(right_model)
        expected = (left_model.x_at(380) + right_model.x_at(380)) * 0.5
        self.assertAlmostEqual(center, expected, delta=0.1)

    def test_perspective_edges_produce_midpoint_center_model_at_all_rows(self):
        left = [(226, 100), (210, 180), (194, 260), (178, 340), (170, 380)]
        right = [(414, 100), (430, 180), (446, 260), (462, 340), (470, 380)]
        bridge = line_new.DualLineBridge(lane_width=300.0)

        center, left_model, right_model = bridge.update(left, right, target_y=380)

        self.assertAlmostEqual(center, 320.0, delta=1.0)
        self.assertIsNotNone(bridge.center_model)
        for y in (100, 240, 380):
            expected = (left_model.x_at(y) + right_model.x_at(y)) * 0.5
            self.assertAlmostEqual(bridge.center_model.x_at(y), expected, delta=0.1)
        self.assertAlmostEqual(bridge.center_model.slope, 0.0, delta=0.01)

    def test_left_edge_uses_learned_perspective_to_restore_center_direction(self):
        left = [(226, 100), (210, 180), (194, 260), (178, 340), (170, 380)]
        right = [(414, 100), (430, 180), (446, 260), (462, 340), (470, 380)]
        bridge = line_new.DualLineBridge(lane_width=300.0)
        bridge.update(left, right, target_y=380)

        center, _, _ = bridge.update(left, [], target_y=380)

        self.assertAlmostEqual(center, 320.0, delta=1.0)
        self.assertAlmostEqual(bridge.center_model.x_at(100), 320.0, delta=1.0)
        self.assertAlmostEqual(bridge.center_model.slope, 0.0, delta=0.01)

    def test_right_edge_uses_learned_perspective_to_restore_center_direction(self):
        left = [(226, 100), (210, 180), (194, 260), (178, 340), (170, 380)]
        right = [(414, 100), (430, 180), (446, 260), (462, 340), (470, 380)]
        bridge = line_new.DualLineBridge(lane_width=300.0)
        bridge.update(left, right, target_y=380)

        center, _, _ = bridge.update([], right, target_y=380)

        self.assertAlmostEqual(center, 320.0, delta=1.0)
        self.assertAlmostEqual(bridge.center_model.x_at(100), 320.0, delta=1.0)
        self.assertAlmostEqual(bridge.center_model.slope, 0.0, delta=0.01)

    def test_dual_bridge_projects_center_from_left_edge_only(self):
        left = [(180, 150), (185, 220), (190, 290), (195, 360), (200, 430)]
        bridge = line_new.DualLineBridge(lane_width=300.0)

        center, left_model, right_model = bridge.update(left, [], target_y=380)

        self.assertIsNotNone(left_model)
        self.assertIsNone(right_model)
        self.assertAlmostEqual(center, left_model.x_at(380) + 150.0, delta=0.1)

    def test_dual_bridge_projects_center_from_right_edge_only(self):
        right = [(480, 150), (485, 220), (490, 290), (495, 360), (500, 430)]
        bridge = line_new.DualLineBridge(lane_width=300.0)

        center, left_model, right_model = bridge.update([], right, target_y=380)

        self.assertIsNone(left_model)
        self.assertIsNotNone(right_model)
        self.assertAlmostEqual(center, right_model.x_at(380) - 150.0, delta=0.1)

    def test_dual_bridge_requires_minimum_points_to_lock_new_line(self):
        sparse_left = [(180, 150), (190, 290), (200, 430)]
        bridge = line_new.DualLineBridge(lane_width=300.0)

        center, left_model, right_model = bridge.update(
            sparse_left, [], target_y=380
        )

        self.assertIsNone(center)
        self.assertIsNone(left_model)
        self.assertIsNone(right_model)

    def test_dual_bridge_uses_other_edge_during_short_left_loss(self):
        left = [(180, 150), (185, 220), (190, 290), (195, 360), (200, 430)]
        right = [(480, 150), (485, 220), (490, 290), (495, 360), (500, 430)]
        bridge = line_new.DualLineBridge(lane_width=300.0)
        center1, _, _ = bridge.update(left, right, 380)
        center2, left_model, right_model = bridge.update([], right, 380)

        self.assertIsNotNone(left_model)
        self.assertIsNotNone(right_model)
        self.assertAlmostEqual(center1, center2, delta=0.1)

    def test_dual_bridge_prefers_fresh_edge_over_held_stale_edge(self):
        left = [(170, 150), (175, 220), (180, 290), (185, 360), (190, 430)]
        right = [(470, 150), (475, 220), (480, 290), (485, 360), (490, 430)]
        shifted_left = [(200, 150), (205, 220), (210, 290), (215, 360), (220, 430)]
        bridge = line_new.DualLineBridge(lane_width=300.0)
        bridge.update(left, right, 380)

        center, left_model, right_model = bridge.update(shifted_left, [], 380)

        self.assertIsNotNone(right_model)
        self.assertGreater(bridge.right_lost_frames, 0)
        self.assertAlmostEqual(center, left_model.x_at(380) + 150.0, delta=0.1)

    def test_dual_bridge_does_not_replace_locked_right_line_with_curve(self):
        bridge = line_new.DualLineBridge(lane_width=300.0)
        straight = [(500, 150), (505, 220), (510, 290), (515, 360), (520, 430)]
        center1, _, model1 = bridge.update([], straight, 380)
        inward_curve = [(430, 150), (405, 220), (380, 290), (360, 360), (350, 430)]

        center2, _, model2 = bridge.update([], inward_curve, 380)

        self.assertIs(model2, model1)
        self.assertAlmostEqual(center2, center1)

    def test_dual_bridge_does_not_replace_locked_left_line_with_curve(self):
        bridge = line_new.DualLineBridge(lane_width=300.0)
        straight = [(140, 150), (145, 220), (150, 290), (155, 360), (160, 430)]
        center1, model1, _ = bridge.update(straight, [], 380)
        outward_curve = [(210, 150), (235, 220), (260, 290), (280, 360), (290, 430)]

        center2, model2, _ = bridge.update(outward_curve, [], 380)

        self.assertIs(model2, model1)
        self.assertAlmostEqual(center2, center1)

    def test_dual_bridge_keeps_center_when_right_edge_curves_away(self):
        left = [(140, 150), (145, 220), (150, 290), (155, 360), (160, 430)]
        right = [(440, 150), (445, 220), (450, 290), (455, 360), (460, 430)]
        bridge = line_new.DualLineBridge(lane_width=300.0)
        center1, _, right_model1 = bridge.update(left, right, 380)
        curved_right = [(370, 150), (345, 220), (320, 290), (300, 360), (290, 430)]

        center2, left_model2, right_model2 = bridge.update(left, curved_right, 380)

        self.assertIsNotNone(left_model2)
        self.assertIs(right_model2, right_model1)
        self.assertAlmostEqual(center2, center1)


class StateTests(unittest.TestCase):
    def _follower_for_maneuver_state(self, turn_cmd):
        follower = line_new.LaneFollower.__new__(line_new.LaneFollower)
        follower.state = "ALIGN"
        follower.state_started = 0.0
        follower.turn_cmd = turn_cmd
        follower.pid = types.SimpleNamespace(reset=lambda: None)
        follower.crosswalk = types.SimpleNamespace(unlock_bar=lambda: None)
        follower.last_angular = 0.0
        follower.last_control_target = None
        follower.lost_hits = follower.align_hits = 0
        return follower

    def test_left_turn_maneuver_starts_with_entry_phase(self):
        follower = self._follower_for_maneuver_state("left")
        original_get_time = getattr(line_new.rospy, "get_time", None)
        original_loginfo = getattr(line_new.rospy, "loginfo", None)
        line_new.rospy.get_time = lambda: 12.0
        line_new.rospy.loginfo = lambda *args: None
        try:
            follower._set_state("MANEUVER")
        finally:
            if original_get_time is None:
                delattr(line_new.rospy, "get_time")
            else:
                line_new.rospy.get_time = original_get_time
            if original_loginfo is None:
                delattr(line_new.rospy, "loginfo")
            else:
                line_new.rospy.loginfo = original_loginfo

        self.assertEqual(follower.maneuver_phase, "ENTRY")
        self.assertEqual(follower.maneuver_phase_started, 12.0)

    def test_straight_maneuver_skips_fixed_turn_phases(self):
        follower = self._follower_for_maneuver_state("straight")
        original_get_time = getattr(line_new.rospy, "get_time", None)
        original_loginfo = getattr(line_new.rospy, "loginfo", None)
        line_new.rospy.get_time = lambda: 15.0
        line_new.rospy.loginfo = lambda *args: None
        try:
            follower._set_state("MANEUVER")
        finally:
            if original_get_time is None:
                delattr(line_new.rospy, "get_time")
            else:
                line_new.rospy.get_time = original_get_time
            if original_loginfo is None:
                delattr(line_new.rospy, "loginfo")
            else:
                line_new.rospy.loginfo = original_loginfo

        self.assertEqual(follower.maneuver_phase, "STRAIGHT")

    def test_fixed_turn_only_relaxes_angular_limit_during_turn_phase(self):
        class Message(object):
            def __init__(self):
                self.linear = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
                self.angular = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)

        follower = line_new.LaneFollower.__new__(line_new.LaneFollower)
        follower.state = "MANEUVER"
        follower.maneuver_phase = "TURN"
        follower.turn_angular = 1.0
        follower.dry_run = False
        published = []
        follower.pub = types.SimpleNamespace(publish=published.append)
        original_twist = line_new.Twist
        line_new.Twist = Message
        try:
            follower.publish(0.16, 1.0)
            follower.state = "FOLLOW"
            follower.publish(0.16, 1.0)
        finally:
            line_new.Twist = original_twist

        self.assertEqual(published[0].angular.z, 1.0)
        self.assertEqual(published[1].angular.z, line_new.MAX_ANGULAR)

    def test_turn_parameters_are_limited_to_timed_motion_values(self):
        self.assertGreaterEqual(line_new.TURN_ENTRY_TIME, 0.0)
        self.assertGreater(line_new.TURN_SPEED, 0.0)
        self.assertGreater(line_new.TURN_ANGULAR, 0.0)
        self.assertGreater(line_new.TURN_TIME, 0.0)
        self.assertFalse(hasattr(line_new, "TURN_CAPTURE_DELAY"))
        self.assertFalse(hasattr(line_new, "TURN_CAPTURE_FRAMES"))
        self.assertFalse(hasattr(line_new, "TURN_TIMEOUT"))

    def test_turn_phases_advance_only_by_elapsed_time(self):
        self.assertIsNone(line_new.turn_phase_next(
            "ENTRY", 0.9, 1.0, 1.6
        ))
        self.assertEqual(line_new.turn_phase_next(
            "ENTRY", 1.0, 1.0, 1.6
        ), "TURN")
        self.assertIsNone(line_new.turn_phase_next(
            "TURN", 1.5, 1.0, 1.6
        ))
        self.assertEqual(line_new.turn_phase_next(
            "TURN", 1.6, 1.0, 1.6
        ), "EXIT_STRAIGHT")

    def test_timed_turn_phase_publishes_fixed_right_turn(self):
        follower = line_new.LaneFollower.__new__(line_new.LaneFollower)
        follower.maneuver_phase = "TURN"
        follower.maneuver_phase_started = 10.0
        follower.turn_cmd = "right"
        follower.turn_entry_time = 7.0
        follower.turn_time = 1.6
        follower.turn_speed = 0.16
        follower.turn_angular = 1.0
        published = []
        follower.publish = lambda linear, angular: published.append(
            (linear, angular)
        )
        follower._set_maneuver_phase = lambda phase, now: setattr(
            follower, "maneuver_phase", phase
        )

        follower._run_timed_turn_phase(11.5)

        self.assertEqual(published[-1], (0.16, -1.0))
        self.assertEqual(follower.maneuver_phase, "TURN")

    def test_timed_turn_phase_goes_straight_after_turn_time(self):
        follower = line_new.LaneFollower.__new__(line_new.LaneFollower)
        follower.maneuver_phase = "TURN"
        follower.maneuver_phase_started = 10.0
        follower.turn_cmd = "left"
        follower.turn_entry_time = 7.0
        follower.turn_time = 1.6
        follower.turn_speed = 0.16
        follower.turn_angular = 1.0
        published = []
        follower.publish = lambda linear, angular: published.append(
            (linear, angular)
        )
        follower._set_maneuver_phase = lambda phase, now: setattr(
            follower, "maneuver_phase", phase
        )

        follower._run_timed_turn_phase(11.61)

        self.assertEqual(follower.maneuver_phase, "EXIT_STRAIGHT")
        self.assertEqual(published[-1], (0.16, 0.0))

    def test_task_final_right_turn_waits_for_exit_bar_after_turn_time(self):
        follower = line_task.LaneFollower.__new__(line_task.LaneFollower)
        follower.state = "MANEUVER"
        follower.maneuver_phase = "TURN"
        follower.maneuver_phase_started = 10.0
        follower.task_index = len(line_task.TASK_TURN_COMMANDS) - 1
        follower.turn_cmd = "right"
        follower.turn_entry_time = 7.0
        follower.turn_time = 1.6
        follower.turn_speed = 0.16
        follower.turn_angular = 1.0
        follower.final_exit_time = 6.0
        published = []
        state_changes = []
        follower.publish = lambda linear, angular: published.append(
            (linear, angular)
        )
        follower._set_maneuver_phase = lambda phase, now: setattr(
            follower, "maneuver_phase", phase
        )
        follower._set_state = lambda state: state_changes.append(state)
        original_loginfo = getattr(line_task.rospy, "loginfo", None)
        line_task.rospy.loginfo = lambda *args: None

        try:
            follower._run_timed_turn_phase(11.61)
        finally:
            if original_loginfo is None:
                delattr(line_task.rospy, "loginfo")
            else:
                line_task.rospy.loginfo = original_loginfo

        self.assertEqual(follower.maneuver_phase, "EXIT_STRAIGHT")
        self.assertEqual(state_changes, [])
        self.assertEqual(published[-1], (0.16, 0.0))

    def test_strong_lane_override_is_limited_to_turn_exit_search(self):
        self.assertTrue(line_new.strong_lane_override_enabled(
            "MANEUVER", "right", "EXIT_STRAIGHT"
        ))
        self.assertTrue(line_new.strong_lane_override_enabled(
            "MANEUVER", "left", "EXIT_STRAIGHT"
        ))
        self.assertFalse(line_new.strong_lane_override_enabled(
            "FOLLOW", "right", "EXIT_STRAIGHT"
        ))
        self.assertFalse(line_new.strong_lane_override_enabled(
            "MANEUVER", "straight", "STRAIGHT"
        ))
        self.assertFalse(line_new.strong_lane_override_enabled(
            "MANEUVER", "right", "TURN"
        ))

    def test_alignment_direction_sign_can_be_reversed(self):
        default = line_new.alignment_angular(10.0, -1.0)
        reversed_direction = line_new.alignment_angular(10.0, 1.0)

        self.assertLess(default, 0.0)
        self.assertGreater(reversed_direction, 0.0)
        self.assertAlmostEqual(abs(default), abs(reversed_direction))

    def test_maneuver_side_control_uses_separate_pid(self):
        follower = line_new.LaneFollower.__new__(line_new.LaneFollower)
        follower.pid = types.SimpleNamespace(
            update=lambda deviation: self.fail("normal PID must not be used")
        )
        follower.maneuver_pid = types.SimpleNamespace(update=lambda deviation: 0.20)
        follower.last_maneuver_angular = 0.0
        published = []
        follower.publish = lambda linear, angular: published.append((linear, angular))

        follower._maneuver_side_control(200.0, 640, 0.16)

        self.assertEqual(len(published), 1)
        self.assertEqual(published[0][0], 0.16)
        self.assertAlmostEqual(
            published[0][1],
            (1.0 - line_new.MANEUVER_SIDE_ANGULAR_SMOOTH) * 0.20,
        )

    def test_fixed_turn_limit_is_independent_from_follow_limit(self):
        self.assertGreater(line_new.TURN_MAX_ANGULAR, line_new.MAX_ANGULAR)
        self.assertEqual(line_new.limit_turn_angular(0.60), 0.60)
        self.assertEqual(
            line_new.limit_turn_angular(2.0), line_new.TURN_MAX_ANGULAR
        )
        self.assertEqual(line_new.limit_publish_angular(0.60), 0.60)

    def test_locked_side_observation_never_falls_back_to_other_line(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(binary, (500, 430), (440, 120), 255, 13)
        left_model = line_new.LineModel(0.0, 140.0, 5, 0.0)
        detector = line_new.LaneDetector(fill_width=300.0)

        observation, model = line_new.observe_locked_side(
            detector, binary, left_model, "left", 300.0, None,
        )

        self.assertFalse(observation.valid)
        self.assertEqual(observation.left_points, [])
        self.assertIs(model, left_model)

    def test_locked_side_observation_requires_entry_model(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(binary, (140, 430), (200, 120), 255, 13)

        observation, model = line_new.observe_locked_side(
            line_new.LaneDetector(fill_width=300.0), binary, None,
            "left", 300.0, None,
        )

        self.assertFalse(observation.valid)
        self.assertIsNone(model)

    def test_maneuver_clearance_moves_target_away_from_followed_edge(self):
        binary = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(binary, (500, 430), (440, 120), 255, 13)
        right_model = line_new.LineModel(60.0 / 310.0, 416.8, 6, 1.0)
        detector = line_new.LaneDetector(fill_width=300.0)

        normal, _ = line_new.observe_locked_side(
            detector, binary, right_model, "right", 300.0, None, 0.0,
        )
        cleared, _ = line_new.observe_locked_side(
            detector, binary, right_model, "right", 300.0, None, 30.0,
        )

        self.assertTrue(normal.valid)
        self.assertTrue(cleared.valid)
        self.assertAlmostEqual(cleared.center_x, normal.center_x - 30.0)

    def test_side_lane_target_has_priority_when_requested_edge_is_stable(self):
        left = line_new.LaneObservation(
            250.0, True, 0,
            [(100, 400), (110, 330), (120, 260), (130, 190)], [],
            follow_side="left",
        )
        right = line_new.LaneObservation(
            390.0, True, 0, [],
            [(540, 400), (530, 330), (520, 260), (510, 190)],
            follow_side="right",
        )

        self.assertEqual(line_new.side_lane_target(left, "left"), 250.0)
        self.assertEqual(line_new.side_lane_target(right, "right"), 390.0)

    def test_side_lane_target_rejects_sparse_or_wrong_side_points(self):
        sparse = line_new.LaneObservation(
            250.0, True, 0, [(100, 400), (110, 330)], [],
            follow_side="left",
        )
        wrong_side = line_new.LaneObservation(
            390.0, True, 0, [],
            [(540, 400), (530, 330), (520, 260), (510, 190)],
            follow_side="right",
        )

        self.assertIsNone(line_new.side_lane_target(sparse, "left"))
        self.assertIsNone(line_new.side_lane_target(wrong_side, "left"))

    def test_side_lane_target_never_steers_opposite_requested_turn(self):
        left_target_on_right = line_new.LaneObservation(
            450.0, True, 0,
            [(100, 400), (110, 330), (120, 260), (130, 190)], [],
            follow_side="left",
        )
        right_target_on_left = line_new.LaneObservation(
            190.0, True, 0, [],
            [(540, 400), (530, 330), (520, 260), (510, 190)],
            follow_side="right",
        )

        self.assertIsNone(
            line_new.side_lane_target(left_target_on_right, "left", 640)
        )
        self.assertIsNone(
            line_new.side_lane_target(right_target_on_left, "right", 640)
        )

    def test_fixed_turn_command_uses_requested_direction(self):
        self.assertEqual(
            line_new.fixed_turn_command("left", 0.12, 0.30),
            (0.12, 0.30),
        )
        self.assertEqual(
            line_new.fixed_turn_command("right", 0.12, 0.30),
            (0.12, -0.30),
        )
        self.assertIsNone(
            line_new.fixed_turn_command("straight", 0.12, 0.30)
        )
        self.assertEqual(
            line_new.fixed_turn_command("left", 0.12, -0.30),
            (0.12, -0.30),
        )

    def test_frame_freeze_guard_stops_after_repeated_identical_frames(self):
        guard = line_new.FrameFreezeGuard(limit=3)
        frame = np.zeros((48, 64, 3), dtype=np.uint8)

        self.assertFalse(guard.update(frame))
        self.assertFalse(guard.update(frame.copy()))
        self.assertFalse(guard.update(frame.copy()))
        self.assertTrue(guard.update(frame.copy()))

        changed = frame.copy()
        changed[10:20, 10:20] = 255
        self.assertFalse(guard.update(changed))

    def test_exit_bar_requires_crosswalk_stripes(self):
        self.assertFalse(line_new.exit_bar_visible(
            True, True, 450, 480,
            stripe_count=line_new.EXIT_STRIPE_MIN_COUNT - 1,
        ))
        self.assertTrue(line_new.exit_bar_visible(
            True, True, 450, 480,
            stripe_count=line_new.EXIT_STRIPE_MIN_COUNT,
        ))
        self.assertFalse(line_new.exit_bar_visible(
            True, True, 450, 480,
            stripe_count=line_new.EXIT_STRIPE_MIN_COUNT,
            frame_frozen=True,
        ))

    def test_exit_bar_accepts_two_visible_stripes_after_turn(self):
        self.assertEqual(line_new.EXIT_STRIPE_MIN_COUNT, 2)
        self.assertTrue(line_new.exit_bar_visible(
            True, True, 450, 480, stripe_count=2,
        ))

    def test_turn_command_defaults_to_straight_and_accepts_ros_values(self):
        self.assertEqual(line_new.TURN_CMD, "straight")
        self.assertEqual(line_new.normalize_turn_cmd(" LEFT "), "left")
        self.assertEqual(line_new.normalize_turn_cmd("right"), "right")
        self.assertEqual(line_new.normalize_turn_cmd("invalid"), "straight")

    def test_follow_entry_hits_accumulate_candidate_frames(self):
        self.assertEqual(line_new.follow_entry_hits(True, 2), 3)

    def test_entry_is_ignored_until_exit_forward_delay_finishes(self):
        accept_after = 20.0 + line_new.EXIT_ENTRY_IGNORE_TIME

        self.assertFalse(line_new.entry_acceptance_enabled(
            accept_after - 0.01, accept_after
        ))
        self.assertTrue(line_new.entry_acceptance_enabled(
            accept_after, accept_after
        ))

    def test_exit_alignment_to_follow_starts_entry_ignore_time(self):
        follower = line_new.LaneFollower.__new__(line_new.LaneFollower)
        follower.state = "EXIT_ALIGN"
        follower.state_started = 0.0
        follower.pid = types.SimpleNamespace(reset=lambda: None)
        follower.crosswalk = types.SimpleNamespace(unlock_bar=lambda: None)
        follower.last_angular = 0.0
        follower.last_control_target = None
        follower.lost_hits = follower.align_hits = follower.stop_hits = 0
        original_get_time = getattr(line_new.rospy, "get_time", None)
        original_loginfo = getattr(line_new.rospy, "loginfo", None)
        line_new.rospy.get_time = lambda: 20.0
        line_new.rospy.loginfo = lambda *args: None
        try:
            follower._set_state("FOLLOW")
        finally:
            if original_get_time is None:
                delattr(line_new.rospy, "get_time")
            else:
                line_new.rospy.get_time = original_get_time
            if original_loginfo is None:
                delattr(line_new.rospy, "loginfo")
            else:
                line_new.rospy.loginfo = original_loginfo

        self.assertAlmostEqual(
            follower.entry_accept_after,
            20.0 + line_new.EXIT_ENTRY_IGNORE_TIME,
        )

    def test_approach_loss_cancels_entry_and_resumes_following(self):
        state = line_new.approach_next_state(
            visible=False, bottom=0, frame_height=480,
            lost_hits=line_new.LOST_LIMIT + 1,
        )

        self.assertEqual(state, "FOLLOW")

    def test_stop_bottom_ignores_bar_reaching_bottom_only_at_frame_edge(self):
        polygon = [(0, 190), (640, 430), (640, 450), (0, 210)]

        bottom = line_new.polygon_bottom_in_center_band(
            polygon, 640, center_width_ratio=0.60,
        )
        full_width_bottom = line_new.polygon_bottom_in_center_band(
            polygon, 640, center_width_ratio=1.0,
        )

        self.assertLess(bottom, 480 * line_new.STOP_NEAR_RATIO)
        self.assertEqual(full_width_bottom, 450)

    def test_stop_bottom_accepts_bar_reaching_bottom_in_center_band(self):
        polygon = [(100, 400), (540, 400), (540, 430), (100, 430)]

        bottom = line_new.polygon_bottom_in_center_band(
            polygon, 640, center_width_ratio=0.60,
        )

        self.assertEqual(bottom, 430)

    def test_stop_bottom_accepts_partial_overlap_with_center_band(self):
        polygon = [(500, 400), (620, 400), (620, 430), (500, 430)]

        bottom = line_new.polygon_bottom_in_center_band(
            polygon, 640, center_width_ratio=0.60,
        )

        self.assertEqual(bottom, 430)

    def test_approach_reaches_near_bar_before_alignment(self):
        state = line_new.approach_next_state(
            visible=True, bottom=int(480 * line_new.ALIGN_START_RATIO) + 1,
            frame_height=480, lost_hits=0,
        )

        self.assertEqual(state, "ALIGN")

    def test_alignment_stable_enters_maneuver(self):
        state = line_new.alignment_next_state(
            angle=2.0, visible=True, align_hits=line_new.ALIGN_STABLE_FRAMES,
            lost_hits=0, elapsed=1.0, stripe_count=6,
        )

        self.assertEqual(state, "MANEUVER")

    def test_alignment_timeout_does_not_relax_angle_requirement(self):
        state = line_new.alignment_next_state(
            angle=6.0, visible=True, align_hits=0, lost_hits=0,
            elapsed=line_new.ALIGN_TIMEOUT + 0.1, stripe_count=6,
        )

        self.assertIsNone(state)

    def test_alignment_loss_after_near_bar_waits_instead_of_entering(self):
        state = line_new.alignment_next_state(
            angle=None, visible=False, align_hits=0,
            lost_hits=line_new.LOST_LIMIT + 1,
            elapsed=line_new.ALIGN_TIMEOUT + 0.1, stripe_count=0,
        )

        self.assertEqual(state, "WAIT")

    def test_alignment_keeps_turning_when_visible_angle_is_still_too_large(self):
        state = line_new.alignment_next_state(
            angle=18.0, visible=True, align_hits=0, lost_hits=0,
            elapsed=line_new.ALIGN_TIMEOUT + 0.1, stripe_count=6,
        )

        self.assertIsNone(state)

    def test_wait_recovery_returns_to_alignment_after_stable_detection(self):
        state = line_new.wait_recovery_state(
            angle=6.0, visible=True, recover_hits=line_new.WAIT_RECOVER_FRAMES,
            stripe_count=6,
        )

        self.assertEqual(state, "ALIGN")

    def test_wait_recovery_never_bypasses_alignment(self):
        state = line_new.wait_recovery_state(
            angle=2.0, visible=True, recover_hits=line_new.WAIT_RECOVER_FRAMES,
            stripe_count=6,
        )

        self.assertEqual(state, "ALIGN")

    def test_maneuver_waits_until_exit_bar_reaches_near_threshold(self):
        exit_hits, near = line_new.maneuver_exit(False, 0, True, 450, 480, 2)
        self.assertFalse(near)
        for _ in range(2):
            exit_hits, near = line_new.maneuver_exit(
                True, exit_hits, True,
                int(480 * line_new.STOP_NEAR_RATIO) - 1, 480, 2
            )

        self.assertFalse(near)
        exit_hits, near = line_new.maneuver_exit(
            True, exit_hits, True,
            int(480 * line_new.STOP_NEAR_RATIO) + 1, 480, 2
        )
        self.assertTrue(near)

    def test_maneuver_does_not_exit_when_dual_lanes_restore_inside_intersection(self):
        exit_hits = 0
        for _ in range(6):
            exit_hits, near = line_new.maneuver_exit(
                True, exit_hits, False, 0, 480, line_new.EXIT_BAR_FRAMES
            )

        self.assertFalse(near)

    def test_maneuver_timeout_resumes_following(self):
        self.assertFalse(line_new.maneuver_timeout_exits_to_follow(
            line_new.MANEUVER_MAX_TIME - 0.01
        ))
        self.assertTrue(line_new.maneuver_timeout_exits_to_follow(
            line_new.MANEUVER_MAX_TIME
        ))

    def test_maneuver_timeout_follow_starts_entry_ignore_time(self):
        follower = line_new.LaneFollower.__new__(line_new.LaneFollower)
        follower.state = "MANEUVER"
        follower.state_started = 0.0
        follower.pid = types.SimpleNamespace(reset=lambda: None)
        follower.crosswalk = types.SimpleNamespace(unlock_bar=lambda: None)
        follower.last_angular = 0.0
        follower.last_control_target = None
        follower.lost_hits = follower.align_hits = follower.stop_hits = 0
        follower.entry_accept_after = 0.0
        original_get_time = getattr(line_new.rospy, "get_time", None)
        original_loginfo = getattr(line_new.rospy, "loginfo", None)
        line_new.rospy.get_time = lambda: 30.0
        line_new.rospy.loginfo = lambda *args: None
        try:
            follower._set_state("FOLLOW")
        finally:
            if original_get_time is None:
                delattr(line_new.rospy, "get_time")
            else:
                line_new.rospy.get_time = original_get_time
            if original_loginfo is None:
                delattr(line_new.rospy, "loginfo")
            else:
                line_new.rospy.loginfo = original_loginfo

        self.assertAlmostEqual(
            follower.entry_accept_after,
            30.0 + line_new.EXIT_ENTRY_IGNORE_TIME,
        )

    def test_task_timeout_completion_advances_route_and_starts_ignore_time(self):
        follower = line_task.LaneFollower.__new__(line_task.LaneFollower)
        follower.state = "MANEUVER"
        follower.state_started = 0.0
        follower.task_index = 4
        follower.turn_cmd = line_task.TASK_TURN_COMMANDS[follower.task_index]
        follower.pid = types.SimpleNamespace(reset=lambda: None)
        follower.crosswalk = types.SimpleNamespace(unlock_bar=lambda: None)
        follower.last_angular = 0.0
        follower.last_control_target = None
        follower.lost_hits = follower.align_hits = follower.stop_hits = 0
        follower.entry_accept_after = 0.0
        original_get_time = getattr(line_task.rospy, "get_time", None)
        original_loginfo = getattr(line_task.rospy, "loginfo", None)
        line_task.rospy.get_time = lambda: 40.0
        line_task.rospy.loginfo = lambda *args: None
        try:
            follower._complete_intersection()
        finally:
            if original_get_time is None:
                delattr(line_task.rospy, "get_time")
            else:
                line_task.rospy.get_time = original_get_time
            if original_loginfo is None:
                delattr(line_task.rospy, "loginfo")
            else:
                line_task.rospy.loginfo = original_loginfo

        self.assertEqual(follower.state, "FOLLOW")
        self.assertEqual(follower.task_index, 5)
        self.assertEqual(follower.turn_cmd, "left")
        self.assertAlmostEqual(
            follower.entry_accept_after,
            40.0 + line_task.EXIT_ENTRY_IGNORE_TIME,
        )

    def test_maneuver_requires_confirmed_second_bar_not_tracking_line(self):
        exit_hits = 0
        for _ in range(line_new.EXIT_BAR_FRAMES + 2):
            exit_hits, near = line_new.maneuver_exit(
                True, exit_hits, False, 450, 480, line_new.EXIT_BAR_FRAMES
            )

        self.assertEqual(exit_hits, 0)
        self.assertFalse(near)

    def test_exit_alignment_stable_resumes_following(self):
        state = line_new.exit_alignment_next_state(
            align_hits=line_new.ALIGN_STABLE_FRAMES,
            lost_hits=0, elapsed=1.0,
        )

        self.assertEqual(state, "FOLLOW")

    def test_exit_alignment_waits_for_five_lost_frames(self):
        self.assertIsNone(line_new.exit_alignment_next_state(
            align_hits=0,
            lost_hits=line_new.EXIT_ALIGN_LOST_FRAMES - 1,
            elapsed=0.2,
        ))
        self.assertEqual(line_new.exit_alignment_next_state(
            align_hits=0,
            lost_hits=line_new.EXIT_ALIGN_LOST_FRAMES,
            elapsed=0.25,
        ), "FOLLOW")

    def test_both_routes_use_camera_four_and_required_traffic_model(self):
        expected_model = (
            "/home/eaibot/handeye-calib/src/model/yolov5/"
            "traffic_lights_yolov5n_320_best.onnx"
        )
        for module in (line_new, line_task):
            self.assertEqual(module.CAMERA_INDEX, 4)
            self.assertTrue(module.TRAFFIC_LIGHT_ENABLED)
            self.assertEqual(module.TRAFFIC_LIGHT_CAMERA_INDEX, 0)
            self.assertEqual(module.TRAFFIC_LIGHT_FRAME_WIDTH, 320)
            self.assertEqual(module.TRAFFIC_LIGHT_FRAME_HEIGHT, 240)
            self.assertEqual(module.TRAFFIC_LIGHT_MODEL_PATH, expected_model)

    def test_entry_alignment_routes_through_traffic_wait(self):
        for module in (line_new, line_task):
            follower = module.LaneFollower.__new__(module.LaneFollower)
            follower.traffic_light_enabled = True
            self.assertEqual(follower._entry_ready_state(), "TRAFFIC_WAIT")
            follower.traffic_light_enabled = False
            self.assertEqual(follower._entry_ready_state(), "MANEUVER")

    def test_closing_traffic_wait_releases_owned_model_and_camera(self):
        events = []
        follower = line_new.LaneFollower.__new__(line_new.LaneFollower)
        follower.traffic_detector = types.SimpleNamespace(
            close=lambda: events.append("model_closed")
        )
        follower.traffic_camera = types.SimpleNamespace(
            release=lambda: events.append("camera_closed")
        )
        follower.traffic_green_hits = 2
        follower.traffic_last_color = "Green"

        follower._close_traffic_light()

        self.assertEqual(events, ["model_closed", "camera_closed"])
        self.assertIsNone(follower.traffic_detector)
        self.assertIsNone(follower.traffic_camera)

    def test_two_green_frames_are_required_before_maneuver(self):
        detection = types.SimpleNamespace(class_name="Green", confidence=0.9)
        follower = line_new.LaneFollower.__new__(line_new.LaneFollower)
        follower.traffic_detector = types.SimpleNamespace(
            detect=lambda frame: [detection]
        )
        follower.traffic_camera = types.SimpleNamespace(
            read=lambda timeout: (True, np.zeros((240, 320, 3), np.uint8))
        )
        follower.traffic_green_hits = 0
        follower.traffic_green_stable_frames = 2
        follower.traffic_last_color = None
        follower.traffic_retry_after = 0.0
        follower.debug_view = False
        follower.publish = lambda linear, angular: None
        transitions = []
        follower._set_state = lambda state: transitions.append(state)
        original_loginfo = getattr(line_new.rospy, "loginfo", None)
        line_new.rospy.loginfo = lambda *args: None
        try:
            follower._handle_traffic_light_wait(10.0)
            self.assertEqual(transitions, [])
            follower._handle_traffic_light_wait(10.1)
        finally:
            if original_loginfo is None:
                delattr(line_new.rospy, "loginfo")
            else:
                line_new.rospy.loginfo = original_loginfo

        self.assertEqual(transitions, ["MANEUVER"])

    def test_task_traffic_wait_does_not_overlap_existing_yolo_inference(self):
        follower = line_task.LaneFollower.__new__(line_task.LaneFollower)
        follower.yolo_worker_active = True
        follower.traffic_detector = None
        follower.traffic_camera = None
        follower.published = []
        follower.publish = lambda linear, angular: follower.published.append(
            (linear, angular)
        )
        follower._open_traffic_light = lambda: self.fail(
            "旧 YOLO 推理未结束时不应加载红绿灯模型"
        )

        follower._handle_traffic_light_wait(10.0)

        self.assertEqual(follower.published, [(0, 0)])


class TaskEntryAlignmentLockTests(unittest.TestCase):
    def _patch_task_rospy(self, now=10.0):
        original_get_time = getattr(line_task.rospy, "get_time", None)
        original_loginfo = getattr(line_task.rospy, "loginfo", None)
        original_logwarn = getattr(line_task.rospy, "logwarn", None)
        line_task.rospy.get_time = lambda: now
        line_task.rospy.loginfo = lambda *args: None
        line_task.rospy.logwarn = lambda *args: None

        def restore():
            if original_get_time is None:
                delattr(line_task.rospy, "get_time")
            else:
                line_task.rospy.get_time = original_get_time
            if original_loginfo is None:
                delattr(line_task.rospy, "loginfo")
            else:
                line_task.rospy.loginfo = original_loginfo
            if original_logwarn is None:
                delattr(line_task.rospy, "logwarn")
            else:
                line_task.rospy.logwarn = original_logwarn
        return restore

    def _follower(self, now=10.0):
        follower = line_task.LaneFollower.__new__(line_task.LaneFollower)
        follower.state = "APPROACH"
        follower.pid = types.SimpleNamespace(reset=lambda: None)
        follower.published = []
        follower.publish = lambda linear, angular: follower.published.append(
            (linear, angular)
        )
        follower.align_lock = None
        follower.last_angular = 0.0
        follower.last_control_target = None
        follower.lost_hits = 0
        follower.align_hits = 0
        follower.last_crosswalk = line_task.CrosswalkResult()
        follower._restore_rospy = self._patch_task_rospy(now)
        return follower

    def tearDown(self):
        restore = getattr(self, "_restore_rospy", None)
        if restore is not None:
            restore()

    def test_lost_bar_alignment_keeps_using_last_bar_angle(self):
        follower = self._follower(now=20.0)
        self._restore_rospy = follower._restore_rospy
        follower.last_crosswalk.candidate = True
        follower.last_crosswalk.stop_angle = 10.0

        follower._set_state("ALIGN")
        follower._lock_entry_alignment(20.0, 10.0)
        follower.last_crosswalk.stop_angle = -25.0
        handled, next_state = follower._run_locked_entry_alignment(20.01)

        self.assertTrue(handled)
        self.assertIsNone(next_state)
        self.assertLess(follower.published[-1][1], 0.0)

    def test_lost_bar_alignment_finishes_after_rotation_and_settle(self):
        follower = self._follower(now=20.0)
        self._restore_rospy = follower._restore_rospy
        follower.last_crosswalk.candidate = True
        follower.last_crosswalk.stop_angle = 10.0
        follower._set_state("ALIGN")
        follower._lock_entry_alignment(20.0, 10.0)

        handled, next_state = follower._run_locked_entry_alignment(25.0)

        self.assertTrue(handled)
        self.assertEqual(next_state, "MANEUVER")


class TaskYoloTests(unittest.TestCase):
    def _patch_task_rospy(self, now=10.0):
        original_get_time = getattr(line_task.rospy, "get_time", None)
        original_loginfo = getattr(line_task.rospy, "loginfo", None)
        original_logwarn = getattr(line_task.rospy, "logwarn", None)
        line_task.rospy.get_time = lambda: now
        line_task.rospy.loginfo = lambda *args: None
        line_task.rospy.logwarn = lambda *args: None

        def restore():
            if original_get_time is None:
                delattr(line_task.rospy, "get_time")
            else:
                line_task.rospy.get_time = original_get_time
            if original_loginfo is None:
                delattr(line_task.rospy, "loginfo")
            else:
                line_task.rospy.loginfo = original_loginfo
            if original_logwarn is None:
                delattr(line_task.rospy, "logwarn")
            else:
                line_task.rospy.logwarn = original_logwarn
        return restore

    def _follower(self, now=10.0, yolo_stop_enabled=True):
        follower = line_task.LaneFollower.__new__(line_task.LaneFollower)
        follower.state = "FOLLOW"
        follower.state_started = 0.0
        follower.pid = types.SimpleNamespace(reset=lambda: None)
        follower.crosswalk = types.SimpleNamespace(unlock_bar=lambda: None)
        follower.last_angular = 0.0
        follower.last_control_target = None
        follower.lost_hits = follower.align_hits = follower.stop_hits = 0
        follower.entry_accept_after = 0.0
        follower.maneuver_phase = "NONE"
        follower.yolo_enabled = True
        follower.yolo_stop_enabled = yolo_stop_enabled
        follower.yolo_debug_view = False
        follower.yolo_confidence = 0.5
        follower.yolo_trash_confidence = 0.65
        follower.yolo_building_confidence = 0.65
        follower.yolo_center_band_ratio = 0.8
        follower.yolo_class_names = line_task.YOLO_CLASS_NAMES
        follower.yolo_street_model_path = "/tmp/street.onnx"
        follower.yolo_building_model_path = "/tmp/building.onnx"
        follower.yolo_street_class_names = line_task.YOLO_STREET_CLASS_NAMES
        follower.yolo_building_class_names = line_task.YOLO_BUILDING_CLASS_NAMES
        follower.yolo_image_size = 320
        follower.yolo_nms_threshold = 0.45
        follower.task_index = 0
        follower.task_ledger = line_task.YoloTaskLedger()
        follower.yolo_segment_key = None
        follower.yolo_segment_start_seq = 0
        follower.yolo_stop_detection = None
        follower.yolo_stop_reported = False
        follower.yolo_stop_report_seq = 0
        follower.yolo_event_ignore_time = 4.0
        follower.yolo_accept_after = 0.0
        follower.yolo_lock = threading.Lock()
        follower.yolo_switch_lock = threading.Lock()
        follower.yolo_latest_seq = 0
        follower.yolo_read_seq = 0
        follower.yolo_latest_detections = []
        follower.yolo_latest_frame = None
        follower.yolo_ready = False
        follower.yolo_active_profile = None
        follower.yolo_running = False
        follower.yolo_thread = None
        follower.yolo_worker_active = False
        follower.published = []
        follower.last_observation = None
        follower.last_crosswalk = None
        follower.last_binary = None
        follower.publish = lambda linear, angular: follower.published.append(
            (linear, angular)
        )
        follower._poll_yolo_detections = lambda: (False, [])
        follower._restore_rospy = self._patch_task_rospy(now)
        return follower

    def tearDown(self):
        restore = getattr(self, "_restore_rospy", None)
        if restore is not None:
            restore()

    def test_yolo_model_directory_prefers_merge_best_onnx(self):
        with tempfile.TemporaryDirectory() as root:
            open(os.path.join(root, "z_other.onnx"), "w").close()
            preferred = os.path.join(root, "merge_new_yolov5n_320_best.onnx")
            open(preferred, "w").close()

            self.assertEqual(line_task.resolve_yolo_model_path(root), preferred)

    def test_default_yolo_models_use_separate_yolov5n_320_onnx_files(self):
        self.assertEqual(
            line_task.YOLO_STREET_MODEL_PATH,
            "/home/eaibot/handeye-calib/src/model/yolov5/"
            "rub_roll_new_yolov5n_320_best.onnx",
        )
        self.assertEqual(
            line_task.YOLO_BUILDING_MODEL_PATH,
            "/home/eaibot/handeye-calib/src/model/yolov5/"
            "building_new_yolov5n_320_best.onnx",
        )
        self.assertEqual(line_task.YOLO_MODEL_PATH,
                         line_task.YOLO_STREET_MODEL_PATH)
        self.assertEqual(line_task.YOLO_IMAGE_SIZE, 320)
        self.assertEqual(line_task.YOLO_CONFIDENCE, 0.60)
        self.assertEqual(line_task.YOLO_TRASH_CONFIDENCE, 0.65)
        self.assertEqual(line_task.YOLO_BUILDING_CONFIDENCE, 0.65)

    def test_line_operation_doc_uses_separate_task_models(self):
        doc_path = os.path.abspath(os.path.join(
            os.path.dirname(line_task.__file__), "..", "..",
            "zcy", "循迹操作.md",
        ))
        with open(doc_path, "r", encoding="utf-8") as handle:
            content = handle.read()

        self.assertIn("rubbish_doll_yolov5n_320_best.onnx", content)
        self.assertIn("building_new_yolov5n_320_best.onnx", content)
        self.assertNotIn("merge_new_1_yolov5n_320_best.onnx", content)
        self.assertNotIn("merge_yolov5n_320_best.onnx", content)
        self.assertNotIn("_yolo_class_profile", content)
        self.assertNotIn("legacy", content)

    def test_yolo_model_file_rejects_non_onnx_extension(self):
        with tempfile.TemporaryDirectory() as root:
            model_path = os.path.join(root, "merge_yolov5n_320_best.weights")
            open(model_path, "w").close()

            with self.assertRaises(IOError):
                line_task.resolve_yolo_model_path(model_path)

    def test_yolo_detector_uses_custom_class_names(self):
        with tempfile.TemporaryDirectory() as root:
            model_path = os.path.join(root, "merge_yolov5n_320_best.onnx")
            open(model_path, "w").close()
            detector = line_task.YoloObstacleDetector(
                model_path,
                class_names=("custom_a", "custom_b"),
            )
            self.assertEqual(detector.names, {0: "custom_a", 1: "custom_b"})

    def test_opencv_dnn_onnx_detector_decodes_yolov8_output(self):
        class FakeNet(object):
            def __init__(self):
                self.blob = None

            def setPreferableBackend(self, backend):
                pass

            def setPreferableTarget(self, target):
                pass

            def setInput(self, blob):
                self.blob = blob

            def forward(self):
                row = np.zeros((4 + len(line_task.YOLO_CLASS_NAMES), 1),
                               dtype=np.float32)
                row[0, 0] = 320.0
                row[1, 0] = 320.0
                row[2, 0] = 100.0
                row[3, 0] = 80.0
                row[4 + 2, 0] = 0.90
                return row.reshape(1, row.shape[0], 1)

        with tempfile.TemporaryDirectory() as root:
            model_path = os.path.join(root, "merge_yolov8n_640_best.onnx")
            open(model_path, "w").close()
            original_read = line_task.cv2.dnn.readNetFromONNX
            line_task.cv2.dnn.readNetFromONNX = lambda path: FakeNet()
            try:
                detector = line_task.YoloObstacleDetector(
                    model_path, confidence=0.5, center_band_ratio=0.8,
                    image_size=640,
                )
                detections = detector.detect(
                    np.zeros((480, 640, 3), dtype=np.uint8)
                )
            finally:
                line_task.cv2.dnn.readNetFromONNX = original_read

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_name, "Fire Building")
        self.assertAlmostEqual(detections[0].confidence, 0.90, places=4)
        self.assertEqual(tuple(round(value) for value in detections[0].box),
                         (270, 200, 370, 280))

    def test_opencv_dnn_onnx_detector_decodes_yolov5_output(self):
        class FakeNet(object):
            def setPreferableBackend(self, backend):
                pass

            def setPreferableTarget(self, target):
                pass

            def setInput(self, blob):
                pass

            def forward(self):
                row = np.zeros((1, 5 + len(line_task.YOLO_CLASS_NAMES)),
                               dtype=np.float32)
                row[0, 0] = 160.0
                row[0, 1] = 160.0
                row[0, 2] = 50.0
                row[0, 3] = 40.0
                row[0, 4] = 0.80
                row[0, 5 + 2] = 0.90
                return row.reshape(1, 1, row.shape[1])

        with tempfile.TemporaryDirectory() as root:
            model_path = os.path.join(root, "merge_yolov5n_320_best.onnx")
            open(model_path, "w").close()
            original_read = line_task.cv2.dnn.readNetFromONNX
            line_task.cv2.dnn.readNetFromONNX = lambda path: FakeNet()
            try:
                detector = line_task.YoloObstacleDetector(
                    model_path, confidence=0.5, center_band_ratio=0.8,
                    image_size=320,
                )
                detections = detector.detect(
                    np.zeros((240, 320, 3), dtype=np.uint8)
                )
            finally:
                line_task.cv2.dnn.readNetFromONNX = original_read

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_name, "Fire Building")
        self.assertAlmostEqual(detections[0].confidence, 0.72, places=4)
        self.assertEqual(tuple(round(value) for value in detections[0].box),
                         (135, 100, 185, 140))

    def test_default_yolo_frame_interval_is_positive(self):
        self.assertGreaterEqual(line_task.YOLO_FRAME_INTERVAL, 1)

    def test_init_yolo_warms_first_inference_before_starting_worker_thread(self):
        follower = self._follower(now=20.0)
        self._restore_rospy = follower._restore_rospy
        events = []
        class FakeDetector(object):
            model_path = "/tmp/fake.onnx"
            backend_name = "opencv-dnn-onnx"

            def __init__(self, *args, **kwargs):
                events.append("detector_created")

            def load(self):
                events.append("load_start")
                events.append("load_done")

            def detect(self, frame):
                events.append("warmup_detect")
                return []

        class FakeCamera(object):
            def __init__(self, index, frame_width=None, frame_height=None):
                events.append("camera_created")
                self.cap = types.SimpleNamespace(isOpened=lambda: True)

            def read(self, timeout=0.0):
                events.append("camera_read")
                return True, np.zeros((100, 200, 3), dtype=np.uint8)

        class FakeThread(object):
            daemon = False

            def __init__(self, target):
                events.append("thread_created")

            def start(self):
                events.append("thread_started")

        original_detector = line_task.YoloObstacleDetector
        original_camera = line_task.CameraReader
        original_thread = line_task.threading.Thread
        line_task.YoloObstacleDetector = FakeDetector
        line_task.CameraReader = FakeCamera
        line_task.threading.Thread = FakeThread
        try:
            delattr(follower, "_poll_yolo_detections")
            follower.yolo_model_path = "/tmp/fake.onnx"
            follower.yolo_camera_index = 0
            follower.yolo_frame_interval = 1
            follower._init_yolo()
        finally:
            line_task.YoloObstacleDetector = original_detector
            line_task.CameraReader = original_camera
            line_task.threading.Thread = original_thread

        self.assertEqual(
            events,
            ["detector_created", "load_start", "load_done",
             "camera_created", "camera_read", "warmup_detect",
             "thread_created", "thread_started"],
        )
        self.assertTrue(follower.yolo_ready)
        self.assertEqual(follower.yolo_latest_seq, 1)

    def test_poll_yolo_detections_reads_cached_result_without_inference(self):
        follower = self._follower(now=20.0)
        self._restore_rospy = follower._restore_rospy
        delattr(follower, "_poll_yolo_detections")
        detection = line_task.YoloDetection(
            3, "Fire Building", 0.9, (80, 30, 100, 60), (100, 200, 3), 0.6
        )
        follower.yolo_detector = types.SimpleNamespace(
            detect=lambda frame: self.fail("main thread must not run YOLO detect")
        )
        follower.yolo_camera = types.SimpleNamespace(
            read=lambda timeout=0.0: (True, np.zeros((100, 200, 3), dtype=np.uint8))
        )
        follower.yolo_frame_interval = 5
        follower.yolo_counter = 4
        with follower.yolo_lock:
            follower.yolo_latest_seq = 1
            follower.yolo_read_seq = 0
            follower.yolo_latest_detections = [detection]

        sampled, detections = follower._poll_yolo_detections()

        self.assertTrue(sampled)
        self.assertEqual(detections, [detection])
        self.assertEqual(follower.yolo_read_seq, 1)

    def test_poll_yolo_detections_returns_false_when_cache_already_read(self):
        follower = self._follower(now=20.0)
        self._restore_rospy = follower._restore_rospy
        delattr(follower, "_poll_yolo_detections")
        follower.yolo_detector = types.SimpleNamespace(
            detect=lambda frame: self.fail("main thread must not run YOLO detect")
        )
        follower.yolo_camera = types.SimpleNamespace(read=lambda timeout=0.0: (True, None))
        follower.yolo_frame_interval = 5
        with follower.yolo_lock:
            follower.yolo_latest_seq = 3
            follower.yolo_read_seq = 3

        sampled, detections = follower._poll_yolo_detections()

        self.assertFalse(sampled)
        self.assertEqual(detections, [])

    def test_yolo_debug_window_draws_detection_box(self):
        follower = self._follower(now=20.0)
        self._restore_rospy = follower._restore_rospy
        detection = line_task.YoloDetection(
            3, "Fire Building", 0.9, (80, 30, 120, 70), (100, 200, 3), 0.6
        )
        with follower.yolo_lock:
            follower.yolo_latest_frame = np.zeros((100, 200, 3), dtype=np.uint8)
            follower.yolo_latest_detections = [detection]
        shown = []
        original_imshow = line_task.cv2.imshow
        line_task.cv2.imshow = lambda name, frame: shown.append((name, frame.copy()))
        try:
            follower.draw_yolo_debug()
        finally:
            line_task.cv2.imshow = original_imshow

        self.assertEqual(shown[0][0], line_task.YOLO_WINDOW_NAME)
        self.assertGreater(int(np.count_nonzero(shown[0][1])), 0)

    def test_yolo_target_filter_accepts_task_classes_only(self):
        detections = [
            line_task.YoloDetection(2, "Fire Building", 0.9,
                                    (10, 10, 80, 80), (100, 100, 3), 0.6),
            line_task.YoloDetection(3, "General population", 0.8,
                                    (10, 10, 80, 80), (100, 100, 3), 0.6),
            line_task.YoloDetection(7, "Recyclable waste", 0.85,
                                    (10, 10, 80, 80), (100, 100, 3), 0.6),
            line_task.YoloDetection(8, "unknown object", 0.99,
                                    (10, 10, 80, 80), (100, 100, 3), 0.6),
        ]

        self.assertEqual(
            [item.class_name for item in detections if item.target],
            ["Fire Building", "General population", "Recyclable waste"],
        )

    def test_yolo_center_band_uses_middle_eighty_percent(self):
        frame_shape = (100, 200, 3)

        center = line_task.YoloDetection(
            3, "Fire Building", 0.9, (25, 30, 45, 60), frame_shape, 0.8
        )
        edge = line_task.YoloDetection(
            3, "Fire Building", 0.9, (5, 30, 15, 60), frame_shape, 0.8
        )

        self.assertTrue(center.in_center)
        self.assertFalse(edge.in_center)

    def test_yolo_route_context_maps_task_segments(self):
        expected = {
            0: {"kind": "off"},
            1: {"kind": "street", "areas": ("C区", "P区")},
            2: {"kind": "street", "areas": ("A区", "S区")},
            3: {"kind": "off"},
            4: {"kind": "building", "area": "楼宇B"},
            5: {"kind": "building", "area": "楼宇C"},
            6: {"kind": "off"},
            7: {"kind": "building", "area": "楼宇A"},
            8: {"kind": "building", "area": "楼宇D"},
        }
        for task_index, context in expected.items():
            self.assertEqual(
                line_task.yolo_route_context(task_index, "FOLLOW"),
                context,
            )
        self.assertEqual(
            line_task.yolo_route_context(8, "FINAL_EXIT"),
            {"kind": "off"},
        )

    def test_yolo_model_profile_switches_after_third_right(self):
        self.assertEqual(line_task.yolo_model_profile(0), "street")
        self.assertEqual(line_task.yolo_model_profile(2), "street")
        self.assertEqual(line_task.yolo_model_profile(3), "building")
        self.assertEqual(line_task.yolo_model_profile(8), "building")

    def test_third_right_completion_switches_before_fourth_left_follow(self):
        follower = self._follower(now=20.0)
        self._restore_rospy = follower._restore_rospy
        follower.state = "EXIT_ALIGN"
        follower.task_index = 2
        follower.turn_cmd = "right"
        switch_indices = []
        follower._switch_yolo_profile_if_needed = lambda: (
            switch_indices.append(follower.task_index) or True
        )

        follower._complete_intersection()

        self.assertEqual(switch_indices, [3])
        self.assertEqual(follower.task_index, 3)
        self.assertEqual(follower.turn_cmd, "left")
        self.assertEqual(follower.state, "FOLLOW")

    def test_yolo_target_classes_include_people_trash_and_buildings(self):
        self.assertEqual(
            line_task.YOLO_STREET_CLASS_NAMES,
            (
                "General population",
                "Medical population",
                "Recyclable waste",
                "other waste",
            ),
        )
        self.assertEqual(
            line_task.YOLO_BUILDING_CLASS_NAMES,
            (
                "Collapsed Building",
                "Electrical Fault Building",
                "Fire Building",
                "Toxic Gas-contaminated Building",
            ),
        )
        self.assertEqual(
            line_task.YOLO_CLASS_NAMES,
            (
                "Collapsed Building",
                "Electrical Fault Building",
                "Fire Building",
                "General population",
                "Medical population",
                "Toxic Gas-contaminated Building",
                "Recyclable waste",
                "other waste",
            ),
        )
        for class_name in (
                "Collapsed Building",
                "Electrical Fault Building",
                "Fire Building",
                "Toxic Gas-contaminated Building",
                "General population",
                "Medical population",
                "Recyclable waste",
                "other waste"):
            self.assertIn(class_name, line_task.YOLO_TARGET_CLASS_NAMES)
        self.assertNotIn("ID1", line_task.YOLO_CLASS_NAMES)

    def test_task_ledger_accepts_street_target_once_per_class_and_area(self):
        ledger = line_task.YoloTaskLedger()
        context = line_task.yolo_route_context(1, "FOLLOW")
        medical = line_task.YoloDetection(
            4, "Medical population", 0.9,
            (80, 30, 120, 80), (100, 200, 3), 0.8
        )
        repeat = line_task.YoloDetection(
            4, "Medical population", 0.88,
            (82, 31, 122, 81), (100, 200, 3), 0.8
        )
        trash = line_task.YoloDetection(
            7, "Recyclable waste", 0.91,
            (70, 20, 130, 90), (100, 200, 3), 0.8
        )

        first = ledger.select_event(context, [medical], 0.5)
        ledger.accept(first)
        self.assertEqual(first.kind, "street")
        self.assertEqual(first.area, "C区")
        self.assertEqual(first.class_name, "Medical population")

        self.assertIsNone(ledger.select_event(context, [repeat], 0.5))

        second = ledger.select_event(context, [trash], 0.5)
        ledger.accept(second)
        self.assertEqual(second.area, "P区")
        self.assertEqual(second.display_name, "可回收垃圾")

        general = line_task.YoloDetection(
            3, "General population", 0.95,
            (70, 20, 130, 90), (100, 200, 3), 0.8
        )
        self.assertIsNone(ledger.select_event(context, [general], 0.5))

    def test_people_majority_requires_stable_frames_for_both_classes(self):
        def detection(class_id, class_name, confidence, x1):
            return line_task.YoloDetection(
                class_id, class_name, confidence,
                (x1, 20, x1 + 20, 80), (100, 200, 3), 0.8,
            )

        context = line_task.yolo_route_context(1, "FOLLOW")
        medical_majority = [
            detection(1, "Medical population", 0.90, 70),
            detection(1, "Medical population", 0.85, 95),
            detection(0, "General population", 0.93, 120),
        ]
        ledger = line_task.YoloTaskLedger()
        self.assertIsNone(ledger.select_event(
            context, medical_majority, 0.5, people_stable_frames=3
        ))
        self.assertIsNone(ledger.select_event(
            context, medical_majority, 0.5, people_stable_frames=3
        ))
        event = ledger.select_event(
            context, medical_majority, 0.5, people_stable_frames=3
        )
        self.assertEqual(event.class_name, "Medical population")
        self.assertEqual(event.display_name, "医疗人群")

        general_majority = [
            detection(0, "General population", 0.86, 70),
            detection(0, "General population", 0.82, 95),
            detection(1, "Medical population", 0.95, 120),
        ]
        ledger = line_task.YoloTaskLedger()
        self.assertIsNone(ledger.select_event(
            context, general_majority, 0.5, people_stable_frames=2
        ))
        event = ledger.select_event(
            context, general_majority, 0.5, people_stable_frames=2
        )
        self.assertEqual(event.class_name, "General population")
        self.assertEqual(event.display_name, "普通人群")

    def test_people_tie_resets_stable_majority(self):
        context = line_task.yolo_route_context(1, "FOLLOW")
        medical = line_task.YoloDetection(
            1, "Medical population", 0.9,
            (70, 20, 90, 80), (100, 200, 3), 0.8,
        )
        general = line_task.YoloDetection(
            0, "General population", 0.9,
            (100, 20, 120, 80), (100, 200, 3), 0.8,
        )
        ledger = line_task.YoloTaskLedger()
        self.assertIsNone(ledger.select_event(
            context, [medical, medical], 0.5, people_stable_frames=2
        ))
        self.assertIsNone(ledger.select_event(
            context, [medical, general], 0.5, people_stable_frames=2
        ))
        self.assertIsNone(ledger.select_event(
            context, [medical, medical], 0.5, people_stable_frames=2
        ))
        event = ledger.select_event(
            context, [medical, medical], 0.5, people_stable_frames=2
        )
        self.assertEqual(event.class_name, "Medical population")

    def test_trash_and_building_use_fixed_065_confidence(self):
        street_context = line_task.yolo_route_context(1, "FOLLOW")
        low_trash = line_task.YoloDetection(
            2, "other waste", 0.64,
            (70, 20, 110, 80), (100, 200, 3), 0.8,
        )
        high_trash = line_task.YoloDetection(
            2, "other waste", 0.66,
            (70, 20, 110, 80), (100, 200, 3), 0.8,
        )
        ledger = line_task.YoloTaskLedger()
        self.assertIsNone(ledger.select_event(
            street_context, [low_trash], 0.60,
            trash_confidence=line_task.YOLO_TRASH_CONFIDENCE,
        ))
        self.assertEqual(ledger.select_event(
            street_context, [high_trash], 0.60,
            trash_confidence=line_task.YOLO_TRASH_CONFIDENCE,
        ).class_name, "other waste")

        building_context = line_task.yolo_route_context(4, "FOLLOW")
        low_building = line_task.YoloDetection(
            0, "Collapsed Building", 0.64,
            (70, 20, 110, 80), (100, 200, 3), 0.8,
        )
        high_building = line_task.YoloDetection(
            0, "Collapsed Building", 0.66,
            (70, 20, 110, 80), (100, 200, 3), 0.8,
        )
        ledger = line_task.YoloTaskLedger()
        self.assertIsNone(ledger.select_event(
            building_context, [low_building], 0.60,
            building_confidence=line_task.YOLO_BUILDING_CONFIDENCE,
        ))
        self.assertEqual(ledger.select_event(
            building_context, [high_building], 0.60,
            building_confidence=line_task.YOLO_BUILDING_CONFIDENCE,
        ).class_name, "Collapsed Building")

    def test_yolo_stop_return_to_follow_starts_four_second_guard(self):
        follower = self._follower(now=20.0)
        self._restore_rospy = follower._restore_rospy
        follower.state = "YOLO_STOP"

        follower._set_state("FOLLOW")

        self.assertEqual(follower.yolo_accept_after, 24.0)
        polled = []
        follower._poll_yolo_detections = lambda: polled.append(True) or (
            True, []
        )
        stopped = follower._maybe_enter_yolo_stop(
            types.SimpleNamespace(valid=True)
        )
        self.assertFalse(stopped)
        self.assertEqual(polled, [])

    def test_task_ledger_accepts_building_area_and_class_once(self):
        ledger = line_task.YoloTaskLedger()
        context = line_task.yolo_route_context(4, "FOLLOW")
        fire = line_task.YoloDetection(
            2, "Fire Building", 0.9,
            (80, 30, 120, 80), (100, 200, 3), 0.8
        )
        event = ledger.select_event(context, [fire], 0.5)
        ledger.accept(event)

        self.assertEqual(event.kind, "building")
        self.assertEqual(event.area, "楼宇B")
        self.assertEqual(event.display_name, "火灾楼宇")
        self.assertIsNone(ledger.select_event(context, [fire], 0.5))

        same_class_new_area = ledger.select_event(
            line_task.yolo_route_context(5, "FOLLOW"), [fire], 0.5
        )
        self.assertIsNone(same_class_new_area)

    def test_task_ledger_ignores_off_route_and_non_center_targets(self):
        ledger = line_task.YoloTaskLedger()
        fire = line_task.YoloDetection(
            2, "Fire Building", 0.9,
            (0, 30, 20, 80), (100, 200, 3), 0.5
        )
        self.assertIsNone(
            ledger.select_event({"kind": "off"}, [fire], 0.5)
        )
        self.assertIsNone(
            ledger.select_event(
                line_task.yolo_route_context(4, "FOLLOW"), [fire], 0.5
            )
        )

    def test_prepare_yolo_save_dir_clears_previous_images(self):
        root = tempfile.mkdtemp()
        try:
            old_path = os.path.join(root, "old.jpg")
            nested = os.path.join(root, "nested")
            os.mkdir(nested)
            open(old_path, "w").close()
            open(os.path.join(nested, "x.txt"), "w").close()
            follower = self._follower(now=30.0)
            self._restore_rospy = follower._restore_rospy
            follower.yolo_save_dir = root

            follower._prepare_yolo_save_dir()

            self.assertTrue(os.path.isdir(root))
            self.assertEqual(os.listdir(root), [])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_report_yolo_task_event_logs_and_saves_people_boxed_image(self):
        root = tempfile.mkdtemp()
        try:
            follower = self._follower(now=30.0)
            self._restore_rospy = follower._restore_rospy
            follower.yolo_save_dir = root
            follower.task_ledger = line_task.YoloTaskLedger()
            detection = line_task.YoloDetection(
                5, "Medical population", 0.9,
                (20, 20, 80, 80), (100, 120, 3), 0.8
            )
            event = line_task.YoloTaskEvent(
                "street", "C区", "Medical population", "医疗人群", detection
            )
            follower.task_ledger.accept(event)
            with follower.yolo_lock:
                follower.yolo_latest_frame = np.zeros((100, 120, 3), dtype=np.uint8)
            logs = []
            original_loginfo = line_task.rospy.loginfo
            line_task.rospy.loginfo = lambda message, *args: logs.append(
                message % args if args else message
            )
            try:
                follower._report_yolo_task_event([detection])
            finally:
                line_task.rospy.loginfo = original_loginfo

            self.assertIn("C区识别到医疗人群", logs)
            files = os.listdir(root)
            self.assertEqual(files, ["01_C区_医疗人群.jpg"])
            saved = line_task.cv2.imread(os.path.join(root, files[0]))
            self.assertGreater(int(np.count_nonzero(saved)), 0)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_saved_yolo_image_omits_center_band_lines(self):
        root = tempfile.mkdtemp()
        try:
            follower = self._follower(now=30.0)
            self._restore_rospy = follower._restore_rospy
            follower.yolo_save_dir = root
            follower.yolo_center_band_ratio = 0.2
            follower.task_ledger = line_task.YoloTaskLedger()
            detection = line_task.YoloDetection(
                5, "Medical population", 0.9,
                (5, 20, 15, 80), (100, 120, 3), 1.0
            )
            follower.task_ledger.accept(line_task.YoloTaskEvent(
                "street", "C区", "Medical population", "医疗人群", detection
            ))
            with follower.yolo_lock:
                follower.yolo_latest_frame = np.zeros((100, 120, 3), dtype=np.uint8)

            follower._report_yolo_task_event([detection])

            saved = line_task.cv2.imread(os.path.join(
                root, "01_C区_医疗人群.jpg"
            ))
            cyan = (
                (saved[:, :, 0] > 150)
                & (saved[:, :, 1] > 150)
                & (saved[:, :, 2] < 150)
            )
            self.assertLess(int(np.max(np.sum(cyan, axis=0))), 20)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_saved_yolo_image_falls_back_to_trigger_box_when_report_misses(self):
        root = tempfile.mkdtemp()
        try:
            follower = self._follower(now=30.0)
            self._restore_rospy = follower._restore_rospy
            follower.yolo_save_dir = root
            follower.task_ledger = line_task.YoloTaskLedger()
            trigger = line_task.YoloDetection(
                2, "Fire Building", 0.9,
                (20, 20, 80, 80), (100, 120, 3), 0.8
            )
            follower.task_ledger.accept(line_task.YoloTaskEvent(
                "building", "楼宇B", "Fire Building", "火灾楼宇", trigger
            ))
            with follower.yolo_lock:
                follower.yolo_latest_frame = np.zeros((100, 120, 3), dtype=np.uint8)

            follower._report_yolo_task_event([])

            saved = line_task.cv2.imread(os.path.join(root, "01_楼宇B_火灾楼宇.jpg"))
            green = np.all(saved == np.array([0, 255, 0], dtype=np.uint8), axis=2)
            self.assertGreater(int(np.count_nonzero(green)), 0)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_report_yolo_task_event_logs_trash_and_building(self):
        root = tempfile.mkdtemp()
        try:
            follower = self._follower(now=30.0)
            self._restore_rospy = follower._restore_rospy
            follower.yolo_save_dir = root
            follower.task_ledger = line_task.YoloTaskLedger()
            trash = line_task.YoloDetection(
                6, "other waste", 0.9,
                (20, 20, 80, 80), (100, 120, 3), 0.8
            )
            event = line_task.YoloTaskEvent(
                "street", "P区", "other waste", "其他垃圾", trash
            )
            building = line_task.YoloDetection(
                0, "Collapsed Building", 0.9,
                (20, 20, 80, 80), (100, 120, 3), 0.8
            )
            follower.task_ledger.accept(event)
            with follower.yolo_lock:
                follower.yolo_latest_frame = np.zeros((100, 120, 3), dtype=np.uint8)
            logs = []
            original_loginfo = line_task.rospy.loginfo
            line_task.rospy.loginfo = lambda message, *args: logs.append(
                message % args if args else message
            )
            try:
                follower._report_yolo_task_event([trash])
                follower.task_ledger.pending_event = line_task.YoloTaskEvent(
                    "building", "楼宇B", "Collapsed Building", "坍塌楼宇",
                    building
                )
                with follower.yolo_lock:
                    follower.yolo_latest_frame = np.zeros((100, 120, 3), dtype=np.uint8)
                follower._report_yolo_task_event([building])
            finally:
                line_task.rospy.loginfo = original_loginfo

            self.assertIn("P区检测到垃圾桶：其他垃圾", logs)
            self.assertIn("楼宇B检测到坍塌楼宇", logs)
            self.assertIn("01_P区_其他垃圾.jpg", os.listdir(root))
            self.assertIn("02_楼宇B_坍塌楼宇.jpg", os.listdir(root))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_street_task_detection_enters_yolo_stop_with_area_event(self):
        follower = self._follower(now=20.0)
        self._restore_rospy = follower._restore_rospy
        follower.task_index = 1
        follower.yolo_ready = True
        follower.yolo_segment_key = ("street", ("C区", "P区"))
        follower.yolo_segment_start_seq = 0
        with follower.yolo_lock:
            follower.yolo_latest_seq = 1
        detection = line_task.YoloDetection(
            5, "Medical population", 0.9,
            (80, 30, 120, 80), (100, 200, 3), 0.8
        )
        follower._poll_yolo_detections = lambda: (True, [detection])

        stopped = follower._maybe_enter_yolo_stop(
            types.SimpleNamespace(valid=True, dual_rows=0)
        )

        self.assertTrue(stopped)
        self.assertEqual(follower.state, "YOLO_STOP")
        self.assertEqual(follower.yolo_stop_detection, detection)
        self.assertEqual(follower.task_ledger.pending_event.area, "C区")
        self.assertEqual(
            follower.task_ledger.pending_event.class_name,
            "Medical population",
        )

    def test_yolo_stop_disabled_only_prints_without_state_change(self):
        follower = self._follower(now=20.0, yolo_stop_enabled=False)
        self._restore_rospy = follower._restore_rospy
        follower.task_index = 4
        follower.yolo_ready = True
        follower.yolo_segment_key = ("building", "楼宇B")
        follower.yolo_segment_start_seq = 0
        with follower.yolo_lock:
            follower.yolo_latest_seq = 1
        detection = line_task.YoloDetection(
            2, "Fire Building", 0.9,
            (80, 30, 120, 80), (100, 200, 3), 0.8
        )
        follower._poll_yolo_detections = lambda: (True, [detection])
        observation = types.SimpleNamespace(valid=True, dual_rows=2)

        stopped = follower._maybe_enter_yolo_stop(observation)

        self.assertFalse(stopped)
        self.assertEqual(follower.state, "FOLLOW")
        self.assertIsNone(follower.task_ledger.pending_event)

    def test_building_task_detection_enters_yolo_stop_with_building_area(self):
        follower = self._follower(now=20.0)
        self._restore_rospy = follower._restore_rospy
        follower.task_index = 4
        follower.yolo_ready = True
        follower.yolo_segment_key = ("building", "楼宇B")
        follower.yolo_segment_start_seq = 0
        with follower.yolo_lock:
            follower.yolo_latest_seq = 1
        detection = line_task.YoloDetection(
            2, "Fire Building", 0.9,
            (80, 30, 120, 80), (100, 200, 3), 0.8
        )
        follower._poll_yolo_detections = lambda: (True, [detection])

        stopped = follower._maybe_enter_yolo_stop(
            types.SimpleNamespace(valid=True, dual_rows=2)
        )

        self.assertTrue(stopped)
        self.assertEqual(follower.task_ledger.pending_event.area, "楼宇B")
        self.assertEqual(follower.task_ledger.pending_event.display_name, "火灾楼宇")

    def test_building_detection_uses_lower_confidence_than_street(self):
        follower = self._follower(now=20.0)
        self._restore_rospy = follower._restore_rospy
        follower.yolo_confidence = 0.60
        follower.yolo_building_confidence = 0.40
        low_building = line_task.YoloDetection(
            2, "Fire Building", 0.45,
            (80, 30, 120, 80), (100, 200, 3), 0.8
        )
        low_people = line_task.YoloDetection(
            4, "Medical population", 0.45,
            (80, 30, 120, 80), (100, 200, 3), 0.8
        )

        follower.task_index = 4
        building_event = follower._select_yolo_stop_event([low_building])
        follower.task_index = 1
        street_event = follower._select_yolo_stop_event([low_people])

        self.assertIsNotNone(building_event)
        self.assertEqual(building_event.kind, "building")
        self.assertIsNone(street_event)

    def test_building_profile_uses_lower_confidence_and_building_classes(self):
        follower = self._follower(now=20.0)
        self._restore_rospy = follower._restore_rospy
        follower.yolo_confidence = 0.60
        follower.yolo_building_confidence = 0.40
        kwargs_seen = []

        class FakeDetector(object):
            def __init__(self, *args, **kwargs):
                kwargs_seen.append(kwargs)

        original_detector = line_task.YoloObstacleDetector
        line_task.YoloObstacleDetector = FakeDetector
        try:
            follower._create_yolo_detector("building")
        finally:
            line_task.YoloObstacleDetector = original_detector

        self.assertEqual(kwargs_seen[0]["confidence"], 0.40)
        self.assertEqual(kwargs_seen[0]["class_names"],
                         line_task.YOLO_BUILDING_CLASS_NAMES)

    def test_model_switch_releases_street_model_and_warms_building_model(self):
        follower = self._follower(now=20.0)
        self._restore_rospy = follower._restore_rospy
        events = []

        class OldDetector(object):
            def close(self):
                events.append("street_closed")

        class FakeDetector(object):
            backend_name = "opencv-dnn-onnx"

            def __init__(self, model_path, **kwargs):
                self.model_path = model_path
                self.names = dict(enumerate(kwargs["class_names"]))
                events.append(("created", model_path, kwargs["class_names"]))

            def load(self):
                events.append("building_loaded")

            def detect(self, frame):
                events.append("building_warmed")
                return []

            def close(self):
                events.append("building_closed")

        follower.task_index = 3
        follower.yolo_active_profile = "street"
        follower.yolo_detector = OldDetector()
        follower.yolo_camera = types.SimpleNamespace(
            read=lambda timeout=0.0: (
                True, np.zeros((240, 320, 3), dtype=np.uint8)
            )
        )
        original_detector = line_task.YoloObstacleDetector
        line_task.YoloObstacleDetector = FakeDetector
        try:
            switched = follower._switch_yolo_profile_if_needed()
        finally:
            line_task.YoloObstacleDetector = original_detector

        self.assertTrue(switched)
        self.assertEqual(follower.yolo_active_profile, "building")
        self.assertEqual(follower.yolo_model_path, "/tmp/building.onnx")
        self.assertEqual(follower.yolo_class_names,
                         line_task.YOLO_BUILDING_CLASS_NAMES)
        self.assertTrue(follower.yolo_ready)
        self.assertEqual(events[0], "street_closed")
        self.assertIn("building_loaded", events)
        self.assertIn("building_warmed", events)

    def test_disabled_yolo_route_does_not_poll_or_stop(self):
        follower = self._follower(now=20.0)
        self._restore_rospy = follower._restore_rospy
        follower.task_index = 3
        calls = []
        follower._poll_yolo_detections = lambda: calls.append(True) or (True, [])

        stopped = follower._maybe_enter_yolo_stop(
            types.SimpleNamespace(valid=True, dual_rows=2)
        )

        self.assertFalse(stopped)
        self.assertEqual(calls, [])

    def test_yolo_worker_skips_inference_on_disabled_route(self):
        follower = self._follower(now=20.0)
        self._restore_rospy = follower._restore_rospy
        follower.task_index = 3
        follower.state = "FOLLOW"

        self.assertFalse(follower._yolo_inference_allowed())

        follower.task_index = 4
        self.assertTrue(follower._yolo_inference_allowed())

    def test_enabled_yolo_route_waits_for_fresh_segment_result(self):
        follower = self._follower(now=20.0)
        self._restore_rospy = follower._restore_rospy
        follower.task_index = 4
        follower.state = "FOLLOW"
        follower.yolo_ready = True
        follower.yolo_segment_key = None
        follower.yolo_segment_start_seq = 3
        with follower.yolo_lock:
            follower.yolo_latest_seq = 3

        ready = follower._wait_for_yolo_ready_if_needed()

        self.assertFalse(ready)
        self.assertEqual(follower.published[-1], (0, 0))

        with follower.yolo_lock:
            follower.yolo_latest_seq = 4
        self.assertTrue(follower._wait_for_yolo_ready_if_needed())

    def test_yolo_unknown_detection_is_silent_and_does_not_stop(self):
        follower = self._follower(now=20.0)
        self._restore_rospy = follower._restore_rospy
        detection = line_task.YoloDetection(
            8, "unknown object", 0.9, (80, 30, 100, 60), (100, 200, 3), 0.8
        )
        logs = []
        original_loginfo = line_task.rospy.loginfo
        line_task.rospy.loginfo = lambda message, *args: logs.append(
            message % args if args else message
        )
        try:
            follower._poll_yolo_detections = lambda: (True, [detection])
            observation = types.SimpleNamespace(valid=True, dual_rows=2)

            stopped = follower._maybe_enter_yolo_stop(observation)
        finally:
            line_task.rospy.loginfo = original_loginfo

        self.assertFalse(stopped)
        self.assertEqual(follower.state, "FOLLOW")
        self.assertEqual(logs, [])

    def test_yolo_stop_waits_until_stop_time(self):
        follower = self._follower(now=20.5)
        self._restore_rospy = follower._restore_rospy
        follower.state = "YOLO_STOP"
        follower.state_started = 20.0
        follower.yolo_stop_time = 1.0

        handled = follower._handle_yolo_stop(20.5)

        self.assertTrue(handled)
        self.assertEqual(follower.state, "YOLO_STOP")
        self.assertEqual(follower.published[-1], (0, 0))

    def test_yolo_stop_waits_for_new_detection_before_resuming(self):
        follower = self._follower(now=21.2)
        self._restore_rospy = follower._restore_rospy
        follower.state = "YOLO_STOP"
        follower.state_started = 20.0
        follower.yolo_stop_time = 1.0

        handled = follower._handle_yolo_stop(21.2)

        self.assertTrue(handled)
        self.assertEqual(follower.state, "YOLO_STOP")

    def test_yolo_stop_prints_new_detection_then_resumes_after_stop_time(self):
        follower = self._follower(now=21.2)
        self._restore_rospy = follower._restore_rospy
        delattr(follower, "_poll_yolo_detections")
        follower.state = "YOLO_STOP"
        follower.state_started = 20.0
        follower.yolo_stop_time = 1.0
        follower.yolo_detector = types.SimpleNamespace()
        follower.yolo_camera = types.SimpleNamespace()
        detection = line_task.YoloDetection(
            2, "Fire Building", 0.9, (80, 30, 100, 60), (100, 200, 3), 0.8
        )
        reports = []
        with follower.yolo_lock:
            follower.yolo_latest_seq = 1
            follower.yolo_read_seq = 0
            follower.yolo_latest_detections = [detection]
        follower._report_yolo_task_event = lambda detections: reports.append(
            detections
        )

        handled = follower._handle_yolo_stop(21.2)

        self.assertTrue(handled)
        self.assertEqual(follower.state, "FOLLOW")
        self.assertEqual(reports, [[detection]])
        self.assertTrue(follower.yolo_stop_reported)

    def test_ledger_stops_same_building_class_only_once(self):
        follower = self._follower(now=30.0)
        self._restore_rospy = follower._restore_rospy
        follower.task_index = 4
        follower.yolo_ready = True
        follower.yolo_segment_key = ("building", "楼宇B")
        follower.yolo_segment_start_seq = 0
        with follower.yolo_lock:
            follower.yolo_latest_seq = 1
        first = line_task.YoloDetection(
            2, "Fire Building", 0.9, (80, 30, 100, 60), (100, 200, 3), 0.8
        )
        second = line_task.YoloDetection(
            2, "Fire Building", 0.88, (82, 31, 102, 61), (100, 200, 3), 0.8
        )
        observation = types.SimpleNamespace(valid=True, dual_rows=2)

        follower._poll_yolo_detections = lambda: (True, [first])
        self.assertTrue(follower._maybe_enter_yolo_stop(observation))
        follower.state = "FOLLOW"
        follower._poll_yolo_detections = lambda: (True, [second])
        stopped = follower._maybe_enter_yolo_stop(observation)

        self.assertFalse(stopped)
        self.assertEqual(follower.state, "FOLLOW")
        self.assertIn("Fire Building", follower.task_ledger.building_seen_classes)

    def test_non_follow_state_does_not_poll_yolo(self):
        follower = self._follower(now=30.0)
        self._restore_rospy = follower._restore_rospy
        follower.state = "MANEUVER"
        calls = []
        follower._poll_yolo_detections = lambda: calls.append(True) or (
            True, []
        )
        observation = types.SimpleNamespace(valid=True, dual_rows=2)

        stopped = follower._maybe_enter_yolo_stop(observation)

        self.assertFalse(stopped)
        self.assertEqual(calls, [])

    def test_crosswalk_entry_has_priority_over_yolo_stop(self):
        follower = self._follower(now=40.0)
        self._restore_rospy = follower._restore_rospy
        follower.state = "FOLLOW"
        follower.turn_cmd = "right"
        follower.maneuver_phase = "NONE"
        follower.stop_hits = line_task.STOP_STABLE_FRAMES - 1
        follower.entry_accept_after = 0.0
        follower.bridge = types.SimpleNamespace(reset=lambda lane_width: None)
        follower.lane_width = 620.0
        follower.crosswalk = types.SimpleNamespace(
            lock_current_bar=lambda: True,
            unlock_bar=lambda: None,
        )
        follower.process_width = 640
        follower.vision = types.SimpleNamespace(
            apply=lambda frame: np.zeros((480, 640), dtype=np.uint8)
        )
        follower.lanes = types.SimpleNamespace(
            points=lambda binary, center_x=None: ([], []),
            observe=lambda binary, lane_width: types.SimpleNamespace(
                valid=True,
                dual_rows=2,
                center_x=320.0,
                measured_width=None,
                left_points=[],
                right_points=[],
                center_points=[],
                virtual_left_points=[],
                virtual_right_points=[],
                follow_side=None,
            ),
        )
        follower._update_lane_width = lambda observation, frame_width: None
        follower.crosswalk = types.SimpleNamespace(
            lock_current_bar=lambda: True,
            unlock_bar=lambda: None,
            detect=lambda binary, lane_points=None,
            allow_strong_lane_override=False: types.SimpleNamespace(
                candidate=True,
                stop_polygon=None,
                stripe_polygons=[],
                loose_polygons=[],
                confidence=1.0,
                stop_angle=None,
                tracking_angle=None,
                tracking_polygon=None,
            ),
        )
        calls = []
        follower._maybe_enter_yolo_stop = lambda observation: calls.append(True) or True
        follower.debug_view = False
        follower._resize = lambda frame: frame

        follower.process(np.zeros((480, 640, 3), dtype=np.uint8))

        self.assertEqual(follower.state, "APPROACH")
        self.assertEqual(calls, [])

if __name__ == "__main__":
    unittest.main()
