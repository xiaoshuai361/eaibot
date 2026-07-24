#!/usr/bin/env python
# coding=utf-8
"""实车循迹 + 斑马线前停车 + 路口单边补线。

设计目标：
1. 代码放到车上 /home/eaibot/robocom_ws/src/line_cy.py 后可直接运行。
2. 默认 dry_run/detect_only，只看处理图，不控制底盘。
3. 高分辨率画面默认缩到 640 宽再识别，避免 PID 像素尺度变化。
"""

import threading
import time

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Twist


# 常用调车参数：详细说明见 /home/eaibot/zcy/循迹操作.md
LANE_CAM_INDEX = 2
PROCESS_WIDTH = 640
DRY_RUN = True
DETECT_ONLY = True
DEBUG_VIEW = True
RAW_VIEW = False

LINEAR_SPEED = 0.10
SINGLE_LINE_SPEED = 0.08
MAX_ANGULAR = 0.6
SINGLE_LINE_MIN_ANGULAR = 0.15
ANGULAR_SMOOTH_KEEP = 0.70
ANGULAR_STEP_LIMIT = 0.05

PID_KP_SMALL = 0.0028
PID_KD_SMALL = 0.0008
PID_KP_BIG = 0.03
PID_KD_BIG = 0.002
PID_KI = 0
DEV_THRESHOLD = 100

ROI_TOP_RATIO = 0.24
ROI_BOTTOM_RATIO = 0.76
SINGLE_CENTER_FACTOR = 0.6

BLACK_V_MAX = 80
ADAPTIVE_BLOCK_SIZE = 31
ADAPTIVE_C = 5
BLUR_KERNEL_SIZE = 5
MORPH_KERNEL_SIZE = 3

STOP_CONFIDENCE_MIN = 0.68
STOP_FRONT_CENTER_MARGIN_RATIO = 0.28
STOP_STABLE_FRAMES = 3
STOP_HOLD_TIME = 1.0
STOP_COOLDOWN_TIME = 5.0

APPROACH_CROSSWALK_SPEED = 0.06
ALIGN_TRIGGER_Y_RATIO = 0.82
ALIGN_KP = 0.025
ALIGN_MAX_ANGULAR = 0.35
ALIGN_MIN_ANGULAR = 0.08
ALIGN_ANGLE_TOLERANCE_DEG = 3.0
ALIGN_STABLE_FRAMES = 5
ALIGN_TIMEOUT = 5.0
ALIGN_ANGULAR_SIGN = 1.0
CROSSWALK_LOST_FRAMES = 6

SIDE_FOLLOW_SPEED = 0.10
ENTER_INTERSECTION_STRAIGHT_TIME = 0.6
INTERSECTION_MIN_TIME = 1.2
INTERSECTION_MAX_TIME = 15.0
RECOVER_DUAL_FRAMES = 10
CROSSWALK_CLEAR_CONFIDENCE = 0.45
CROSSWALK_TRACK_CONFIDENCE = 0.52
CROSSWALK_CLEAR_FRAMES = 8
MANEUVER_CROSSWALK_MEMORY_FRAMES = 16
LEFT_TURN_BIAS = 0.12
RIGHT_TURN_BIAS = 0.12
STRAIGHT_BIAS = 0.0

# 算法内部阈值，通常不用调。
CAMERA_BACKEND = "v4l2"
CAMERA_STARTUP_WAIT = 3.0
DEBUG_MAX_WIDTH = 960
DESTROY_WINDOWS_ON_EXIT = False
RAW_VIEW_WINDOW = "line_cy_raw"

SEARCH_SPEED = 0.08
SEARCH_ANGULAR_LIMIT = 0.25

BLACK_HSV_LOWER = np.array([0, 0, 0], dtype=np.uint8)
BLACK_HSV_UPPER = np.array([180, 255, BLACK_V_MAX], dtype=np.uint8)
GROUP_GAP = 20
MIN_GROUP_WIDTH = 3
MAX_GROUP_WIDTH_RATIO = 0.18
SCAN_ROWS = 6

LANE_WIDTH_PIXELS = 0.0
DEFAULT_LANE_WIDTH_RATIO = 0.56
LANE_WIDTH_MIN_RATIO = 0.35
LANE_WIDTH_MAX_RATIO = 0.85
PAIR_WIDTH_TOLERANCE = 0.50
PAIR_MAX_GAP_RATIO = 0.96
PAIR_CENTER_JUMP_RATIO = 0.30
OUTSIDE_MARGIN_RATIO = 0.10
SINGLE_WIDTH_FLOOR_RATIO = 0.52
LANE_CONTINUITY_JUMP_RATIO = 0.12
SINGLE_SIDE_HINT_FRAMES = 12
SINGLE_TURN_CONFIRM_FRAMES = 4
SINGLE_TURN_RELEASE_FRAMES = 6
KALMAN_FAIL_MAX = 8


def clamp(value, low, high):
    return max(low, min(high, value))


def long_edge_angle_deg(points):
    """返回旋转矩形最长边相对画面水平线的角度，范围为 [-90, 90]。"""
    polygon = np.asarray(points, dtype=np.float32).reshape(4, 2)
    edges = np.roll(polygon, -1, axis=0) - polygon
    lengths = np.sum(edges * edges, axis=1)
    dx, dy = edges[int(np.argmax(lengths))]
    angle = float(np.degrees(np.arctan2(float(dy), float(dx))))
    while angle > 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    return angle


def undirected_angle_delta_deg(a, b):
    """比较线段方向，不区分 180 度反向。"""
    delta = abs(float(a) - float(b))
    while delta > 90.0:
        delta = abs(delta - 180.0)
    return delta


def alignment_angular(angle_deg, kp, min_angular, max_angular, direction_sign):
    """把横条角度误差转换成原地摆正角速度。"""
    if abs(angle_deg) < 1e-6:
        return 0.0
    angular = -float(angle_deg) * float(kp) * float(direction_sign)
    magnitude = clamp(abs(angular), abs(float(min_angular)), abs(float(max_angular)))
    return magnitude if angular > 0.0 else -magnitude


def motion_commands_enabled(dry_run, detect_only):
    return not dry_run and not detect_only


def capped_speed(default_speed, requested_speed):
    return float(default_speed) if requested_speed is None else min(float(default_speed), float(requested_speed))


def crosswalk_next_state(
    state, entry_ready, candidate, bottom_ratio, trigger_ratio, lost_count, lost_limit, timed_out=False
):
    """纯状态转换；实际速度发布和计时保护由 LaneFollower 负责。"""
    if state == "FOLLOW_LINE" and entry_ready:
        return "APPROACH_CROSSWALK"
    if state == "APPROACH_CROSSWALK":
        if lost_count > lost_limit:
            return "CROSSWALK_WAIT"
        if candidate and bottom_ratio >= trigger_ratio:
            return "ALIGN_STOPLINE"
    if state == "ALIGN_STOPLINE" and (lost_count > lost_limit or timed_out):
        return "CROSSWALK_WAIT"
    return state


def find_contours(binary):
    """兼容 OpenCV 3/4 的 findContours 返回值。"""
    result = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return result[0] if len(result) == 2 else result[1]


class CameraReader:
    """后台读摄像头，避免主线程卡在 cap.read() 后 Ctrl-C 不响应。"""
    def __init__(self, camera_index, backend):
        self.cap = self._open(camera_index, backend)
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_ok = False
        self.latest_seq = 0
        self.last_read_seq = -1
        self.running = False
        self.thread = None
        if self.cap is not None and self.cap.isOpened():
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            self.running = True
            self.thread = threading.Thread(target=self._loop)
            self.thread.daemon = True
            self.thread.start()
    def _open(self, camera_index, backend):
        if backend == "v4l2" and hasattr(cv2, "CAP_V4L2"):
            cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
            if cap.isOpened():
                return cap
        return cv2.VideoCapture(camera_index)
    def isOpened(self):
        return self.cap is not None and self.cap.isOpened()
    def _loop(self):
        while self.running and not rospy.is_shutdown():
            ok, frame = self.cap.read()
            with self.lock:
                self.latest_ok = bool(ok)
                if ok:
                    self.latest_frame = frame
                    self.latest_seq += 1
            if not ok:
                time.sleep(0.02)
    def read(self, timeout=0.5):
        end = time.time() + timeout
        while not rospy.is_shutdown() and time.time() < end:
            with self.lock:
                if self.latest_ok and self.latest_frame is not None and self.latest_seq != self.last_read_seq:
                    frame = self.latest_frame.copy()
                    self.last_read_seq = self.latest_seq
                    return True, frame
            time.sleep(0.01)
        return False, None
    def release(self):
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(0.8)
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class PIDController:
    def __init__(self, kp, ki, kd, output_limits):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None
    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None
    def update(self, deviation):
        now = rospy.get_time()
        dt = 0.05 if self.last_time is None else max(now - self.last_time, 0.001)
        error = -float(deviation)
        self.integral = clamp(self.integral + error * dt, -300.0, 300.0)
        derivative = (error - self.last_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = clamp(output, self.output_limits[0], self.output_limits[1])
        self.last_error = error
        self.last_time = now
        return output


class LineVision:
    """黑线提取、车道中心计算、停车线检测和调试信息生成。"""
    def __init__(self):
        self.lower_black = BLACK_HSV_LOWER
        self.upper_black = BLACK_HSV_UPPER
        self.blur_kernel_size = BLUR_KERNEL_SIZE
        self.adaptive_block_size = ADAPTIVE_BLOCK_SIZE
        self.adaptive_c = ADAPTIVE_C
        self.morph_kernel_size = MORPH_KERNEL_SIZE
        self.group_gap = GROUP_GAP
        self.min_group_width = MIN_GROUP_WIDTH
        self.max_group_width_ratio = MAX_GROUP_WIDTH_RATIO
        self.roi_top_ratio = ROI_TOP_RATIO
        self.roi_bottom_ratio = ROI_BOTTOM_RATIO
        self.scan_rows = SCAN_ROWS
        self.width_min_ratio = LANE_WIDTH_MIN_RATIO
        self.width_max_ratio = LANE_WIDTH_MAX_RATIO
        self.default_width_ratio = DEFAULT_LANE_WIDTH_RATIO
        self.pair_width_tolerance = PAIR_WIDTH_TOLERANCE
        self.pair_max_gap_ratio = PAIR_MAX_GAP_RATIO
        self.pair_center_jump_ratio = PAIR_CENTER_JUMP_RATIO
        self.outside_margin_ratio = OUTSIDE_MARGIN_RATIO
        self.single_center_factor = SINGLE_CENTER_FACTOR
        self.single_width_floor_ratio = SINGLE_WIDTH_FLOOR_RATIO
        self.stop_confidence_min = STOP_CONFIDENCE_MIN
        self.stop_front_center_margin_ratio = STOP_FRONT_CENTER_MARGIN_RATIO
    def mask_black(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, self.lower_black, self.upper_black)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.adaptive_block_size,
            self.adaptive_c,
        )
        binary = cv2.bitwise_and(color_mask, adaptive)
        kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary
    def expected_width(self, width, lane_width_estimate, lane_width_pixels):
        if lane_width_pixels > 0:
            value = lane_width_pixels
        elif lane_width_estimate is not None:
            value = lane_width_estimate
        else:
            value = width * self.default_width_ratio
        return clamp(float(value), width * self.width_min_ratio, width * self.width_max_ratio)
    def row_groups(self, row_pixels, width):
        if len(row_pixels) == 0:
            return []
        groups = []
        start = int(row_pixels[0])
        last = start
        for pixel in row_pixels[1:]:
            pixel = int(pixel)
            if pixel - last <= self.group_gap:
                last = pixel
                continue
            self._append_group(groups, start, last, width)
            start = last = pixel
        self._append_group(groups, start, last, width)
        return groups
    def _append_group(self, groups, left, right, width):
        group_width = right - left + 1
        if self.min_group_width <= group_width <= width * self.max_group_width_ratio:
            groups.append((left, right, (left + right) * 0.5, group_width))
    def best_pair(self, groups, width, expected_width, prior_center):
        if len(groups) < 2:
            return None
        min_gap = max(width * 0.18, expected_width * (1.0 - self.pair_width_tolerance))
        max_gap = min(width * self.pair_max_gap_ratio, expected_width * (1.0 + self.pair_width_tolerance))
        prior = width * 0.5 if prior_center is None else float(prior_center)
        best = None
        ordered = sorted(groups, key=lambda item: item[2])
        for i in range(len(ordered) - 1):
            for j in range(i + 1, len(ordered)):
                left, right = ordered[i], ordered[j]
                inner_gap = right[0] - left[1]
                if inner_gap < min_gap or inner_gap > max_gap:
                    continue
                pair_center = (left[1] + right[0]) * 0.5
                center_jump = abs(pair_center - prior) / float(width)
                jump_penalty = 0.0
                if center_jump > self.pair_center_jump_ratio:
                    jump_penalty = 0.8
                width_score = abs(inner_gap - expected_width) / max(expected_width, 1.0)
                center_score = center_jump
                edge_penalty = 0.15 if left[2] < width * 0.05 or right[2] > width * 0.95 else 0.0
                score = width_score * 0.55 + center_score * 0.35 + edge_penalty + jump_penalty
                if best is None or score < best[0]:
                    best = (score, left, right)
        if best is None:
            return None
        return best[1], best[2]
    def filter_outside(self, groups, pair, width, expected_width):
        if pair is None:
            return groups, []
        left, right = pair
        margin = max(width * 0.03, expected_width * self.outside_margin_ratio)
        min_x = left[2] - margin
        max_x = right[2] + margin
        kept = [group for group in groups if min_x <= group[2] <= max_x]
        ignored = [group for group in groups if group not in kept]
        return kept, ignored
    def single_reference(self, groups, width, side_hint, prior_center, expected_width):
        if not groups:
            return None
        prior = width * 0.5 if prior_center is None else float(prior_center)
        if side_hint == "right":
            target = clamp(prior + expected_width * 0.5, width * 0.15, width * 0.92)
            candidates = [g for g in groups if g[2] > width * 0.10] or groups
            return min(candidates, key=lambda g: abs(g[2] - target) - g[2] * 0.02 / width)
        if side_hint == "left":
            target = clamp(prior - expected_width * 0.5, width * 0.08, width * 0.85)
            candidates = [g for g in groups if g[2] < width * 0.90] or groups
            return min(candidates, key=lambda g: abs(g[2] - target) + g[2] * 0.02 / width)
        return min(groups, key=lambda g: abs(g[2] - prior))
    def single_side(self, ref, width, side_hint):
        """判断单条线是左边界还是右边界；有历史提示时不因过中心线立即翻边。"""
        center = ref[2]
        if side_hint in ("left", "right"):
            if side_hint == "left" and center > width * 0.70:
                return "right"
            if side_hint == "right" and center < width * 0.30:
                return "left"
            return side_hint
        if center > width * 0.55:
            return "right"
        if center < width * 0.45:
            return "left"
        return "left" if center < width * 0.5 else "right"
    def row_center(self, groups, width, expected_width, prior_center, follow_mode, side_hint):
        pair = self.best_pair(groups, width, expected_width, prior_center)
        valid, ignored = self.filter_outside(groups, pair, width, expected_width)
        if pair is not None:
            left, right = pair
            valid = [left, right]
            ignored = [group for group in groups if group != left and group != right]
            lane_width = right[0] - left[1]
            if follow_mode == "left":
                center = left[1] + expected_width * self.single_center_factor
                return center, valid, ignored, "left_ref_pair", lane_width, left, right, left[1]
            if follow_mode == "right":
                center = right[0] - expected_width * self.single_center_factor
                return center, valid, ignored, "right_ref_pair", lane_width, left, right, right[0]
            return (left[1] + right[0]) * 0.5, valid, ignored, "dual", lane_width, left, right, None
        if follow_mode in ("left", "right"):
            ref = self.single_reference(valid, width, follow_mode, prior_center, expected_width)
            ignored.extend([g for g in valid if g != ref])
            if ref is None:
                return None, valid, ignored, "missing", None, None, None, None
            if follow_mode == "left":
                return ref[1] + expected_width * self.single_center_factor, [ref], ignored, "left_ref_single", None, None, None, ref[1]
            return ref[0] - expected_width * self.single_center_factor, [ref], ignored, "right_ref_single", None, None, None, ref[0]
        if len(valid) > 1:
            ref = self.single_reference(valid, width, side_hint, prior_center, expected_width)
            ignored.extend([g for g in valid if g != ref])
            valid = [ref] if ref is not None else []
        if not valid:
            return None, valid, ignored, "missing", None, None, None, None
        ref = valid[0]
        single_width = max(expected_width, width * self.single_width_floor_ratio)
        side = self.single_side(ref, width, side_hint)
        if side == "left":
            return ref[1] + single_width * self.single_center_factor, valid, ignored, "left_single", None, None, None, ref[1]
        return ref[0] - single_width * self.single_center_factor, valid, ignored, "right_single", None, None, None, ref[0]
    def scan(self, binary, kalman, last_mid, failed_count, lane_width, follow_mode, side_hint):
        height, width = binary.shape[:2]
        search_top = int(height * self.roi_top_ratio)
        search_bot = int(height * self.roi_bottom_ratio)
        scan_rows = np.linspace(search_bot, search_top, self.scan_rows).astype(int)
        candidates = []
        lane_rows = []
        debug_groups = []
        ignored_groups = []
        lane_widths = []
        row_entries = []
        dual_rows = left_single_rows = right_single_rows = 0
        for index, y in enumerate(scan_rows):
            pixels = np.where(binary[y, :] == 255)[0]
            groups = self.row_groups(pixels, width)
            center, valid, ignored, kind, measured_width, left, right, ref_edge = self.row_center(
                groups, width, lane_width, last_mid, follow_mode, side_hint
            )
            debug_groups.extend([(int(g[2]), int(y), int(g[0]), int(g[1])) for g in valid])
            ignored_groups.extend([(int(g[2]), int(y), int(g[0]), int(g[1])) for g in ignored])
            if center is None:
                continue

            weight = 1.0 + (len(scan_rows) - index) * 0.25
            candidates.append((center, y, weight, kind))
            if measured_width is not None:
                lane_widths.append(float(measured_width))
            if kind in ("dual", "left_ref_pair", "right_ref_pair"):
                dual_rows += 1
            elif kind == "left_single":
                left_single_rows += 1
            elif kind == "right_single":
                right_single_rows += 1
            row_entries.append({
                "center": center,
                "y": y,
                "weight": weight,
                "kind": kind,
                "left_edge": None if left is None else left[1],
                "right_edge": None if right is None else right[0],
                "single_left": None if len(valid) != 1 else valid[0][0],
                "single_right": None if len(valid) != 1 else valid[0][1],
            })
            lane_rows.append(self._row_debug(width, y, center, kind, left, right, ref_edge, lane_width))

        if follow_mode == "normal":
            candidates, row_entries = self._fix_single_line_side(candidates, row_entries, width, lane_width, side_hint)
            candidates, row_entries = self._fix_discontinuous_pairs(candidates, row_entries, width, lane_width)
            lane_rows = self._lane_rows_from_entries(row_entries, width, lane_width)
            dual_rows = len([item for item in row_entries if item["kind"] == "dual"])
            left_single_rows = len([item for item in row_entries if item["kind"] == "left_single"])
            right_single_rows = len([item for item in row_entries if item["kind"] == "right_single"])
        raw_mid, dominant = self.fuse_candidates(candidates, dual_rows, left_single_rows, right_single_rows, follow_mode)
        predicted = kalman.predict()
        if raw_mid is not None:
            raw_mid = int(clamp(raw_mid, 0, width - 1))
        max_jump = width * (0.65 if follow_mode != "normal" else 0.45)
        if raw_mid is not None and abs(raw_mid - last_mid) < max_jump:
            measurement = np.array([[np.float32(raw_mid)]])
            mid = int(kalman.correct(measurement)[0])
            failed_count = 0
        else:
            mid = int(predicted[0])
            failed_count += 1
            if failed_count >= KALMAN_FAIL_MAX:
                kalman.statePost = np.array([[width // 2], [0]], np.float32)
                mid = width // 2
                failed_count = 0
        mid = int(clamp(mid, 0, width - 1))
        debug = {
            "raw_mid": raw_mid,
            "dominant": dominant,
            "groups": debug_groups,
            "ignored": ignored_groups,
            "lane_rows": lane_rows,
            "lane_widths": lane_widths,
            "search_top": search_top,
            "search_bot": search_bot,
            "dual_rows": dual_rows,
            "left_single_rows": left_single_rows,
            "right_single_rows": right_single_rows,
            "follow_mode": follow_mode,
            "side_hint": side_hint,
        }
        return mid - width * 0.5, [(mid, search_bot)], failed_count, debug
    def _single_side_from_entries(self, entries, width, side_hint):
        single_entries = [item for item in entries if item["kind"] in ("left_single", "right_single")]
        if len(single_entries) < 2:
            return None
        lower = max(single_entries, key=lambda item: item["y"])
        ref_center = (lower["single_left"] + lower["single_right"]) * 0.5
        if side_hint in ("left", "right"):
            if side_hint == "left" and ref_center > width * 0.70:
                return "right"
            if side_hint == "right" and ref_center < width * 0.30:
                return "left"
            return side_hint
        return "left" if ref_center < width * 0.5 else "right"
    def _fix_single_line_side(self, candidates, entries, width, lane_width, side_hint):
        if any(item["kind"] == "dual" for item in entries):
            return candidates, entries
        side = self._single_side_from_entries(entries, width, side_hint)
        if side is None:
            return candidates, entries
        fixed_entries = []
        fixed_candidates = []
        for item in entries:
            item = item.copy()
            if item["kind"] in ("left_single", "right_single"):
                if side == "left":
                    item["kind"] = "left_single"
                    item["left_edge"] = item["single_right"]
                    item["right_edge"] = None
                    item["center"] = item["single_right"] + lane_width * self.single_center_factor
                else:
                    item["kind"] = "right_single"
                    item["left_edge"] = None
                    item["right_edge"] = item["single_left"]
                    item["center"] = item["single_left"] - lane_width * self.single_center_factor
            fixed_entries.append(item)
            fixed_candidates.append((item["center"], item["y"], item["weight"], item["kind"]))
        return fixed_candidates, fixed_entries
    def _continuous_side(self, entries, key, width):
        values = [float(item[key]) for item in entries if item.get(key) is not None]
        if len(values) < 3:
            return False
        max_jump = width * LANE_CONTINUITY_JUMP_RATIO
        jumps = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]
        return max(jumps) <= max_jump
    def _fix_discontinuous_pairs(self, candidates, entries, width, lane_width):
        pair_entries = [item for item in entries if item["kind"] == "dual"]
        if len(pair_entries) < 2:
            return candidates, entries
        left_ok = self._continuous_side(pair_entries, "left_edge", width)
        right_ok = self._continuous_side(pair_entries, "right_edge", width)
        if left_ok == right_ok:
            return candidates, entries
        fixed_entries = []
        fixed_candidates = []
        for item in entries:
            if item["kind"] == "dual":
                if left_ok:
                    item = item.copy()
                    item["kind"] = "left_single"
                    item["center"] = item["left_edge"] + lane_width * self.single_center_factor
                elif right_ok:
                    item = item.copy()
                    item["kind"] = "right_single"
                    item["center"] = item["right_edge"] - lane_width * self.single_center_factor
            fixed_entries.append(item)
            fixed_candidates.append((item["center"], item["y"], item["weight"], item["kind"]))
        return fixed_candidates, fixed_entries
    def _lane_rows_from_entries(self, entries, width, lane_width):
        rows = []
        for item in entries:
            center = item["center"]
            kind = item["kind"]
            info = {"y": int(item["y"]), "center_x": int(clamp(center, 0, width - 1)),
                    "left_x": None, "right_x": None, "virtual_x": None}
            if kind == "dual":
                info["left_x"] = int(clamp(item["left_edge"], 0, width - 1))
                info["right_x"] = int(clamp(item["right_edge"], 0, width - 1))
            elif kind == "left_single":
                ref = item["single_right"] if item["single_right"] is not None else item["left_edge"]
                info["left_x"] = int(clamp(ref, 0, width - 1))
                info["virtual_x"] = int(clamp(ref + lane_width, 0, width - 1))
                info["right_x"] = info["virtual_x"]
            elif kind == "right_single":
                ref = item["single_left"] if item["single_left"] is not None else item["right_edge"]
                info["right_x"] = int(clamp(ref, 0, width - 1))
                info["virtual_x"] = int(clamp(ref - lane_width, 0, width - 1))
                info["left_x"] = info["virtual_x"]
            rows.append(info)
        return rows
    def _row_debug(self, width, y, center, kind, left, right, ref_edge, lane_width):
        info = {"y": int(y), "center_x": int(clamp(center, 0, width - 1)), "left_x": None, "right_x": None, "virtual_x": None}
        if left is not None:
            info["left_x"] = int(left[2])
        if right is not None:
            info["right_x"] = int(right[2])
        if ref_edge is not None:
            if "left" in kind:
                info["left_x"] = int(ref_edge)
                info["virtual_x"] = int(clamp(ref_edge + lane_width, 0, width - 1))
                info["right_x"] = info["virtual_x"]
            elif "right" in kind:
                info["right_x"] = int(ref_edge)
                info["virtual_x"] = int(clamp(ref_edge - lane_width, 0, width - 1))
                info["left_x"] = info["virtual_x"]
        return info
    def fuse_candidates(self, candidates, dual_rows, left_single_rows, right_single_rows, follow_mode):
        if not candidates:
            return None, None
        fused = candidates
        dominant = None
        if follow_mode == "normal":
            if right_single_rows >= 2 and right_single_rows > dual_rows and right_single_rows >= left_single_rows:
                dominant = "right_single"
                fused = [c for c in candidates if c[3] == "right_single"] or candidates
            elif left_single_rows >= 2 and left_single_rows > dual_rows:
                dominant = "left_single"
                fused = [c for c in candidates if c[3] == "left_single"] or candidates
            elif dual_rows >= 1:
                dominant = "dual"
                fused = [c for c in candidates if c[3] == "dual"] or candidates
        values = np.array([c[0] for c in fused], dtype=np.float32)
        weights = np.array([c[2] for c in fused], dtype=np.float32)
        return int(np.average(values, weights=weights)), dominant
    def detect_stopline_before_crosswalk(self, binary):
        height, width = binary.shape[:2]
        roi = binary.copy()
        roi[: int(height * 0.18), :] = 0
        roi[int(height * 0.95) :, :] = 0
        contours = find_contours(roi)

        stops, stripes = [], []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if area < 25:
                continue

            # 用最小外接旋转矩形判断形状，能识别斜着进入画面的停车横线和斑马线条。
            rect = cv2.minAreaRect(contour)
            (cx, cy), (rw, rh), angle = rect
            long_side = max(rw, rh)
            short_side = max(1.0, min(rw, rh))
            rect_fill = area / float(rw * rh + 1.0)
            ratio = long_side / short_side
            polygon = np.rint(cv2.boxPoints(rect)).astype(np.int32)
            candidate = {
                "box": (int(x), int(y), int(w), int(h)),
                "polygon": polygon.tolist(),
                "angle_deg": long_edge_angle_deg(polygon),
                "bottom_y": int(np.max(polygon[:, 1])),
                "center": (float(cx), float(cy)),
                "long_side": float(long_side),
                "short_side": float(short_side),
                "ratio": float(ratio),
            }

            fill = area / float(w * h + 1)
            aspect = w / float(max(h, 1))

            axis_stop = w > width * 0.25 and h < height * 0.08 and aspect > 4.0 and fill > 0.18
            rotated_stop = long_side > width * 0.25 and short_side < height * 0.11 and ratio > 3.2 and rect_fill > 0.28
            if (axis_stop or rotated_stop) and cy > height * 0.22:
                stops.append(candidate)

            stripe_shape = (
                1.6 <= ratio <= 5.5
                and width * 0.025 <= short_side <= width * 0.12
                and height * 0.06 <= long_side <= height * 0.36
                and rect_fill > 0.22
            )
            if stripe_shape:
                stripes.append(candidate)
        best = None
        for stop in stops:
            scored = self._score_stop_group(stop, stripes, binary.shape)
            if scored is not None and (best is None or scored["confidence"] > best["confidence"]):
                best = scored
        if best is None:
            # 没有完整停止线时，只返回“成组”的斑马线条纹，避免边线被误框后也被屏蔽。
            selected = self._select_crosswalk_stripes(stripes, binary.shape)
            confidence = min(0.6, len(selected) / 5.0 * 0.45)
            return {
                "candidate": False,
                "confidence": confidence,
                "stop_box": None,
                "stop_polygon": None,
                "stop_angle_deg": None,
                "stop_bottom_y": 0,
                "stripe_polygons": [item["polygon"] for item in selected],
                "loose_stripe_polygons": [item["polygon"] for item in stripes],
            }
        best["in_front"] = self._stop_is_in_front(best["stop_box"], binary.shape)
        best["candidate"] = (
            best["order_ok"]
            and best["in_front"]
            and best["confidence"] >= self.stop_confidence_min
        )
        best["loose_stripe_polygons"] = [item["polygon"] for item in stripes]
        return best
    def _select_crosswalk_stripes(self, stripes, shape):
        """只保留横向成组分布的斑马线条纹，单根边线/区域线不算斑马线。"""
        height, width = shape[:2]
        if len(stripes) < 3:
            return []
        best_group = []
        ordered = sorted(stripes, key=lambda item: item["center"][0])
        for base in ordered:
            base_cy = base["center"][1]
            group = []
            for item in ordered:
                cx, cy = item["center"]
                same_band = abs(cy - base_cy) <= height * 0.18
                not_side_edge = width * 0.08 <= cx <= width * 0.92
                if same_band and not_side_edge:
                    group.append(item)
            group = self._consistent_stripe_group(group, width, height)
            if len(group) > len(best_group):
                best_group = group
        if len(best_group) < 3:
            return []
        centers_x = [item["center"][0] for item in best_group]
        spread = max(centers_x) - min(centers_x)
        if spread < width * 0.16:
            return []
        return self._expand_crosswalk_group(best_group, stripes, shape)
    def _expand_crosswalk_group(self, core_group, stripes, shape):
        height, width = shape[:2]
        center_y = float(np.median([item["center"][1] for item in core_group]))
        long_median = float(np.median([item["long_side"] for item in core_group]))
        short_median = float(np.median([item["short_side"] for item in core_group]))
        angle_median = float(np.median([item["angle_deg"] for item in core_group]))
        min_x = min(item["center"][0] for item in core_group) - width * 0.22
        max_x = max(item["center"][0] for item in core_group) + width * 0.22
        expanded = []
        for item in stripes:
            cx, cy = item["center"]
            if not (width * 0.08 <= cx <= width * 0.94 and min_x <= cx <= max_x):
                continue
            same_band = abs(cy - center_y) <= height * 0.22
            size_ok = (
                long_median * 0.45 <= item["long_side"] <= long_median * 2.70
                and short_median * 0.35 <= item["short_side"] <= short_median * 2.60
            )
            angle_ok = undirected_angle_delta_deg(item["angle_deg"], angle_median) <= 35.0
            if same_band and size_ok and angle_ok:
                expanded.append(item)
        expanded = sorted(expanded, key=lambda item: item["center"][0])
        return expanded if len(expanded) >= len(core_group) else core_group
    def _consistent_stripe_group(self, group, width, height):
        if len(group) < 3:
            return []
        long_median = float(np.median([item["long_side"] for item in group]))
        short_median = float(np.median([item["short_side"] for item in group]))
        angle_median = float(np.median([item["angle_deg"] for item in group]))
        consistent = []
        for item in group:
            long_ok = long_median * 0.55 <= item["long_side"] <= long_median * 1.80
            short_ok = short_median * 0.55 <= item["short_side"] <= short_median * 1.80
            angle_ok = undirected_angle_delta_deg(item["angle_deg"], angle_median) <= 20.0
            if long_ok and short_ok and angle_ok:
                consistent.append(item)
        if len(consistent) < 3:
            return []
        consistent = sorted(consistent, key=lambda item: item["center"][0])
        centers_x = [item["center"][0] for item in consistent]
        gaps = [centers_x[i + 1] - centers_x[i] for i in range(len(centers_x) - 1)]
        if min(gaps) < max(3.0, short_median * 0.50):
            return []
        return consistent
    def _stop_is_in_front(self, stop_box, shape):
        """停车线必须接近车头正前方；侧边远处露出的横条不能触发路口。"""
        height, width = shape[:2]
        x, y, w, h = stop_box
        center_x = width * 0.5
        stop_center_x = x + w * 0.5
        margin = width * self.stop_front_center_margin_ratio
        crosses_center = x <= center_x <= x + w
        center_in_band = abs(stop_center_x - center_x) <= margin
        wide_enough = w >= width * 0.18
        return crosses_center or (center_in_band and wide_enough)
    def _score_stop_group(self, stop, stripes, shape):
        height, width = shape[:2]
        sx, sy, sw, sh = stop["box"]
        selected = self._select_crosswalk_stripes(stripes, shape)
        if len(selected) < 3:
            return None
        centers = [item["center"][0] for item in selected]
        group_x = float(np.mean(centers))
        if not (sx - width * 0.05 <= group_x <= sx + sw + width * 0.05):
            return None
        stop_y = self._line_y_at_x(stop, group_x)
        if stop_y is None:
            return None
        nearest = max(item["bottom_y"] for item in selected)
        order_ok = nearest - height * 0.01 <= stop_y <= nearest + height * 0.35
        spread = (max(centers) - min(centers)) / float(width * 0.35)
        angle_penalty = min(0.15, abs(stop["angle_deg"]) / 90.0 * 0.15)
        confidence = (
            min(1.0, len(selected) / 5.0) * 0.45
            + min(1.0, spread) * 0.25
            + (0.30 if order_ok else 0.0)
            - angle_penalty
        )
        return {
            "confidence": confidence,
            "order_ok": order_ok,
            "stop_box": stop["box"],
            "stop_polygon": stop["polygon"],
            "stop_angle_deg": stop["angle_deg"],
            "stop_bottom_y": stop["bottom_y"],
            "stripe_polygons": [item["polygon"] for item in selected],
        }
    def _line_y_at_x(self, item, x):
        angle = np.radians(item["angle_deg"])
        if abs(np.cos(angle)) < 0.12:
            return None
        cx, cy = item["center"]
        return float(cy + np.tan(angle) * (float(x) - cx))


class LaneFollower:
    def __init__(self):
        rospy.init_node("line_cy", anonymous=True)
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.twist = Twist()
        self.vision = LineVision()
        self.camera_backend = CAMERA_BACKEND
        self.camera_startup_wait = CAMERA_STARTUP_WAIT
        self.debug_max_width = DEBUG_MAX_WIDTH
        self.single_line_min_angular = SINGLE_LINE_MIN_ANGULAR
        self.angular_smooth_keep = ANGULAR_SMOOTH_KEEP
        self.angular_step_limit = ANGULAR_STEP_LIMIT
        self.search_angular_limit = SEARCH_ANGULAR_LIMIT
        self.single_confirm = SINGLE_TURN_CONFIRM_FRAMES
        self.single_release = SINGLE_TURN_RELEASE_FRAMES
        params = (
            ("camera_index", LANE_CAM_INDEX, int),
            ("process_width", PROCESS_WIDTH, int),
            ("dry_run", DRY_RUN, bool),
            ("detect_only", DETECT_ONLY, bool),
            ("debug_view", DEBUG_VIEW, bool),
            ("raw_view", RAW_VIEW, bool),
            ("line_speed", LINEAR_SPEED, float),
            ("single_line_speed", SINGLE_LINE_SPEED, float),
            ("search_speed", SEARCH_SPEED, float),
            ("max_angular", MAX_ANGULAR, float),
            ("stop_stable_frames", STOP_STABLE_FRAMES, int),
            ("stop_hold_time", STOP_HOLD_TIME, float),
            ("stop_cooldown_time", STOP_COOLDOWN_TIME, float),
            ("approach_crosswalk_speed", APPROACH_CROSSWALK_SPEED, float),
            ("align_trigger_y_ratio", ALIGN_TRIGGER_Y_RATIO, float),
            ("align_kp", ALIGN_KP, float),
            ("align_max_angular", ALIGN_MAX_ANGULAR, float),
            ("align_min_angular", ALIGN_MIN_ANGULAR, float),
            ("align_angle_tolerance_deg", ALIGN_ANGLE_TOLERANCE_DEG, float),
            ("align_stable_frames", ALIGN_STABLE_FRAMES, int),
            ("align_timeout", ALIGN_TIMEOUT, float),
            ("align_angular_sign", ALIGN_ANGULAR_SIGN, float),
            ("crosswalk_lost_frames", CROSSWALK_LOST_FRAMES, int),
            ("side_follow_speed", SIDE_FOLLOW_SPEED, float),
            ("enter_intersection_straight_time", ENTER_INTERSECTION_STRAIGHT_TIME, float),
            ("intersection_min_time", INTERSECTION_MIN_TIME, float),
            ("intersection_max_time", INTERSECTION_MAX_TIME, float),
            ("recover_dual_frames", RECOVER_DUAL_FRAMES, int),
            ("crosswalk_clear_confidence", CROSSWALK_CLEAR_CONFIDENCE, float),
            ("crosswalk_track_confidence", CROSSWALK_TRACK_CONFIDENCE, float),
            ("crosswalk_clear_frames", CROSSWALK_CLEAR_FRAMES, int),
            ("left_turn_bias", LEFT_TURN_BIAS, float),
            ("right_turn_bias", RIGHT_TURN_BIAS, float),
            ("straight_bias", STRAIGHT_BIAS, float),
            ("lane_width_pixels", LANE_WIDTH_PIXELS, float),
        )
        for name, default, cast in params:
            setattr(self, name, cast(rospy.get_param("~" + name, default)))

        self.lane_width_estimate = None
        self.configure_vision_params()

        self.pid = PIDController(
            PID_KP_SMALL,
            PID_KI,
            PID_KD_SMALL,
            (-self.max_angular, self.max_angular),
        )
        self.kalman = self._make_kalman()
        self.initialized = False
        self.last_mid = 320
        self.failed_count = 0
        self.last_angular = 0.0
        self.last_debug = {}
        self.last_stop = {"candidate": False, "confidence": 0.0, "stripe_polygons": []}
        self.state = "FOLLOW_LINE"
        self.stop_hits = 0
        self.stop_cooldown_until = 0.0
        self.crosswalk_lost_count = 0
        self.align_stable_count = 0
        self.align_started_at = 0.0
        self.last_reliable_stop = None
        self.single_latched = None
        self.single_candidate = None
        self.single_candidate_frames = 0
        self.single_release_count = 0
        self.single_side_hint = None
        self.single_side_hint_frames = 0
        self.cleaned = False

        self.cap = CameraReader(self.camera_index, self.camera_backend)
        if not self.cap.isOpened():
            rospy.logerr("无法打开巡线摄像头：camera_index=%d backend=%s", self.camera_index, self.camera_backend)
            rospy.signal_shutdown("无法打开巡线摄像头")
        rospy.on_shutdown(self.cleanup)
        rospy.loginfo(
            "line_cy 启动: camera_index=%d, process_width=%d, turn_cmd=%s, dry_run=%s, detect_only=%s, raw_view=%s",
            self.camera_index, self.process_width, self.get_turn_cmd(), self.dry_run, self.detect_only, self.raw_view,
        )
    def configure_vision_params(self):
        """常用视觉参数接 ROS；黑线提取参数优先直接改文件顶部宏定义。"""
        black_v_max = int(clamp(BLACK_V_MAX, 0, 255))
        blur_kernel_size = int(BLUR_KERNEL_SIZE)
        adaptive_block_size = int(ADAPTIVE_BLOCK_SIZE)
        morph_kernel_size = int(MORPH_KERNEL_SIZE)
        if blur_kernel_size < 1:
            blur_kernel_size = 1
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
        if adaptive_block_size < 3:
            adaptive_block_size = 3
        if adaptive_block_size % 2 == 0:
            adaptive_block_size += 1
        if morph_kernel_size < 1:
            morph_kernel_size = 1

        self.vision.lower_black = BLACK_HSV_LOWER.copy()
        self.vision.upper_black = np.array([180, 255, black_v_max], dtype=np.uint8)
        self.vision.blur_kernel_size = blur_kernel_size
        self.vision.adaptive_block_size = adaptive_block_size
        self.vision.adaptive_c = float(ADAPTIVE_C)
        self.vision.morph_kernel_size = morph_kernel_size
        self.vision.roi_top_ratio = float(rospy.get_param("~roi_top_ratio", ROI_TOP_RATIO))
        self.vision.roi_bottom_ratio = float(rospy.get_param("~roi_bottom_ratio", ROI_BOTTOM_RATIO))
        self.vision.default_width_ratio = float(rospy.get_param("~default_lane_width_ratio", DEFAULT_LANE_WIDTH_RATIO))
        self.vision.stop_confidence_min = float(rospy.get_param("~stop_confidence_min", STOP_CONFIDENCE_MIN))
        self.vision.stop_front_center_margin_ratio = float(
            rospy.get_param("~stop_front_center_margin_ratio", STOP_FRONT_CENTER_MARGIN_RATIO)
        )
    def _make_kalman(self):
        kalman = cv2.KalmanFilter(2, 1)
        kalman.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
        kalman.measurementMatrix = np.array([[1, 0]], np.float32)
        kalman.processNoiseCov = np.eye(2, dtype=np.float32) * 1e-4
        kalman.measurementNoiseCov = np.array([[1]], np.float32) * 1e-1
        return kalman
    def resize_frame(self, frame):
        if self.process_width <= 0:
            return frame
        height, width = frame.shape[:2]
        if width <= self.process_width:
            return frame
        scale = float(self.process_width) / float(width)
        return cv2.resize(frame, (self.process_width, int(height * scale)), interpolation=cv2.INTER_AREA)
    def get_turn_cmd(self):
        cmd = str(rospy.get_param("~turn_cmd", "straight")).lower().strip()
        if cmd in ("left", "straight", "right"):
            return cmd
        if hasattr(rospy, "logwarn"):
            rospy.logwarn("未知 turn_cmd=%s，已按 straight 处理；可选值: left / straight / right", cmd)
        return "straight"
    def lane_width(self, frame_width):
        return self.vision.expected_width(frame_width, self.lane_width_estimate, self.lane_width_pixels)
    def update_lane_width(self, debug, frame_width):
        widths = debug.get("lane_widths", [])
        if not widths:
            return
        measured = float(np.median(widths))
        if measured < frame_width * LANE_WIDTH_MIN_RATIO or measured > frame_width * LANE_WIDTH_MAX_RATIO:
            return
        self.lane_width_estimate = measured if self.lane_width_estimate is None else 0.85 * self.lane_width_estimate + 0.15 * measured
    def publish_cmd(self, linear, angular):
        self.twist.linear.x = float(linear)
        self.twist.linear.y = self.twist.linear.z = 0.0
        self.twist.angular.x = self.twist.angular.y = 0.0
        self.twist.angular.z = float(clamp(angular, -self.max_angular, self.max_angular))
        if motion_commands_enabled(self.dry_run, self.detect_only):
            self.pub.publish(self.twist)
    def stop_robot(self, duration=0.0):
        self.publish_cmd(0.0, 0.0)
        if duration > 0:
            rospy.sleep(duration)
            self.publish_cmd(0.0, 0.0)
    def update_pid_gain(self, deviation):
        if abs(deviation) < DEV_THRESHOLD:
            self.pid.kp = float(rospy.get_param("~kp_small", PID_KP_SMALL))
            self.pid.kd = float(rospy.get_param("~kd_small", PID_KD_SMALL))
        else:
            self.pid.kp = float(rospy.get_param("~kp_big", PID_KP_BIG))
            self.pid.kd = float(rospy.get_param("~kd_big", PID_KD_BIG))
    def raw_single_turn(self, debug):
        dual = debug.get("dual_rows", 0)
        left = debug.get("left_single_rows", 0)
        right = debug.get("right_single_rows", 0)
        if dual > 0:
            return None
        if right >= 2 and right > left:
            return "left"
        if left >= 2 and left > right:
            return "right"
        return None
    def stable_single_turn(self, raw_turn):
        if raw_turn is None:
            self.single_candidate = None
            self.single_candidate_frames = 0
            if self.single_latched is not None:
                self.single_release_count += 1
                if self.single_release_count >= max(1, self.single_release):
                    self.single_latched = None
                    self.single_release_count = 0
            return self.single_latched

        self.single_release_count = 0
        if raw_turn == self.single_latched:
            self.single_candidate = None
            self.single_candidate_frames = 0
            return self.single_latched
        if raw_turn == self.single_candidate:
            self.single_candidate_frames += 1
        else:
            self.single_candidate = raw_turn
            self.single_candidate_frames = 1
        need = max(1, self.single_confirm) if self.single_latched is None else max(1, self.single_confirm) + 1
        if self.single_candidate_frames >= need:
            if self.single_latched != raw_turn:
                self.pid.reset()
            self.single_latched = raw_turn
            self.single_candidate = None
            self.single_candidate_frames = 0
        return self.single_latched
    def limit_angular_step(self, angular):
        step = max(0.0, self.angular_step_limit)
        if step <= 0:
            return angular
        return self.last_angular + clamp(angular - self.last_angular, -step, step)
    def update_single_side_hint(self, debug, follow_mode):
        if follow_mode != "normal":
            return
        dominant = debug.get("dominant")
        if dominant in ("left_single", "right_single"):
            self.single_side_hint = "left" if dominant == "left_single" else "right"
            self.single_side_hint_frames = SINGLE_SIDE_HINT_FRAMES
        elif self.single_side_hint_frames > 0:
            self.single_side_hint_frames -= 1
            if self.single_side_hint_frames <= 0:
                self.single_side_hint = None
    def line_control(self, frame, binary, follow_mode="normal", speed=None, bias=0.0, allow_single_turn=True):
        width = frame.shape[1]
        if not self.initialized:
            self.last_mid = width // 2
            self.kalman.statePost = np.array([[self.last_mid], [0]], np.float32)
            self.initialized = True
        side_hint = self.single_side_hint if self.single_side_hint_frames > 0 else None
        deviation, centers, self.failed_count, debug = self.vision.scan(
            binary, self.kalman, self.last_mid, self.failed_count, self.lane_width(width), follow_mode, side_hint
        )
        self.last_mid = centers[-1][0]
        self.last_debug = debug
        self.update_single_side_hint(debug, follow_mode)
        self.update_lane_width(debug, width)
        self.update_pid_gain(deviation)
        if self.failed_count > 3:
            angular = clamp(self.last_angular * 0.5, -self.search_angular_limit, self.search_angular_limit)
            if follow_mode == "left":
                angular = max(angular, self.left_turn_bias)
            elif follow_mode == "right":
                angular = min(angular, -self.right_turn_bias)
            linear = capped_speed(self.search_speed, speed)
        else:
            angular = self.pid.update(deviation) + bias
            keep = clamp(self.angular_smooth_keep, 0.0, 0.9)
            angular = keep * self.last_angular + (1.0 - keep) * angular
            linear = capped_speed(self.line_speed, speed)
            if follow_mode == "normal" and allow_single_turn:
                raw = self.raw_single_turn(debug)
                stable = self.stable_single_turn(raw)
                debug["raw_single_turn"] = raw
                debug["single_turn"] = stable
                if stable == "left":
                    angular = max(angular, self.single_line_min_angular)
                    linear = min(linear, self.single_line_speed)
                elif stable == "right":
                    angular = min(angular, -self.single_line_min_angular)
                    linear = min(linear, self.single_line_speed)
        angular = self.limit_angular_step(clamp(angular, -self.max_angular, self.max_angular))
        self.last_angular = angular
        self.publish_cmd(linear, angular)
        return centers, angular
    def suppress_crosswalk_regions(self, binary, stop_result, include_loose=False):
        """路口补线时抹掉斑马线/停止线候选框，避免条纹被当成左右车道线。"""
        if not stop_result:
            return binary
        cleaned = binary.copy()
        height, width = cleaned.shape[:2]
        pad = max(4, int(width * 0.018))
        mask = np.zeros_like(cleaned)
        polygons = []
        if stop_result.get("stop_polygon"):
            polygons.append(stop_result["stop_polygon"])
        polygons.extend(stop_result.get("stripe_polygons", []))
        if include_loose:
            polygons.extend(stop_result.get("loose_stripe_polygons", []))
        for polygon in polygons:
            points = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
            cv2.fillConvexPoly(mask, points, 255)
        if np.any(mask):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1))
            mask = cv2.dilate(mask, kernel)
            cleaned[mask > 0] = 0
        return cleaned
    def maneuver_mode(self, cmd):
        if cmd == "left":
            return "left", self.left_turn_bias
        if cmd == "right":
            return "right", -self.right_turn_bias
        return "normal", self.straight_bias
    def maneuver_follow_choice(self, cmd, elapsed, mode, bias):
        if cmd == "straight" or elapsed < self.enter_intersection_straight_time:
            return "normal", self.straight_bias, False
        return mode, bias, True
    def crosswalk_visible_for_maneuver(self, stop_result):
        return (
            stop_result.get("confidence", 0.0) >= self.crosswalk_clear_confidence
            or len(stop_result.get("stripe_polygons", [])) >= 3
            or stop_result.get("candidate", False)
        )
    def crosswalk_mask_result(self, current, remembered=None):
        if remembered is None:
            return current
        merged = {
            "stop_polygon": current.get("stop_polygon") or remembered.get("stop_polygon"),
            "stripe_polygons": [],
            "loose_stripe_polygons": [],
        }
        for source in (remembered, current):
            merged["stripe_polygons"].extend(source.get("stripe_polygons", []))
            merged["loose_stripe_polygons"].extend(source.get("loose_stripe_polygons", []))
        return merged
    def run_maneuver(self, cmd):
        mode, bias = self.maneuver_mode(cmd)
        self.state = "MANEUVER"
        self.pid.reset()
        dual_stable = 0
        crosswalk_clear = 0
        crosswalk_memory = None
        crosswalk_memory_frames = 0
        start = rospy.get_time()
        rate = rospy.Rate(20)
        rospy.loginfo("进入路口补线: cmd=%s mode=%s", cmd, mode)
        while not rospy.is_shutdown() and rospy.get_time() - start <= self.intersection_max_time:
            ok, frame = self.cap.read()
            if not ok:
                rospy.logerr("路口补线中无法读取图像")
                break
            raw_frame = frame.copy()
            frame = self.resize_frame(frame)
            binary = self.vision.mask_black(frame)
            self.last_stop = self.vision.detect_stopline_before_crosswalk(binary)
            crosswalk_visible = self.crosswalk_visible_for_maneuver(self.last_stop)
            if crosswalk_visible:
                crosswalk_memory = self.last_stop
                crosswalk_memory_frames = MANEUVER_CROSSWALK_MEMORY_FRAMES
            elif crosswalk_memory_frames > 0:
                crosswalk_memory_frames -= 1
            remembered = crosswalk_memory if crosswalk_memory_frames > 0 else None
            mask_result = self.crosswalk_mask_result(self.last_stop, remembered)
            lane_binary = self.suppress_crosswalk_regions(
                binary, mask_result, include_loose=remembered is not None
            )
            elapsed = rospy.get_time() - start
            active_mode, active_bias, allow_single_turn = self.maneuver_follow_choice(cmd, elapsed, mode, bias)
            centers, _ = self.line_control(
                frame, lane_binary, active_mode, self.side_follow_speed, active_bias, allow_single_turn
            )

            if crosswalk_visible:
                crosswalk_clear = 0
            else:
                crosswalk_clear += 1

            recover_allowed_time = self.enter_intersection_straight_time + self.intersection_min_time
            can_recover = (
                elapsed >= recover_allowed_time
                and crosswalk_clear >= self.crosswalk_clear_frames
                and self.last_debug.get("dual_rows", 0) >= 2
                and self.failed_count <= 1
            )
            if can_recover:
                dual_stable += 1
            else:
                dual_stable = max(0, dual_stable - 1)
            self.show_views(raw_frame, lane_binary, centers)
            if dual_stable >= self.recover_dual_frames:
                break
            rate.sleep()
        elapsed = rospy.get_time() - start
        if elapsed >= self.intersection_max_time:
            rospy.logwarn("路口补线达到 %.1f 秒上限，按保护逻辑恢复巡线", self.intersection_max_time)
        self.state = "FOLLOW_LINE"
        self.pid.reset()
        self.failed_count = 0
        self.stop_hits = 0
        self.crosswalk_lost_count = 0
        self.align_stable_count = 0
        self.last_reliable_stop = None
        self.stop_cooldown_until = rospy.get_time() + self.stop_cooldown_time
        rospy.loginfo("路口动作完成，恢复巡线")
    def process_frame(self, frame):
        raw_frame = frame.copy()
        frame = self.resize_frame(frame)
        binary = self.vision.mask_black(frame)
        self.last_stop = self.vision.detect_stopline_before_crosswalk(binary)
        lane_binary = self.suppress_crosswalk_regions(binary, self.last_stop)
        now = rospy.get_time()
        candidate = bool(self.last_stop.get("candidate", False))
        tracking_visible = (
            self.state in ("APPROACH_CROSSWALK", "ALIGN_STOPLINE")
            and self.last_stop.get("stop_polygon") is not None
            and self.last_stop.get("confidence", 0.0) >= self.crosswalk_track_confidence
        )
        cooldown_ready = now >= self.stop_cooldown_until
        if candidate or tracking_visible:
            self.last_reliable_stop = self.last_stop
            self.crosswalk_lost_count = 0
        elif self.state in ("APPROACH_CROSSWALK", "ALIGN_STOPLINE"):
            self.crosswalk_lost_count += 1
        entry_ready = False
        if self.state == "FOLLOW_LINE":
            if candidate and cooldown_ready:
                self.stop_hits += 1
            else:
                self.stop_hits = max(0, self.stop_hits - 1)
            entry_ready = self.stop_hits >= self.stop_stable_frames and not self.detect_only
        visible_stop = candidate or tracking_visible
        effective_stop = self.last_stop if visible_stop else self.last_reliable_stop
        bottom_ratio = 0.0
        if effective_stop is not None and frame.shape[0] > 0:
            bottom_ratio = effective_stop.get("stop_bottom_y", 0) / float(frame.shape[0])
        align_timed_out = (
            self.state == "ALIGN_STOPLINE"
            and self.align_started_at > 0.0
            and now - self.align_started_at > self.align_timeout
        )
        next_state = crosswalk_next_state(
            self.state,
            entry_ready,
            visible_stop,
            bottom_ratio,
            self.align_trigger_y_ratio,
            self.crosswalk_lost_count,
            self.crosswalk_lost_frames,
            timed_out=align_timed_out,
        )
        if next_state != self.state:
            previous_state = self.state
            self.state = next_state
            if self.state == "ALIGN_STOPLINE":
                self.align_started_at = now
                self.align_stable_count = 0
                self.pid.reset()
                self.last_angular = 0.0
                self.last_debug = {}
                rospy.loginfo("停车横条已到近端，停车并开始摆正车身")
            elif self.state == "APPROACH_CROSSWALK":
                self.stop_hits = 0
                self.crosswalk_lost_count = 0
                rospy.loginfo("检测到斑马线入口，低速靠近停车横条")
            elif self.state == "FOLLOW_LINE":
                self.stop_hits = 0
                self.crosswalk_lost_count = 0
                self.last_reliable_stop = None
                rospy.logwarn("接近阶段持续丢失斑马线入口，恢复普通巡线")
            elif self.state == "CROSSWALK_WAIT":
                if align_timed_out:
                    rospy.logerr("停车横条摆正超过 %.1f 秒，保持停车等待人工检查", self.align_timeout)
                else:
                    rospy.logerr("%s 持续丢失停车横条，保持停车等待人工检查", previous_state)
            rospy.loginfo("循迹状态切换: %s -> %s", previous_state, self.state)
        if self.state == "APPROACH_CROSSWALK":
            if effective_stop is not None and not candidate:
                lane_binary = self.suppress_crosswalk_regions(lane_binary, effective_stop)
            centers, _ = self.line_control(
                frame, lane_binary, "normal", speed=min(self.line_speed, self.approach_crosswalk_speed)
            )
            self.show_views(raw_frame, lane_binary, centers)
            return
        if self.state == "ALIGN_STOPLINE":
            angle = effective_stop.get("stop_angle_deg") if effective_stop is not None and visible_stop else None
            if angle is None:
                self.align_stable_count = 0
                angular = 0.0
            elif abs(angle) <= self.align_angle_tolerance_deg:
                self.align_stable_count += 1
                angular = 0.0
            else:
                self.align_stable_count = 0
                angular = alignment_angular(
                    angle,
                    self.align_kp,
                    self.align_min_angular,
                    min(self.align_max_angular, self.max_angular),
                    self.align_angular_sign,
                )
            self.last_angular = angular
            self.publish_cmd(0.0, angular)

            if self.align_stable_count >= max(1, self.align_stable_frames):
                rospy.loginfo("停车横条已水平稳定 %d 帧，准备通过斑马线", self.align_stable_count)
                self.stop_robot(self.stop_hold_time)
                self.run_maneuver(self.get_turn_cmd())
                return

            self.show_views(raw_frame, lane_binary, [])
            return
        if self.state == "CROSSWALK_WAIT":
            self.stop_robot(0.0)
            self.show_views(raw_frame, lane_binary, [])
            return

        centers, _ = self.line_control(frame, lane_binary, "normal")
        self.show_views(raw_frame, lane_binary, centers)
    def show_views(self, raw_frame, binary, centers):
        if self.raw_view:
            self.show_raw_frame(raw_frame)
        if self.debug_view:
            self.draw_debug(binary, centers, self.last_stop)
    def show_raw_frame(self, frame):
        display = frame
        height, width = frame.shape[:2]
        if self.debug_max_width > 0 and width > self.debug_max_width:
            scale = float(self.debug_max_width) / float(width)
            display = cv2.resize(frame, (self.debug_max_width, int(height * scale)))
        try:
            cv2.imshow(RAW_VIEW_WINDOW, display)
            cv2.waitKey(1)
        except cv2.error as exc:
            rospy.logwarn("原图窗口打开失败，关闭 raw_view: %s", exc)
            self.raw_view = False
    def draw_debug(self, binary, centers, stop_result):
        display = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        height, width = binary.shape[:2]
        top = self.last_debug.get("search_top", int(height * ROI_TOP_RATIO))
        bot = self.last_debug.get("search_bot", int(height * ROI_BOTTOM_RATIO))
        cv2.rectangle(display, (0, top), (width - 1, bot), (0, 180, 0), 2)
        cv2.line(display, (width // 2, 0), (width // 2, height - 1), (90, 90, 90), 1)
        for x, y, left, right in self.last_debug.get("groups", []):
            cv2.circle(display, (x, y), 3, (255, 80, 0), -1)
            cv2.line(display, (left, y), (right, y), (255, 80, 0), 1)
        for x, y, left, right in self.last_debug.get("ignored", []):
            cv2.circle(display, (x, y), 3, (40, 40, 180), -1)
            cv2.line(display, (left, y), (right, y), (40, 40, 180), 1)
        for row in self.last_debug.get("lane_rows", []):
            y = row["y"]
            for key in ("left_x", "right_x"):
                if row.get(key) is not None:
                    cv2.line(display, (row[key], y - 12), (row[key], y + 12), (255, 255, 0), 2)
            if row.get("virtual_x") is not None:
                cv2.line(display, (row["virtual_x"], y - 18), (row["virtual_x"], y + 18), (255, 0, 255), 2)
            cv2.circle(display, (row["center_x"], y), 5, (0, 255, 0), -1)
        for x, y in centers:
            cv2.circle(display, (int(x), int(y)), 8, (0, 255, 0), 2)
        if stop_result:
            if stop_result.get("stop_polygon"):
                polygon = np.asarray(stop_result["stop_polygon"], dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(display, [polygon], True, (0, 0, 255), 3)
            for points in stop_result.get("stripe_polygons", []):
                polygon = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(display, [polygon], True, (0, 255, 255), 2)
        align_y = int(height * self.align_trigger_y_ratio)
        cv2.line(display, (0, align_y), (width - 1, align_y), (0, 128, 255), 1)
        status = "state={} hint={} raw={} single={} conf={:.2f} width={:.0f} fail={}".format(
            self.state,
            self.last_debug.get("side_hint"),
            self.last_debug.get("raw_single_turn"),
            self.last_debug.get("single_turn"),
            0.0 if not stop_result else stop_result.get("confidence", 0.0),
            self.lane_width(width),
            self.failed_count,
        )
        cv2.putText(display, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2)
        stop_angle = None if not stop_result else stop_result.get("stop_angle_deg")
        stop_bottom = 0.0
        if stop_result and height > 0:
            stop_bottom = stop_result.get("stop_bottom_y", 0) / float(height)
        crosswalk_status = "cross angle={} bottom={:.2f} align={}/{}".format(
            "None" if stop_angle is None else "{:.1f}".format(stop_angle),
            stop_bottom,
            self.align_stable_count,
            self.align_stable_frames,
        )
        cv2.putText(display, crosswalk_status, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 128, 255), 2)
        cv2.putText(display, "ROI green | active cyan/blue | ignored dark-red | virtual magenta | center green",
                    (10, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1)
        if self.debug_max_width > 0 and width > self.debug_max_width:
            scale = float(self.debug_max_width) / float(width)
            display = cv2.resize(display, (self.debug_max_width, int(height * scale)))
        try:
            cv2.imshow("line_cy_processed", display)
            cv2.waitKey(1)
        except cv2.error as exc:
            rospy.logwarn("处理图窗口打开失败，关闭 debug_view: %s", exc)
            self.debug_view = False
    def run(self):
        rate = rospy.Rate(20)
        no_frame_count = 0
        last_log = 0.0
        try:
            while not rospy.is_shutdown():
                timeout = self.camera_startup_wait if no_frame_count == 0 else 0.5
                ok, frame = self.cap.read(timeout=timeout)
                if not ok:
                    no_frame_count += 1
                    now = rospy.get_time()
                    if now - last_log > 2.0:
                        rospy.logerr("无法读取巡线摄像头图像：camera_index=%d backend=%s", self.camera_index, self.camera_backend)
                        last_log = now
                    self.stop_robot(0.0)
                    rate.sleep()
                    continue
                no_frame_count = 0
                self.process_frame(frame)
                rate.sleep()
        finally:
            self.cleanup()
    def cleanup(self):
        if self.cleaned:
            return
        self.cleaned = True
        try:
            self.stop_robot(0.0)
        except Exception:
            pass
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        if DESTROY_WINDOWS_ON_EXIT:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == "__main__":
    try:
        LaneFollower().run()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        pass
