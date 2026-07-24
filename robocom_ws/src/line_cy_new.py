#!/usr/bin/env python
# coding=utf-8
"""精简循迹：中心向外取双边线、斑马线横条、路口单边补线。"""
#能完整跑一圈，但是左右拐弯还是有问题
import threading
import time

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Twist


# ===== 摄像头与运行模式 =====
CAMERA_INDEX = 2          # 巡线摄像头编号；打不开画面时依次试 0/1/2。
PROCESS_WIDTH = 640       # 处理图宽度；通常保持 640，改动后 PID 和像素距离要重调。
DRY_RUN = True            # True 只识别不发速度；实车确认画面正确后改 False。
DEBUG_VIEW = True         # True 显示识别窗口；无显示器运行时可改 False。
TURN_CMD = "straight"    # 路口方向：left/straight/right；ROS ~turn_cmd 可覆盖。

# ===== 速度与转向 =====
FOLLOW_SPEED = 0.16       # 普通巡线 linear.x 前进速度(m/s)；整车过弯慢可小幅加。
APPROACH_SPEED = 0.16     # 靠近横条 linear.x 前进速度(m/s)；冲过横条就降。
MANEUVER_SPEED = 0.16     # 路口内 linear.x 前进速度(m/s)；路口通过慢可小幅加。
MAX_ANGULAR = 0.50       # angular.z 偏航角速度上限(rad/s)；只影响转头快慢，不提高前进速度。

# 左右转只需要调整下面四项；直行路口和普通巡线不使用这些参数。！！！
TURN_ENTRY_TIME = 6.8     # 摆正后进入盲区的直行时间(s)；起转太早加，太晚减。
TURN_SPEED = 0.16         # 盲区直行和固定转弯线速度(m/s)。
TURN_ANGULAR = 0.7        # 固定转弯角速度绝对值(rad/s)；越大转弯半径越小。
TURN_TIME = 3.4          # 固定转弯持续时间(s)；转不够就加，转过头就减。

KP = 0.0015              # 小误差比例；直线摆动就降，轻微修正不够就加。
KD = 0.0008              # 小误差阻尼；直线摆动就加，反应迟钝或尖峰大时降。
LARGE_ERROR_THRESHOLD_PIXELS = 136.0 # 误差达到此值切换急转 PD；太晚切换就减小。
LARGE_ERROR_KP = 0.0024 # 大误差比例；急弯拐不过就加，转得过猛就减。
LARGE_ERROR_KD = 0.01  # 大误差阻尼；急弯摆动就加，响应尖峰过大就减。
ANGULAR_SMOOTH = 0.88    # 转向保留比例；加大更平稳但迟钝，减小更灵敏。

# ===== 黑白二值图 =====
BLACK_V_MAX = 160         # 黑色亮度上限；黑线断裂就加，阴影/杂物太多就降。

# ===== 车道边线 =====
ROI_TOP = 0.2           # 识别区域上边界；减小看得更远，但更容易收到远处干扰。
ROI_BOTTOM = 0.92       # 识别区域下边界；增大看得更近，车头遮挡或噪声多就减小。
LANE_WIDTH_PIXELS = 620.0 # 车道内边缘间距；按当前 640 宽处理图估算，实测后可微调。
FILL_WIDTH_PIXELS = 620.0 # 单边补线间距；增大时跟左线向右移、跟右线向左移。
FOLLOW_CENTER_BIAS_PIXELS = 0.0 # 巡线目标横向偏置；正数向右、负数向左，固定偏航先调这里。
SCAN_ROWS = 9                 # 水平扫描行数；加大更稳但稍慢，过少容易漏线。
MIN_SEGMENT_WIDTH = 4        # 最小黑段宽度；噪点多就加，细线漏检就减。
MAX_SEGMENT_WIDTH_RATIO = 0.18 # 最大黑段占画面宽度；大黑块误识别就减。
DEFAULT_LANE_WIDTH_RATIO = 0.60 # 尚未学习时的默认车道宽度比例。
MIN_LANE_WIDTH_RATIO = 0.6  # 双边线最小间距；近线误配成双线时加大。
MAX_LANE_WIDTH_RATIO = 0.95  # 双边线最大间距；真实双线被拒绝时加大。
LANE_TRACK_MIN_POINTS = 2     # 构成边线点列的最少扫描点；误锁杂线就加，短边线漏锁就减。
LANE_TRACK_MIN_Y_SPAN_RATIO = 0.16 # 边线点列最小纵向跨度；短杂线参与裁剪就加。
LANE_TRACK_MAX_ERROR_RATIO = 0.025 # 边线拟合最大误差；弯道漏锁就加，杂线误锁就减。
LANE_OUTSIDE_MARGIN_RATIO = 0.012 # 边线外侧裁剪余量；切掉真实元素就加，外侧干扰多就减。

# ===== 斑马线竖条 =====
STRIPE_MIN_AREA = 25          # 单根竖条最小面积；噪点多就加，远处竖条漏检就减。
STRIPE_RATIO_MIN = 1.7        # 竖条最小长宽比；方块误识别就加。
STRIPE_RATIO_MAX = 5.8        # 竖条最大长宽比；细长真实竖条漏检就加。
STRIPE_SHORT_MIN_RATIO = 0.025 # 竖条短边最小图宽比例；小噪声多就加。
STRIPE_SHORT_MAX_RATIO = 0.13 # 竖条短边最大图宽比例；大块误识别就减。
STRIPE_LONG_MIN_RATIO = 0.055 # 竖条长边最小图高比例；远处竖条漏检就减。
STRIPE_LONG_MAX_RATIO = 0.45  # 竖条长边最大图高比例；近景斑马条漏检就加。
STRIPE_MIN_FILL = 0.25        # 竖条矩形填充率；空心杂物误识别就加。
STRIPE_GROUP_Y_RATIO = 0.18   # 同组竖条中心最大纵向差；斜拍漏组就加，乱组就减。
STRIPE_GROUP_ANGLE = 22.0     # 同组竖条最大角度差；透视大就加，乱组就减。
STRIPE_GROUP_SIZE_MIN = 0.5   # 同组竖条最小尺寸倍数。
STRIPE_GROUP_SIZE_MAX = 2.0   # 同组竖条最大尺寸倍数。
STRIPE_GROUP_MAX_GAP_RATIO = 0.18 # 相邻竖条最大横向间距；漏组就加，串入杂物就减。
STRIPE_GROUP_MIN_SPAN = 0.13  # 三根竖条最小横向跨度；局部噪声成组就加。
STRIPE_STRONG_COUNT = 2       # 强竖条证据数量；误触发就加，远处漏检就减到 2。
STRIPE_CENTER_X_MIN_RATIO = 0.04 # 竖条中心最小横坐标；画面边缘干扰多就加。
STRIPE_CENTER_X_MAX_RATIO = 0.98 # 竖条中心最大横坐标；画面边缘干扰多就减。

# ===== 停车横条几何 =====
BAR_HOUGH_THRESHOLD_RATIO = 0.055 # Hough 投票阈值比例；杂线多就加，断线漏检就减。
BAR_HOUGH_MIN_LENGTH_RATIO = 0.16 # 单个横条片段最小图宽比例；短杂线多就加。

BAR_HOUGH_MAX_GAP_RATIO = 0.05 # Hough 内部允许断口；横条断裂就加，乱连就减。

BAR_MAX_ABS_ANGLE = 45.0      # 横条相对画面水平最大角度；急斜拍漏检就加。
BAR_LANE_PARALLEL_ANGLE = 10.0 # 横条与边线方向接近到此值时判为边线。
BAR_LANE_DISTANCE_RATIO = 0.05 # 横条中心距边线最大图宽比例；边线误报就加，误排横条就减。
BAR_MERGE_ANGLE = 7.0         # 横条片段合并最大角差；半条不合并就加，乱并就减。
BAR_MERGE_DISTANCE_RATIO = 0.035 # 共线片段法向距离；双框不合并就加。
BAR_MERGE_GAP_RATIO = 0.14    # 共线片段最大断口；只识别半条就加，乱连就减。
BAR_STRIPE_X_MARGIN_RATIO = 0.04 # 横条与竖条匹配的横向余量。
BAR_STRIPE_Y_ABOVE_RATIO = 0.02 # 横条可高于竖条底部的图高比例。
BAR_STRIPE_Y_BELOW_RATIO = 0.38 # 横条可低于竖条底部的图高比例。
BAR_STRIPE_TOP_ABOVE_RATIO = 0.14 # 横条位于斑马条远端时，允许高于条纹顶部的距离。
BAR_STRIPE_TOP_BELOW_RATIO = 0.10 # 横条与条纹顶部少量重叠时的容差。
BAR_STRIPE_MIN_ANGLE = 55.0   # 横条与竖条最小夹角；边线误配就加。
BAR_LANE_OVERRIDE_MIN_ANGLE = 70.0 # 强横条绕过边线排除所需的最小条纹夹角。
BAR_LANE_OVERRIDE_GAP_RATIO = 0.15 # 强横条与斑马条端点的最小间隔，相对条纹短边。
BAR_TRACK_MIN_BOTTOM_RATIO = 0.34 # 横条进入画面的最低跟踪位置。
BAR_FRONT_MARGIN_RATIO = 0.30 # 横条必须覆盖车头中心附近的半宽比例。
BAR_STRONG_AXIS_MARGIN_RATIO = 0.06 # 强竖条通道横条覆盖车头轴线余量。
BAR_STRONG_MIN_MATCHED = 2    # 强通道横条至少匹配的竖条数；误识别就加，遮挡漏检就减。
BAR_THICKNESS_SEARCH_RATIO = 0.055 # 横条法向厚度搜索范围；厚横条截断就加。
BAR_THICKNESS_MIN_OCCUPANCY = 0.35 # 横条每层最小白像素占比；噪声变厚就加，断条就减。
BAR_DEFAULT_THICKNESS_RATIO = 0.025 # 无法测厚时默认图宽比例。
BAR_ONLY_MIN_THICKNESS_RATIO = 0.010 # 纯横条最小厚度；细线误报就加。
BAR_ONLY_MAX_THICKNESS_RATIO = 0.075 # 纯横条最大厚度；大块误报就减。

# ===== 横条时间防抖与纯横条通道 =====
STOP_STABLE_FRAMES = 1       # 入口连续确认帧数；误触发就加，反应太慢就减。
EXIT_ENTRY_IGNORE_TIME = 2 # 出口摆正后继续巡线且忽略新横条的时间(s)。
STOP_NEAR_RATIO = 0.8       # 横条接近画面底部比例；停得太早加，太晚减。
STOP_CENTER_WIDTH_RATIO = 0.15 # 底部停车区占画面中央宽度比例；减小可排除两侧干扰。
BAR_ONLY_STABLE_FRAMES = 1   # 无竖纹时横条连续确认帧数；误识别就加，出现太慢就减。
BAR_ONLY_MIN_LENGTH_RATIO = 0.2 # 无竖纹横条最小宽度比例；误识别就加，近景短条漏检就减。
BAR_ONLY_MAX_ABS_ANGLE = 20.0 # 无竖纹横条最大倾角；斜车道线误报就减，真实斜横条漏检就加。
BAR_TRACK_MAX_Y_RATIO = 0.18 # 前后帧横条最大纵向跳变；抖动串线就减，车速快就加。
BAR_TRACK_MAX_X_RATIO = 0.24 # 前后帧横条最大横向跳变；串线就减，急转跟丢就加。
BAR_TRACK_MAX_ANGLE = 10.0   # 前后帧横条最大角度跳变；串线就减，转动车身快就加。
BAR_TRACK_HOLD_FRAMES = 1    # 横条短暂丢失保持帧数；闪烁就加，残影久就减。
BAR_TRACK_SMOOTH = 0.25      # 横条位置历史保留比例；当前帧占 75%，框跟随迟钝就继续减。

# ===== 停车摆正 =====
ALIGN_TOLERANCE_DEG = 3.0    # 横条水平容差；难以完成摆正就加，要求更正就减。
ALIGN_STABLE_FRAMES = 5      # 摆正稳定帧数；容易误通过就加，等待太久就减。
ALIGN_KP = 0.025             # 摆正转向比例；摆正太慢就加，来回过冲就减。
ALIGN_MIN_ANGULAR = 0.08     # 摆正最小角速度；小误差转不动就加。
ALIGN_MAX_ANGULAR = 0.35     # 摆正最大角速度；转得太猛就减，太慢可加。
LOST_LIMIT = 7               # 横条允许丢失帧数；偶发丢线就加，错误等待太久就减。
EXIT_ALIGN_LOST_FRAMES = 5   # 出口摆正横条连续丢失帧数；防止单帧闪烁提前恢复巡线。
ALIGN_TIMEOUT = 5.0          # 摆正最长时间；横条仍稳定时超时会按放宽角度进入。
ALIGN_ENTRY_MAX_ANGLE = 10.0 # 超时进入路口允许的最大横条角度；入场太斜就减。
ALIGN_ENTRY_MIN_STRIPES = 3  # 超时进入至少需要的竖条数；误进入就加。
WAIT_RECOVER_FRAMES = 3      # WAIT 中重新确认入口的连续帧数；误恢复就加。

# ===== 路口通过与双边透视补线 =====
MANEUVER_MIN_TIME = 1.0      # 路口最短通过时间；过早恢复巡线就加。
MANEUVER_MAX_TIME = 14.0     # 未识别到出口横条时，超过此时间恢复普通巡线。
MANEUVER_LOOKAHEAD_RATIO = 0.60 # 路口中心线前视控制行；减小看得更远，增大看得更近。
ENTRY_CLEAR_FRAMES = 6       # 入口斑马线消失确认；入口被当出口就加。
EXIT_BAR_FRAMES = 1          # 第二条横条连续确认帧数；误触发就加，退出太慢就减。
RESTORE_DUAL_FRAMES = 4      # 仅保留兼容配置；双边线恢复不再作为路口退出条件。
RANSAC_RESIDUAL_PIXELS = 12.0 # 直线内点容差像素；线断/抖就加，圆角混入就减。
RANSAC_MIN_INLIERS = 4       # 直线最少内点数；误拟合就加，难锁定就减。
MODEL_HOLD_FRAMES = 8        # 边线丢失后保持帧数；短暂丢线就加，旧线残留就减。
MODEL_MAX_SHIFT_RATIO = 0.08 # 新旧直线最大位置跳变；误换圆角就减，转向变化大就加。
MODEL_MAX_SLOPE_DELTA = 0.35 # 新旧直线最大斜率变化；误换线就减，允许急变就加。
MODEL_CENTER_CONSISTENCY_RATIO = 0.20 # 双边补出的中心最大差异；圆角抢线时优先连续模型。
MODEL_MAX_ABS_SLOPE = 0.85  # 新锁边线最大横向斜率；斑马条拟合成贯穿画面的斜线时减小。
MODEL_CENTER_CROSS_MARGIN_RATIO = 0.08 # 边线在 ROI 内越过画面中心的容差；误锁对侧杂线时减小。
WINDOW_NAME = "line_cy_new" # 调试窗口名称；不影响算法。
PROCESSED_WINDOW_NAME = "line_cy_new_processed" # 二值处理结果窗口名称。


def clamp(value, low, high):
    return max(low, min(high, value))


def motion_enabled(dry_run):
    return not bool(dry_run)


def normalize_angle(angle):
    angle = float(angle)
    while angle > 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    return angle


def angle_delta(a, b):
    delta = abs(normalize_angle(a) - normalize_angle(b))
    return min(delta, 180.0 - delta)


def polygon_long_angle(points):
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    edges = np.roll(points, -1, axis=0) - points
    edge = edges[int(np.argmax(np.sum(edges * edges, axis=1)))]
    return normalize_angle(np.degrees(np.arctan2(edge[1], edge[0])))


def polygon_bottom_in_center_band(polygon, frame_width,
                                  center_width_ratio=STOP_CENTER_WIDTH_RATIO):
    """返回多边形在画面中央指定宽度内的最低点，未相交时返回 0。"""
    if polygon is None or frame_width <= 0:
        return 0.0
    ratio = clamp(float(center_width_ratio), 0.0, 1.0)
    if ratio <= 0.0:
        return 0.0
    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if len(points) < 2:
        return 0.0
    left = float(frame_width) * (1.0 - ratio) * 0.5
    right = float(frame_width) - left
    y_values = []
    for index, first in enumerate(points):
        second = points[(index + 1) % len(points)]
        if left <= first[0] <= right:
            y_values.append(float(first[1]))
        delta_x = float(second[0] - first[0])
        if abs(delta_x) < 1e-6:
            continue
        for boundary in (left, right):
            amount = (boundary - float(first[0])) / delta_x
            if 0.0 <= amount <= 1.0:
                y_values.append(float(first[1] + amount * (second[1] - first[1])))
    return max(y_values) if y_values else 0.0


def find_contours(binary):
    result = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return result[0] if len(result) == 2 else result[1]


def row_segments(row, min_width=MIN_SEGMENT_WIDTH, max_width=None):
    pixels = np.flatnonzero(row > 0)
    if len(pixels) == 0:
        return []
    segments = []
    start = last = int(pixels[0])
    for pixel in pixels[1:]:
        pixel = int(pixel)
        if pixel == last + 1:
            last = pixel
            continue
        width = last - start + 1
        if width >= min_width and (max_width is None or width <= max_width):
            segments.append((start, last))
        start = last = pixel
    width = last - start + 1
    if width >= min_width and (max_width is None or width <= max_width):
        segments.append((start, last))
    return segments


def center_out_segments(row, center_x, min_width=MIN_SEGMENT_WIDTH, max_width=None):
    """从画面中心向两侧取第一段黑色，外部黑块不会进入车道配对。"""
    segments = row_segments(row, min_width, max_width)
    left = [item for item in segments if item[1] < center_x]
    right = [item for item in segments if item[0] > center_x]
    return (max(left, key=lambda item: item[1]) if left else None,
            min(right, key=lambda item: item[0]) if right else None)


def clip_points_for_display(points, width):
    """仅把调试点裁到画面边缘，不改变控制使用的真实补线坐标。"""
    return [(int(clamp(round(x), 0, width - 1)), int(y)) for x, y in points]


def control_target_x(center_x, bias_pixels):
    return float(center_x) + float(bias_pixels)


def pd_gains(deviation):
    if abs(float(deviation)) >= LARGE_ERROR_THRESHOLD_PIXELS:
        return LARGE_ERROR_KP, LARGE_ERROR_KD, "large"
    return KP, KD, "small"


def normalize_turn_cmd(value):
    command = str(value).strip().lower()
    return command if command in ("left", "straight", "right") else "straight"


def maneuver_follow_side(turn_cmd):
    command = normalize_turn_cmd(turn_cmd)
    return command if command in ("left", "right") else None


def strong_lane_override_enabled(state, turn_cmd, maneuver_phase):
    return (
        state == "MANEUVER"
        and maneuver_follow_side(turn_cmd) is not None
        and maneuver_phase == "EXIT_STRAIGHT"
    )


def fixed_turn_command(turn_cmd, speed, angular):
    command = normalize_turn_cmd(turn_cmd)
    if command == "left":
        return float(speed), float(angular)
    if command == "right":
        return float(speed), -float(angular)
    return None


def turn_phase_next(phase, elapsed, entry_time, turn_time):
    if phase == "ENTRY" and elapsed >= entry_time:
        return "TURN"
    if phase == "TURN" and elapsed >= turn_time:
        return "EXIT_STRAIGHT"
    return None


def maneuver_observation_target(observation):
    return observation.center_x if observation.valid else None


def follow_entry_hits(candidate, current_hits):
    return current_hits + 1 if candidate else max(0, current_hits - 1)


def entry_acceptance_enabled(now, accept_after):
    return float(now) >= float(accept_after)


def same_tracked_bar(first, second, shape):
    height, width = shape[:2]
    return (
        abs(first["center"][1] - second["center"][1])
        <= height * BAR_TRACK_MAX_Y_RATIO
        and abs(first["center"][0] - second["center"][0])
        <= width * BAR_TRACK_MAX_X_RATIO
        and angle_delta(first["angle"], second["angle"])
        <= BAR_TRACK_MAX_ANGLE
    )


class LaneObservation(object):
    def __init__(self, center_x, valid, dual_rows, left_points, right_points,
                 measured_width=None, follow_side=None, center_points=None,
                 virtual_left_points=None, virtual_right_points=None):
        self.center_x = float(center_x)
        self.valid = bool(valid)
        self.dual_rows = int(dual_rows)
        self.left_points = list(left_points)
        self.right_points = list(right_points)
        self.measured_width = measured_width
        self.follow_side = follow_side
        self.center_points = list(center_points or [])
        self.virtual_left_points = list(virtual_left_points or [])
        self.virtual_right_points = list(virtual_right_points or [])


class LaneDetector(object):
    def __init__(self, roi_top=ROI_TOP, roi_bottom=ROI_BOTTOM, scan_rows=SCAN_ROWS,
                 fill_width=0.0):
        self.roi_top = float(roi_top)
        self.roi_bottom = float(roi_bottom)
        self.scan_rows = int(scan_rows)
        self.fill_width = float(fill_width)

    def points(self, binary, center_x=None):
        height, width = binary.shape[:2]
        center_x = width * 0.5 if center_x is None else float(center_x)
        rows = np.linspace(int(height * self.roi_bottom), int(height * self.roi_top),
                           self.scan_rows).astype(np.int32)
        left_points, right_points = [], []
        max_width = int(width * MAX_SEGMENT_WIDTH_RATIO)
        for y in rows:
            left, right = center_out_segments(binary[y], center_x, MIN_SEGMENT_WIDTH, max_width)
            if left is not None:
                left_points.append((int(left[1]), int(y)))
            if right is not None:
                right_points.append((int(right[0]), int(y)))
        return left_points, right_points

    def points_near_model(self, binary, model, band_ratio=0.10):
        """锁线后按预测位置取点，靠中心的圆角不能抢走外侧直线。"""
        height, width = binary.shape[:2]
        rows = np.linspace(int(height * self.roi_bottom), int(height * self.roi_top),
                           self.scan_rows).astype(np.int32)
        max_width = int(width * MAX_SEGMENT_WIDTH_RATIO)
        max_distance = width * float(band_ratio)
        points = []
        for y in rows:
            segments = row_segments(binary[y], MIN_SEGMENT_WIDTH, max_width)
            if not segments:
                continue
            expected = model.x_at(y)
            segment = min(segments, key=lambda item: abs((item[0] + item[1]) * 0.5 - expected))
            center = (segment[0] + segment[1]) * 0.5
            if abs(center - expected) <= max_distance:
                points.append((int(round(center)), int(y)))
        return points

    def observe(self, binary, lane_width, follow_side=None, center_hint=None,
                side_center_transform=None):
        height, width = binary.shape[:2]
        center_hint = width * 0.5 if center_hint is None else float(center_hint)
        left_points, right_points = self.points(binary, center_hint)
        left_by_y = {y: x for x, y in left_points}
        right_by_y = {y: x for x, y in right_points}
        expected = float(lane_width) if lane_width > 0 else width * DEFAULT_LANE_WIDTH_RATIO
        fill_width = self.fill_width if self.fill_width > 0 else expected
        all_rows = sorted(set(left_by_y).union(right_by_y), reverse=True)
        center_points = []
        virtual_left, virtual_right = [], []
        widths = []
        last_center = center_hint
        used_sides = []
        for y in all_rows:
            left_x = left_by_y.get(y)
            right_x = right_by_y.get(y)
            gap = None if left_x is None or right_x is None else right_x - left_x
            dual_ok = (
                gap is not None
                and width * MIN_LANE_WIDTH_RATIO <= gap <= width * MAX_LANE_WIDTH_RATIO
            )
            if follow_side is None and dual_ok:
                center = (left_x + right_x) * 0.5
                widths.append(float(gap))
                used_sides.append("dual")
            else:
                candidates = []
                if left_x is not None and follow_side in (None, "left"):
                    offset = fill_width * 0.5
                    if follow_side == "left" and side_center_transform is not None:
                        offset = (side_center_transform[0] * y
                                  + side_center_transform[1])
                    candidates.append((left_x + offset, "left", offset))
                if right_x is not None and follow_side in (None, "right"):
                    offset = -fill_width * 0.5
                    if follow_side == "right" and side_center_transform is not None:
                        offset = (side_center_transform[0] * y
                                  + side_center_transform[1])
                    candidates.append((right_x + offset, "right", offset))
                if not candidates:
                    continue
                center, side, offset = min(
                    candidates, key=lambda item: abs(item[0] - last_center)
                )
                if side == "left":
                    virtual_right.append((left_x + offset * 2.0, y))
                else:
                    virtual_left.append((right_x + offset * 2.0, y))
                used_sides.append(side)
            center_points.append((center, y))
            last_center = center

        if not center_points:
            return LaneObservation(center_hint, False, 0, left_points, right_points,
                                   None, follow_side)
        measured = float(np.median(widths)) if widths else None
        if follow_side in ("left", "right"):
            active_side = follow_side
        elif widths:
            active_side = None
        else:
            active_side = max(set(used_sides), key=used_sides.count)
        return LaneObservation(
            np.average([x for x, _ in center_points]), True, len(widths),
            left_points, right_points, measured, active_side, center_points,
            virtual_left_points=virtual_left,
            virtual_right_points=virtual_right,
        )


class LineModel(object):
    def __init__(self, slope, intercept, inlier_count, error):
        self.slope = float(slope)
        self.intercept = float(intercept)
        self.inlier_count = int(inlier_count)
        self.error = float(error)

    def x_at(self, y):
        return self.slope * float(y) + self.intercept

    def shifted(self, slope_delta, intercept_delta):
        return LineModel(self.slope + float(slope_delta),
                         self.intercept + float(intercept_delta),
                         self.inlier_count, self.error)


def fit_line_ransac(points, residual=12.0, prior=None):
    """确定性点对 RANSAC；直线模型会将入口圆角当成离群点。"""
    points = [(float(x), float(y)) for x, y in points]
    if len(points) < 2:
        return None
    best = None
    for i in range(len(points) - 1):
        for j in range(i + 1, len(points)):
            x1, y1 = points[i]
            x2, y2 = points[j]
            if abs(y2 - y1) < 1.0:
                continue
            slope = (x2 - x1) / (y2 - y1)
            intercept = x1 - slope * y1
            errors = np.array([abs(x - (slope * y + intercept)) for x, y in points])
            mask = errors <= residual
            count = int(np.count_nonzero(mask))
            if count < 2:
                continue
            mean_error = float(np.mean(errors[mask]))
            continuity = 0.0
            if prior is not None:
                continuity = abs((slope * 360.0 + intercept) - prior.x_at(360.0)) * 0.03
            score = (-count, mean_error + continuity)
            if best is None or score < best[0]:
                best = (score, mask)
    if best is None:
        return None
    inliers = np.asarray(points, dtype=np.float64)[best[1]]
    if len(inliers) < 2:
        return None
    slope, intercept = np.polyfit(inliers[:, 1], inliers[:, 0], 1)
    error = float(np.mean(np.abs(inliers[:, 0] - (slope * inliers[:, 1] + intercept))))
    return LineModel(slope, intercept, len(inliers), error)


class DualLineBridge(object):
    def __init__(self, lane_width, fill_width=0.0,
                 hold_frames=MODEL_HOLD_FRAMES):
        self.lane_width = float(lane_width)
        self.fill_width = float(fill_width)
        self.hold_frames = int(hold_frames)
        self.left_model = None
        self.right_model = None
        self.left_lost_frames = 0
        self.right_lost_frames = 0
        self.last_center = None
        self.center_model = None
        self.left_to_center = None
        self.right_to_center = None

    def reset(self, lane_width=None):
        if lane_width is not None:
            self.lane_width = float(lane_width)
        self.left_model = None
        self.right_model = None
        self.left_lost_frames = 0
        self.right_lost_frames = 0
        self.last_center = None
        self.center_model = None
        self.left_to_center = None
        self.right_to_center = None

    def _learn_center_geometry(self):
        left = self.left_model
        right = self.right_model
        if (left is None or right is None or self.left_lost_frames != 0
                or self.right_lost_frames != 0):
            return
        center_slope = (left.slope + right.slope) * 0.5
        center_intercept = (left.intercept + right.intercept) * 0.5
        inliers = min(left.inlier_count, right.inlier_count)
        error = max(left.error, right.error)
        self.center_model = LineModel(center_slope, center_intercept, inliers, error)
        self.left_to_center = (center_slope - left.slope,
                               center_intercept - left.intercept)
        self.right_to_center = (center_slope - right.slope,
                                center_intercept - right.intercept)

    def _center_from_side(self, model, transform, direction, fill_width):
        if model is None:
            return None
        if transform is not None:
            return model.shifted(transform[0], transform[1])
        shift = fill_width * 0.5 * direction
        return model.shifted(0.0, shift)

    def _model_geometry_valid(self, candidate, side, target_y,
                              frame_width, validation_top_y):
        if candidate is None or abs(candidate.slope) > MODEL_MAX_ABS_SLOPE:
            return False
        center_x = float(frame_width) * 0.5
        margin = float(frame_width) * MODEL_CENTER_CROSS_MARGIN_RATIO
        sample_ys = (float(validation_top_y), float(target_y))
        if side == "left":
            return all(candidate.x_at(y) <= center_x + margin for y in sample_ys)
        return all(candidate.x_at(y) >= center_x - margin for y in sample_ys)

    def _update_model(self, points, model, lost_frames, target_y, side,
                      frame_width, validation_top_y):
        candidate = fit_line_ransac(points, residual=RANSAC_RESIDUAL_PIXELS,
                                    prior=model)
        enough = candidate is not None \
            and candidate.inlier_count >= RANSAC_MIN_INLIERS \
            and self._model_geometry_valid(
                candidate, side, target_y, frame_width, validation_top_y
            )
        continuous = enough
        if continuous and model is not None:
            max_shift = float(frame_width) * MODEL_MAX_SHIFT_RATIO
            continuous = (
                abs(candidate.x_at(target_y) - model.x_at(target_y)) <= max_shift
                and abs(candidate.slope - model.slope) <= MODEL_MAX_SLOPE_DELTA
            )
        if continuous:
            return candidate, 0
        lost_frames += 1
        if lost_frames > self.hold_frames:
            model = None
        return model, lost_frames

    def update(self, left_points, right_points, target_y, center_hint=None,
               frame_width=PROCESS_WIDTH, validation_top_y=None):
        if validation_top_y is None:
            all_points = list(left_points) + list(right_points)
            validation_top_y = min([y for _, y in all_points] or [target_y])
        self.left_model, self.left_lost_frames = self._update_model(
            left_points, self.left_model, self.left_lost_frames, target_y,
            "left", frame_width, validation_top_y
        )
        self.right_model, self.right_lost_frames = self._update_model(
            right_points, self.right_model, self.right_lost_frames, target_y,
            "right", frame_width, validation_top_y
        )
        self._learn_center_geometry()

        fill_width = self.fill_width if self.fill_width > 0 else self.lane_width
        fresh_candidates = []
        held_candidates = []
        if self.left_model is not None:
            model = self._center_from_side(
                self.left_model, self.left_to_center, 1.0, fill_width
            )
            item = (model.x_at(target_y), "left", model)
            (fresh_candidates if self.left_lost_frames == 0 else held_candidates).append(item)
        if self.right_model is not None:
            model = self._center_from_side(
                self.right_model, self.right_to_center, -1.0, fill_width
            )
            item = (model.x_at(target_y), "right", model)
            (fresh_candidates if self.right_lost_frames == 0 else held_candidates).append(item)
        candidates = fresh_candidates if fresh_candidates else held_candidates
        if not candidates:
            self.last_center = None
            return None, None, None

        if len(candidates) == 1:
            center = candidates[0][0]
            center_model = candidates[0][2]
        else:
            left_center, right_center = candidates[0][0], candidates[1][0]
            consistent = abs(left_center - right_center) \
                <= fill_width * MODEL_CENTER_CONSISTENCY_RATIO
            if consistent:
                center = (left_center + right_center) * 0.5
                left_center_model, right_center_model = candidates[0][2], candidates[1][2]
                center_model = LineModel(
                    (left_center_model.slope + right_center_model.slope) * 0.5,
                    (left_center_model.intercept + right_center_model.intercept) * 0.5,
                    min(left_center_model.inlier_count, right_center_model.inlier_count),
                    max(left_center_model.error, right_center_model.error),
                )
            else:
                reference = self.last_center if self.last_center is not None else center_hint
                if reference is None:
                    center = (left_center + right_center) * 0.5
                    center_model = self.center_model
                else:
                    chosen = min(candidates, key=lambda item: abs(item[0] - reference))
                    center, center_model = chosen[0], chosen[2]
        self.last_center = float(center)
        if center_model is not None:
            self.center_model = center_model
        return self.last_center, self.left_model, self.right_model


class RightLineBridge(object):
    """兼容旧调用；新路口控制使用 DualLineBridge。"""
    def __init__(self, lane_width, fill_width=0.0,
                 hold_frames=MODEL_HOLD_FRAMES):
        self.bridge = DualLineBridge(lane_width, fill_width, hold_frames)

    @property
    def lane_width(self):
        return self.bridge.lane_width

    @lane_width.setter
    def lane_width(self, value):
        self.bridge.lane_width = float(value)

    @property
    def model(self):
        return self.bridge.right_model

    def reset(self, lane_width=None):
        self.bridge.reset(lane_width)

    def update(self, right_points, target_y):
        center, _, model = self.bridge.update([], right_points, target_y)
        return center, model


class CrosswalkResult(object):
    def __init__(self):
        self.candidate = False
        self.confidence = 0.0
        self.stop_polygon = None
        self.stop_angle = None
        self.stop_bottom = 0
        self.tracking_polygon = None
        self.tracking_angle = None
        self.tracking_bottom = 0
        self.stripe_polygons = []
        self.loose_polygons = []


class CrosswalkDetector(object):
    def __init__(self):
        self.last_bar = None
        self.bar_only_hits = 0
        self.bar_lost_hits = 0
        self.bar_locked = False

    def lock_current_bar(self):
        self.bar_locked = self.last_bar is not None
        return self.bar_locked

    def unlock_bar(self):
        self.bar_locked = False

    def _drivable_mask(self, shape, lane_points):
        """只保留边线内侧；单边线时以画面中心判断哪一侧可通行。"""
        height, width = shape[:2]
        mask = np.full((height, width), 255, dtype=np.uint8)
        if not lane_points:
            return mask
        tracks = [lane_points] if isinstance(lane_points[0], tuple) else lane_points
        center_x = width * 0.5
        left_tracks, right_tracks = [], []
        for track in tracks:
            if not self._reliable_lane_track(track, shape):
                continue
            median_x = float(np.median([point[0] for point in track]))
            (left_tracks if median_x < center_x else right_tracks).append(track)

        margin = max(4, int(width * LANE_OUTSIDE_MARGIN_RATIO))
        rows = np.arange(height, dtype=np.float64)

        def boundary(track):
            points = np.asarray(track, dtype=np.float64)
            slope, intercept = np.polyfit(points[:, 1], points[:, 0], 1)
            return np.clip(slope * rows + intercept, 0, width - 1)

        if left_tracks:
            left = boundary(max(left_tracks, key=len))
            for y, x in enumerate(left):
                mask[y, :max(0, int(round(x)) - margin)] = 0
        if right_tracks:
            right = boundary(max(right_tracks, key=len))
            for y, x in enumerate(right):
                mask[y, min(width, int(round(x)) + margin + 1):] = 0
        return mask

    def _reliable_lane_track(self, track, shape):
        if len(track) < LANE_TRACK_MIN_POINTS:
            return False
        height, width = shape[:2]
        points = np.asarray(track, dtype=np.float64)
        y_span = float(np.max(points[:, 1]) - np.min(points[:, 1]))
        slope, intercept = np.polyfit(points[:, 1], points[:, 0], 1)
        error = float(np.mean(np.abs(points[:, 0] - (slope * points[:, 1] + intercept))))
        return y_span >= height * LANE_TRACK_MIN_Y_SPAN_RATIO \
            and error <= width * LANE_TRACK_MAX_ERROR_RATIO

    def _lane_models(self, lane_points, shape):
        if not lane_points:
            return []
        tracks = [lane_points] if isinstance(lane_points[0], tuple) else lane_points
        models = []
        for track in tracks:
            if not self._reliable_lane_track(track, shape):
                continue
            points = np.asarray(track, dtype=np.float64)
            slope, intercept = np.polyfit(points[:, 1], points[:, 0], 1)
            models.append(LineModel(slope, intercept, len(points), 0.0))
        return models

    def _bar_matches_lane(self, bar, lane_models, width,
                          allow_strong_override=False):
        if self._detached_strong_bar_geometry(bar, width):
            return False
        if allow_strong_override and self._strong_bar_geometry(bar, width):
            return False
        if not lane_models:
            return False
        bar_angle = bar["angle"]
        center_x, center_y = bar["center"]
        for model in lane_models:
            lane_angle = normalize_angle(np.degrees(np.arctan2(1.0, model.slope)))
            distance = abs(center_x - model.x_at(center_y))
            if (angle_delta(bar_angle, lane_angle) <= BAR_LANE_PARALLEL_ANGLE
                    and distance <= width * BAR_LANE_DISTANCE_RATIO):
                return True
        return False

    def _stripe_candidates(self, binary):
        height, width = binary.shape[:2]
        candidates = []
        for contour in find_contours(binary):
            area = cv2.contourArea(contour)
            if area < STRIPE_MIN_AREA:
                continue
            (cx, cy), (rw, rh), _ = cv2.minAreaRect(contour)
            long_side = max(rw, rh)
            short_side = max(1.0, min(rw, rh))
            fill = area / float(rw * rh + 1.0)
            ratio = long_side / short_side
            if not (STRIPE_RATIO_MIN <= ratio <= STRIPE_RATIO_MAX
                    and width * STRIPE_SHORT_MIN_RATIO <= short_side
                    <= width * STRIPE_SHORT_MAX_RATIO
                    and height * STRIPE_LONG_MIN_RATIO <= long_side
                    <= height * STRIPE_LONG_MAX_RATIO
                    and fill > STRIPE_MIN_FILL):
                continue
            polygon = np.rint(cv2.boxPoints(((cx, cy), (rw, rh),
                                              cv2.minAreaRect(contour)[2]))).astype(np.int32)
            candidates.append({"center": (float(cx), float(cy)),
                               "long": float(long_side), "short": float(short_side),
                               "bottom": int(np.max(polygon[:, 1])),
                               "angle": polygon_long_angle(polygon),
                               "polygon": polygon.tolist()})
        return candidates

    def _stripe_group(self, candidates, shape):
        height, width = shape[:2]
        best = []
        ordered = sorted(
            [item for item in candidates
             if width * STRIPE_CENTER_X_MIN_RATIO <= item["center"][0]
             <= width * STRIPE_CENTER_X_MAX_RATIO],
            key=lambda item: item["center"][0],
        )

        def neighbors(first, second):
            return (
                0 < second["center"][0] - first["center"][0]
                <= width * STRIPE_GROUP_MAX_GAP_RATIO
                and abs(second["center"][1] - first["center"][1])
                <= height * STRIPE_GROUP_Y_RATIO
                and angle_delta(second["angle"], first["angle"])
                <= STRIPE_GROUP_ANGLE
                and STRIPE_GROUP_SIZE_MIN <= second["long"] / first["long"]
                <= STRIPE_GROUP_SIZE_MAX
                and STRIPE_GROUP_SIZE_MIN <= second["short"] / first["short"]
                <= STRIPE_GROUP_SIZE_MAX
            )

        for start in range(len(ordered)):
            group = [ordered[start]]
            for item in ordered[start + 1:]:
                if neighbors(group[-1], item):
                    group.append(item)
                elif item["center"][0] - group[-1]["center"][0] \
                        > width * STRIPE_GROUP_MAX_GAP_RATIO:
                    break
            if len(group) > len(best):
                best = group
        best.sort(key=lambda item: item["center"][0])
        if (len(best) >= STRIPE_STRONG_COUNT
                and best[-1]["center"][0] - best[0]["center"][0]
                >= width * STRIPE_GROUP_MIN_SPAN):
            return best
        return best if 1 <= len(best) <= 2 else []

    def _bar_polygon(self, binary, line):
        """沿 Hough 线法线寻找实际白色带，校正横条中心和厚度。"""
        height, width = binary.shape[:2]
        x1, y1, x2, y2 = [float(value) for value in line]
        length = float(np.hypot(x2 - x1, y2 - y1))
        angle = normalize_angle(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if length < 1.0:
            return np.rint(cv2.boxPoints(((x1, y1), (1.0, 1.0), angle))).astype(np.int32)

        sample_count = max(20, int(length * 0.8))
        along = np.linspace(0.08, 0.92, sample_count)
        base_x = x1 + (x2 - x1) * along
        base_y = y1 + (y2 - y1) * along
        normal_x = -(y2 - y1) / length
        normal_y = (x2 - x1) / length
        search = max(12, int(width * BAR_THICKNESS_SEARCH_RATIO))
        offsets = np.arange(-search, search + 1, dtype=np.int32)
        occupied = []
        for offset in offsets:
            xs = np.rint(base_x + normal_x * offset).astype(np.int32)
            ys = np.rint(base_y + normal_y * offset).astype(np.int32)
            inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            ratio = 0.0 if not np.any(inside) else float(
                np.mean(binary[ys[inside], xs[inside]] > 0)
            )
            occupied.append(ratio >= BAR_THICKNESS_MIN_OCCUPANCY)

        runs = []
        start = None
        for index, value in enumerate(occupied + [False]):
            if value and start is None:
                start = index
            elif not value and start is not None:
                runs.append((start, index - 1))
                start = None
        if runs:
            first, last = max(runs, key=lambda run: run[1] - run[0])
            center_offset = (float(offsets[first]) + float(offsets[last])) * 0.5
            thickness = max(3.0, float(offsets[last] - offsets[first] + 1))
        else:
            center_offset = 0.0
            thickness = max(8.0, width * BAR_DEFAULT_THICKNESS_RATIO)

        center_x = (x1 + x2) * 0.5 + normal_x * center_offset
        center_y = (y1 + y2) * 0.5 + normal_y * center_offset
        return np.rint(cv2.boxPoints(((center_x, center_y),
                                      (length, thickness), angle))).astype(np.int32)

    def _merge_hough_lines(self, lines, width):
        """合并方向和位置接近的共线片段，避免横条只框住其中一半。"""
        pending = []
        for line in lines:
            x1, y1, x2, y2 = [float(value) for value in line]
            if x2 < x1:
                x1, y1, x2, y2 = x2, y2, x1, y1
            length = float(np.hypot(x2 - x1, y2 - y1))
            angle = normalize_angle(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            pending.append({"line": (x1, y1, x2, y2), "length": length,
                            "angle": angle})
        pending.sort(key=lambda item: item["length"], reverse=True)
        groups = []
        angle_limit = BAR_MERGE_ANGLE
        distance_limit = max(10.0, width * BAR_MERGE_DISTANCE_RATIO)
        gap_limit = width * BAR_MERGE_GAP_RATIO
        for item in pending:
            x1, y1, x2, y2 = item["line"]
            midpoint = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5])
            target = None
            for group in groups:
                direction = group["direction"]
                normal = np.array([-direction[1], direction[0]])
                distance = abs(float(np.dot(midpoint - group["origin"], normal)))
                projections = [float(np.dot(np.array(point) - group["origin"], direction))
                               for point in ((x1, y1), (x2, y2))]
                gap = max(0.0, group["min_t"] - max(projections),
                          min(projections) - group["max_t"])
                if (angle_delta(item["angle"], group["angle"]) <= angle_limit
                        and distance <= distance_limit and gap <= gap_limit):
                    target = group
                    break
            if target is None:
                radians = np.radians(item["angle"])
                direction = np.array([np.cos(radians), np.sin(radians)])
                origin = midpoint
                projections = [float(np.dot(np.array(point) - origin, direction))
                               for point in ((x1, y1), (x2, y2))]
                groups.append({"points": [(x1, y1), (x2, y2)], "origin": origin,
                               "direction": direction, "angle": item["angle"],
                               "min_t": min(projections), "max_t": max(projections)})
                continue
            target["points"].extend([(x1, y1), (x2, y2)])
            points = np.asarray(target["points"], dtype=np.float32)
            vx, vy, ox, oy = [float(value) for value in cv2.fitLine(
                points, cv2.DIST_L2, 0, 0.01, 0.01
            ).reshape(-1)]
            if vx < 0:
                vx, vy = -vx, -vy
            target["origin"] = np.array([ox, oy])
            target["direction"] = np.array([vx, vy])
            target["angle"] = normalize_angle(np.degrees(np.arctan2(vy, vx)))
            projections = np.dot(points - target["origin"], target["direction"])
            target["min_t"], target["max_t"] = float(np.min(projections)), float(np.max(projections))

        merged = []
        for group in groups:
            start = group["origin"] + group["direction"] * group["min_t"]
            end = group["origin"] + group["direction"] * group["max_t"]
            merged.append(tuple(np.rint(np.concatenate((start, end))).astype(np.int32)))
        return merged

    def _hough_bars(self, binary, stripes):
        height, width = binary.shape[:2]
        lines = cv2.HoughLinesP(binary, 1, np.pi / 180.0,
                                threshold=max(28, int(width * BAR_HOUGH_THRESHOLD_RATIO)),
                                minLineLength=max(55, int(width * BAR_HOUGH_MIN_LENGTH_RATIO)),
                                maxLineGap=max(14, int(width * BAR_HOUGH_MAX_GAP_RATIO)))
        bars = []
        if lines is None:
            return bars
        valid_lines = []
        for x1, y1, x2, y2 in lines[:, 0, :]:
            length = float(np.hypot(x2 - x1, y2 - y1))
            angle = normalize_angle(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if (length < width * BAR_HOUGH_MIN_LENGTH_RATIO
                    or abs(angle) > BAR_MAX_ABS_ANGLE):
                continue
            valid_lines.append((x1, y1, x2, y2))
        for x1, y1, x2, y2 in self._merge_hough_lines(valid_lines, width):
            length = float(np.hypot(x2 - x1, y2 - y1))
            angle = normalize_angle(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            polygon = self._bar_polygon(binary, (x1, y1, x2, y2))
            center_x = float(np.mean(polygon[:, 0]))
            center_y = float(np.mean(polygon[:, 1]))
            item = {"center": (center_x, center_y), "length": length, "angle": angle,
                    "bottom": int(np.max(polygon[:, 1])), "polygon": polygon.tolist()}
            matched = []
            min_x, max_x = sorted((x1, x2))
            for stripe in stripes:
                sx = stripe["center"][0]
                if not (min_x - width * BAR_STRIPE_X_MARGIN_RATIO
                        <= sx <= max_x + width * BAR_STRIPE_X_MARGIN_RATIO):
                    continue
                radians = np.radians(angle)
                stop_y = center_y + np.tan(radians) * (sx - center_x)
                stripe_polygon = np.asarray(
                    stripe["polygon"], dtype=np.float32
                )
                stripe_top = float(np.min(stripe_polygon[:, 1]))
                near_bottom = (
                    stripe["bottom"] - height * BAR_STRIPE_Y_ABOVE_RATIO
                    <= stop_y
                    <= stripe["bottom"] + height * BAR_STRIPE_Y_BELOW_RATIO
                )
                near_top = (
                    stripe_top - height * BAR_STRIPE_TOP_ABOVE_RATIO
                    <= stop_y
                    <= stripe_top + height * BAR_STRIPE_TOP_BELOW_RATIO
                )
                if ((near_bottom or near_top)
                        and angle_delta(stripe["angle"], angle)
                        >= BAR_STRIPE_MIN_ANGLE):
                    matched.append(stripe)
            item["matched"] = matched
            bars.append(item)
        return bars

    def _same_bar(self, first, second, shape):
        return same_tracked_bar(first, second, shape)

    def _select_bar(self, bars, shape):
        if not bars:
            return None
        if self.last_bar is not None:
            continuous = [bar for bar in bars if self._same_bar(bar, self.last_bar, shape)]
            if continuous:
                return max(continuous, key=lambda bar: (len(bar["matched"]), bar["length"]))
            if self.bar_locked:
                return None
        return max(bars, key=lambda bar: (len(bar["matched"]), bar["length"],
                                          bar["bottom"]))

    def _bar_only_valid(self, bar, shape):
        height, width = shape[:2]
        polygon = np.asarray(bar["polygon"], dtype=np.float32)
        (_, _), (side_a, side_b), _ = cv2.minAreaRect(polygon)
        thickness = min(side_a, side_b)
        return (
            bar["length"] >= width * BAR_ONLY_MIN_LENGTH_RATIO
            and width * BAR_ONLY_MIN_THICKNESS_RATIO <= thickness
            <= width * BAR_ONLY_MAX_THICKNESS_RATIO
            and bar["bottom"] >= height * BAR_TRACK_MIN_BOTTOM_RATIO
            and abs(bar["angle"]) <= BAR_ONLY_MAX_ABS_ANGLE
            and self._crosses_vehicle_axis(bar, width)
        )

    def _smooth_bar(self, current, previous):
        keep = BAR_TRACK_SMOOTH
        center_x = keep * previous["center"][0] + (1.0 - keep) * current["center"][0]
        center_y = keep * previous["center"][1] + (1.0 - keep) * current["center"][1]
        angle = keep * previous["angle"] + (1.0 - keep) * current["angle"]
        polygon = np.rint(cv2.boxPoints(((center_x, center_y),
                                          (current["length"], self._bar_thickness(current)),
                                          angle))).astype(np.int32)
        smoothed = dict(current)
        smoothed.update({"center": (center_x, center_y), "angle": angle,
                         "bottom": int(np.max(polygon[:, 1])),
                         "polygon": polygon.tolist()})
        return smoothed

    def _bar_thickness(self, bar):
        polygon = np.asarray(bar["polygon"], dtype=np.float32)
        (_, _), (side_a, side_b), _ = cv2.minAreaRect(polygon)
        return max(3.0, min(side_a, side_b))

    def _crosses_vehicle_axis(self, bar, width):
        polygon = np.asarray(bar["polygon"], dtype=np.float32)
        margin = width * BAR_STRONG_AXIS_MARGIN_RATIO
        return (float(np.min(polygon[:, 0])) <= width * 0.5 + margin
                and float(np.max(polygon[:, 0])) >= width * 0.5 - margin)

    def _strong_bar_geometry(self, bar, width):
        matched = bar.get("matched", [])
        if len(matched) < BAR_STRONG_MIN_MATCHED:
            return False
        stripe_xs = [stripe["center"][0] for stripe in matched]
        return (
            max(stripe_xs) - min(stripe_xs)
            >= width * STRIPE_GROUP_MIN_SPAN
            and self._crosses_vehicle_axis(bar, width)
        )

    def _detached_strong_bar_geometry(self, bar, width):
        """只信任位于多根斑马条同一侧、且不贴着条纹端点的强横条。"""
        if not self._strong_bar_geometry(bar, width):
            return False
        center_x, center_y = bar["center"]
        slope = np.tan(np.radians(bar["angle"]))
        sides = {"above": [], "below": []}
        for stripe in bar.get("matched", []):
            polygon = stripe.get("polygon")
            short_side = stripe.get("short")
            if not polygon or short_side is None:
                continue
            if angle_delta(stripe["angle"], bar["angle"]) \
                    < BAR_LANE_OVERRIDE_MIN_ANGLE:
                continue
            stripe_points = np.asarray(polygon, dtype=np.float32)
            stripe_top = float(np.min(stripe_points[:, 1]))
            stripe_bottom = float(np.max(stripe_points[:, 1]))
            stripe_x = float(stripe["center"][0])
            bar_y = center_y + slope * (stripe_x - center_x)
            min_gap = max(3.0, float(short_side) * BAR_LANE_OVERRIDE_GAP_RATIO)
            if bar_y <= stripe_top - min_gap:
                sides["above"].append(stripe)
            elif bar_y >= stripe_bottom + min_gap:
                sides["below"].append(stripe)

        for stripes in sides.values():
            if len(stripes) < BAR_STRONG_MIN_MATCHED:
                continue
            stripe_xs = [stripe["center"][0] for stripe in stripes]
            if max(stripe_xs) - min(stripe_xs) \
                    >= width * STRIPE_GROUP_MIN_SPAN:
                return True
        return False

    def detect(self, binary, lane_points=None,
               allow_strong_lane_override=False):
        result = CrosswalkResult()
        drivable_mask = self._drivable_mask(binary.shape, lane_points)
        detection_binary = cv2.bitwise_and(binary, drivable_mask)
        candidates = self._stripe_candidates(detection_binary)
        stripes = self._stripe_group(candidates, detection_binary.shape)
        bars = self._hough_bars(detection_binary, stripes)
        height, width = binary.shape[:2]

        full_candidates = self._stripe_candidates(binary)
        full_stripes = self._stripe_group(full_candidates, binary.shape)
        lane_models = self._lane_models(lane_points, binary.shape)
        bars = [bar for bar in bars
                if not self._bar_matches_lane(
                    bar, lane_models, width, allow_strong_lane_override
                )]
        if len(full_stripes) >= STRIPE_STRONG_COUNT:
            full_bars = self._hough_bars(binary, full_stripes)
            detached = [bar for bar in full_bars
                        if self._detached_strong_bar_geometry(bar, width)]
            fallback = detached or [bar for bar in full_bars
                                    if self._strong_bar_geometry(bar, width)]
            fallback = [bar for bar in fallback
                        if not self._bar_matches_lane(
                            bar, lane_models, width,
                            allow_strong_lane_override,
                        )]
            if fallback:
                candidates = full_candidates
                stripes = full_stripes
                bars = fallback

        result.loose_polygons = [item["polygon"] for item in candidates]
        result.stripe_polygons = [item["polygon"] for item in stripes]
        front_margin = width * BAR_FRONT_MARGIN_RATIO
        tracking = [bar for bar in bars
                    if bar["bottom"] >= height * BAR_TRACK_MIN_BOTTOM_RATIO
                    and bar["center"][0] + bar["length"] * 0.5 >= width * 0.5 - front_margin
                    and bar["center"][0] - bar["length"] * 0.5 <= width * 0.5 + front_margin]
        selected = self._select_bar(tracking, binary.shape)
        if selected is not None:
            same = self.last_bar is not None and self._same_bar(selected, self.last_bar,
                                                                binary.shape)
            if same:
                selected = self._smooth_bar(selected, self.last_bar)
            self.bar_lost_hits = 0
            self.last_bar = selected
            result.tracking_polygon = selected["polygon"]
            result.tracking_angle = selected["angle"]
            result.tracking_bottom = selected["bottom"]
        else:
            same = False
            self.bar_only_hits = 0
            self.bar_lost_hits += 1
            if self.last_bar is not None and self.bar_lost_hits <= BAR_TRACK_HOLD_FRAMES:
                result.tracking_polygon = self.last_bar["polygon"]
                result.tracking_angle = self.last_bar["angle"]
                result.tracking_bottom = self.last_bar["bottom"]
            else:
                self.last_bar = None

        enough_stripes = len(stripes) >= STRIPE_STRONG_COUNT
        strong = (selected is not None and enough_stripes
                  and self._strong_bar_geometry(selected, width))
        if selected is not None and not strong and self._bar_only_valid(selected, binary.shape):
            self.bar_only_hits = self.bar_only_hits + 1 if same else 1
        elif not strong:
            self.bar_only_hits = 0
        bar_only_confirmed = self.bar_only_hits >= BAR_ONLY_STABLE_FRAMES
        locked_confirmed = self.bar_locked and selected is not None
        locked_held = (self.bar_locked and selected is None
                       and result.tracking_polygon is not None)
        if selected is not None and (strong or bar_only_confirmed or locked_confirmed):
            result.candidate = True
            result.confidence = (min(1.0, 0.55 + len(stripes) * 0.08)
                                 if strong else (0.75 if locked_confirmed else 0.68))
            result.stop_polygon = selected["polygon"]
            result.stop_angle = selected["angle"]
            result.stop_bottom = selected["bottom"]
        elif locked_held:
            result.candidate = True
            result.confidence = 0.62
            result.stop_polygon = result.tracking_polygon
            result.stop_angle = result.tracking_angle
            result.stop_bottom = result.tracking_bottom
        else:
            temporal = min(0.55, self.bar_only_hits * 0.10)
            result.confidence = max(min(0.6, len(stripes) * 0.12), temporal)
        return result


def mask_crosswalk(binary, result, include_loose=False):
    polygons = list(result.stripe_polygons)
    if result.stop_polygon is not None:
        polygons.append(result.stop_polygon)
    if include_loose:
        polygons.extend(result.loose_polygons)
    if not polygons:
        return binary
    mask = np.zeros_like(binary)
    for polygon in polygons:
        cv2.fillConvexPoly(mask, np.asarray(polygon, dtype=np.int32), 255)
    pad = max(5, int(binary.shape[1] * 0.018))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1))
    mask = cv2.dilate(mask, kernel)
    cleaned = binary.copy()
    cleaned[mask > 0] = 0
    return cleaned


def maneuver_exit(entry_cleared, exit_hits, exit_visible, exit_bottom,
                  frame_height,
                  exit_required=EXIT_BAR_FRAMES):
    if not entry_cleared:
        return 0, False
    exit_hits = exit_hits + 1 if exit_visible else max(0, exit_hits - 1)
    near = exit_hits >= exit_required \
        and exit_bottom >= float(frame_height) * STOP_NEAR_RATIO
    return exit_hits, near


def approach_next_state(visible, bottom, frame_height, lost_hits):
    if lost_hits > LOST_LIMIT:
        return "FOLLOW"
    if visible and bottom >= float(frame_height) * STOP_NEAR_RATIO:
        return "ALIGN"
    return None


def alignment_next_state(angle, visible, align_hits, lost_hits, elapsed,
                         stripe_count):
    if align_hits >= ALIGN_STABLE_FRAMES:
        return "MANEUVER"
    if lost_hits > LOST_LIMIT:
        return "MANEUVER"
    if elapsed <= ALIGN_TIMEOUT:
        return None
    safe_timeout = (
        visible and angle is not None
        and abs(float(angle)) <= ALIGN_ENTRY_MAX_ANGLE
        and stripe_count >= ALIGN_ENTRY_MIN_STRIPES
    )
    if safe_timeout:
        return "MANEUVER"
    if visible and angle is not None and stripe_count >= ALIGN_ENTRY_MIN_STRIPES:
        return None
    return None


def exit_alignment_next_state(align_hits, lost_hits, elapsed):
    if align_hits >= ALIGN_STABLE_FRAMES:
        return "FOLLOW"
    if lost_hits >= EXIT_ALIGN_LOST_FRAMES:
        return "FOLLOW"
    return None


def maneuver_timeout_exits_to_follow(elapsed):
    return float(elapsed) >= MANEUVER_MAX_TIME


def wait_recovery_state(angle, visible, recover_hits, stripe_count):
    if visible and angle is not None and stripe_count >= ALIGN_ENTRY_MIN_STRIPES \
            and abs(float(angle)) > ALIGN_ENTRY_MAX_ANGLE:
        return "ALIGN"
    safe = (
        visible and angle is not None
        and abs(float(angle)) <= ALIGN_ENTRY_MAX_ANGLE
        and stripe_count >= ALIGN_ENTRY_MIN_STRIPES
    )
    return "MANEUVER" if safe and recover_hits >= WAIT_RECOVER_FRAMES else None


class BinaryVision(object):
    def __init__(self, black_v_max=BLACK_V_MAX):
        self.black_v_max = int(black_v_max)

    def apply(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color = cv2.inRange(hsv, np.array([0, 0, 0], np.uint8),
                            np.array([180, 255, self.black_v_max], np.uint8))
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 31, 5)
        binary = cv2.bitwise_and(color, adaptive)
        kernel = np.ones((3, 3), np.uint8)
        return cv2.morphologyEx(cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel),
                               cv2.MORPH_CLOSE, kernel)


class CameraReader(object):
    def __init__(self, index):
        backend = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else 0
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index)
        self.lock = threading.Lock()
        self.frame = None
        self.seq = 0
        self.read_seq = -1
        self.running = self.cap.isOpened()
        self.thread = None
        if self.running:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.thread = threading.Thread(target=self._loop)
            self.thread.daemon = True
            self.thread.start()

    def _loop(self):
        while self.running and not rospy.is_shutdown():
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame
                    self.seq += 1
            else:
                time.sleep(0.02)

    def read(self, timeout=0.5):
        end = time.time() + timeout
        while time.time() < end and not rospy.is_shutdown():
            with self.lock:
                if self.frame is not None and self.seq != self.read_seq:
                    self.read_seq = self.seq
                    return True, self.frame.copy()
            time.sleep(0.01)
        return False, None

    def release(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(0.8)
        self.cap.release()


class PID(object):
    def __init__(self, kp, kd, limit):
        self.kp = float(kp)
        self.kd = float(kd)
        self.limit = float(limit)
        self.last_error = 0.0
        self.last_time = None

    def reset(self):
        self.last_error = 0.0
        self.last_time = None

    def update(self, deviation, kp=None, kd=None):
        now = rospy.get_time()
        dt = 0.05 if self.last_time is None else max(0.001, now - self.last_time)
        error = -float(deviation)
        active_kp = self.kp if kp is None else float(kp)
        active_kd = self.kd if kd is None else float(kd)
        output = active_kp * error + active_kd * (error - self.last_error) / dt
        self.last_error = error
        self.last_time = now
        return clamp(output, -self.limit, self.limit)


class LaneFollower(object):
    def __init__(self):
        rospy.init_node("line_cy_new", anonymous=True)
        self.camera_index = int(rospy.get_param("~camera_index", CAMERA_INDEX))
        self.process_width = int(rospy.get_param("~process_width", PROCESS_WIDTH))
        self.dry_run = bool(rospy.get_param("~dry_run", DRY_RUN))
        self.debug_view = bool(rospy.get_param("~debug_view", DEBUG_VIEW))
        self.turn_entry_time = max(0.0, float(rospy.get_param(
            "~turn_entry_time", TURN_ENTRY_TIME
        )))
        self.turn_speed = max(0.0, float(rospy.get_param(
            "~turn_speed", TURN_SPEED
        )))
        self.turn_angular = clamp(abs(float(rospy.get_param(
            "~turn_angular", TURN_ANGULAR
        ))), 0.01, 1.0)
        self.turn_time = max(0.1, float(rospy.get_param(
            "~turn_time", TURN_TIME
        )))
        self.turn_cmd = normalize_turn_cmd(
            rospy.get_param("~turn_cmd", TURN_CMD)
        )
        rospy.loginfo(
            "line_cy_new turn_cmd=%s entry=%.2f speed=%.2f angular=%.2f "
            "turn_time=%.2f",
            self.turn_cmd, self.turn_entry_time, self.turn_speed,
            self.turn_angular, self.turn_time,
        )
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.vision = BinaryVision()
        self.lanes = LaneDetector(fill_width=FILL_WIDTH_PIXELS)
        self.crosswalk = CrosswalkDetector()
        self.camera = CameraReader(self.camera_index)
        self.pid = PID(KP, KD, MAX_ANGULAR)
        self.lane_width = LANE_WIDTH_PIXELS if LANE_WIDTH_PIXELS > 0 else PROCESS_WIDTH * DEFAULT_LANE_WIDTH_RATIO
        self.bridge = DualLineBridge(self.lane_width, fill_width=FILL_WIDTH_PIXELS)
        self.state = "FOLLOW"
        self.state_started = rospy.get_time()
        self.stop_hits = self.lost_hits = self.align_hits = 0
        self.wait_recover_hits = 0
        self.clear_hits = self.exit_hits = self.dual_hits = 0
        self.entry_cleared = False
        self.maneuver_timeout_warned = False
        self.maneuver_phase = "NONE"
        self.maneuver_phase_started = self.state_started
        self.entry_accept_after = 0.0
        self.last_angular = 0.0
        self.last_control_target = None
        self.last_observation = None
        self.last_crosswalk = CrosswalkResult()
        self.last_binary = None
        self.cleaned = False
        if not self.camera.cap.isOpened():
            rospy.signal_shutdown("cannot open lane camera")
        rospy.on_shutdown(self.cleanup)

    def _resize(self, frame):
        height, width = frame.shape[:2]
        if self.process_width <= 0 or width <= self.process_width:
            return frame
        scale = float(self.process_width) / width
        return cv2.resize(frame, (self.process_width, int(height * scale)), interpolation=cv2.INTER_AREA)

    def publish(self, linear, angular):
        angular_limit = MAX_ANGULAR
        if (getattr(self, "state", None) == "MANEUVER"
                and getattr(self, "maneuver_phase", None) == "TURN"):
            angular_limit = max(MAX_ANGULAR, abs(self.turn_angular))
        angular = clamp(float(angular), -angular_limit, angular_limit)
        command = Twist()
        command.linear.x = float(linear)
        command.linear.y = command.linear.z = 0.0
        command.angular.x = command.angular.y = 0.0
        command.angular.z = angular
        if motion_enabled(self.dry_run):
            self.pub.publish(command)

    def _control(self, center_x, width, speed, bias_pixels=0.0):
        target_x = control_target_x(center_x, bias_pixels)
        self.last_control_target = target_x
        deviation = target_x - width * 0.5
        kp, kd, _ = pd_gains(deviation)
        raw = self.pid.update(deviation, kp, kd)
        angular = ANGULAR_SMOOTH * self.last_angular + (1.0 - ANGULAR_SMOOTH) * raw
        self.last_angular = angular
        self.publish(speed, angular)

    def _set_maneuver_phase(self, phase, now=None):
        if phase == self.maneuver_phase:
            return
        rospy.loginfo("line_cy_new maneuver phase: %s -> %s",
                      self.maneuver_phase, phase)
        self.maneuver_phase = phase
        self.maneuver_phase_started = rospy.get_time() if now is None else float(now)
        self.pid.reset()
        self.last_angular = 0.0
        self.last_control_target = None

    def _run_timed_turn_phase(self, now):
        elapsed = float(now) - self.maneuver_phase_started
        next_phase = turn_phase_next(
            self.maneuver_phase, elapsed,
            self.turn_entry_time, self.turn_time,
        )
        if next_phase is not None:
            self._set_maneuver_phase(next_phase, now)

        if self.maneuver_phase in ("ENTRY", "EXIT_STRAIGHT"):
            self.publish(self.turn_speed, 0.0)
        elif self.maneuver_phase == "TURN":
            linear, angular = fixed_turn_command(
                self.turn_cmd, self.turn_speed, self.turn_angular
            )
            self.publish(linear, angular)
        else:
            self.publish(0, 0)

    def _set_state(self, state):
        if state == self.state:
            return
        previous_state = self.state
        rospy.loginfo("line_cy_new state: %s -> %s", previous_state, state)
        self.state = state
        self.state_started = rospy.get_time()
        self.pid.reset()
        self.last_angular = 0.0
        self.last_control_target = None
        self.lost_hits = self.align_hits = 0
        if (state == "FOLLOW"
                and previous_state in ("EXIT_ALIGN", "MANEUVER")):
            self.stop_hits = 0
            self.entry_accept_after = (
                self.state_started + EXIT_ENTRY_IGNORE_TIME
            )
        if state in ("FOLLOW", "MANEUVER"):
            self.crosswalk.unlock_bar()
        if state == "MANEUVER":
            self.entry_cleared = False
            self.clear_hits = self.exit_hits = self.dual_hits = 0
            self.maneuver_timeout_warned = False
            self.maneuver_phase = (
                "ENTRY" if maneuver_follow_side(self.turn_cmd) is not None
                else "STRAIGHT"
            )
            self.maneuver_phase_started = self.state_started
        else:
            self.maneuver_phase = "NONE"

    def _update_lane_width(self, observation, frame_width):
        measured = observation.measured_width
        if LANE_WIDTH_PIXELS > 0 or measured is None:
            return
        if frame_width * MIN_LANE_WIDTH_RATIO <= measured <= frame_width * MAX_LANE_WIDTH_RATIO:
            self.lane_width = 0.90 * self.lane_width + 0.10 * measured
            self.bridge.lane_width = self.lane_width

    def _update_bridge(self, binary, center_hint=None, target_y=None):
        height, width = binary.shape[:2]
        raw_left, raw_right = self.lanes.points(binary, center_hint)
        if self.bridge.left_model is None:
            left_points = raw_left
        else:
            left_points = self.lanes.points_near_model(binary, self.bridge.left_model)
        if self.bridge.right_model is None:
            right_points = raw_right
        else:
            right_points = self.lanes.points_near_model(binary, self.bridge.right_model)
        target_y = int(height * ROI_BOTTOM) if target_y is None else int(target_y)
        return self.bridge.update(
            left_points, right_points, target_y, center_hint,
            frame_width=width, validation_top_y=int(height * ROI_TOP),
        )

    def process(self, raw_frame):
        frame = self._resize(raw_frame)
        binary = self.vision.apply(frame)
        current_left, current_right = self.lanes.points(binary)
        lane_tracks = [current_left, current_right]
        if self.last_observation is not None:
            lane_tracks.extend([self.last_observation.left_points,
                                self.last_observation.right_points])
        allow_strong_lane_override = strong_lane_override_enabled(
            self.state, self.turn_cmd, self.maneuver_phase
        )
        cross = self.crosswalk.detect(
            binary, lane_points=lane_tracks,
            allow_strong_lane_override=allow_strong_lane_override,
        )
        lane_binary = mask_crosswalk(binary, cross)
        observation = self.lanes.observe(lane_binary, self.lane_width)
        self._update_lane_width(observation, frame.shape[1])
        self.last_crosswalk = cross
        self.last_observation = observation
        self.last_binary = lane_binary
        now = rospy.get_time()

        if self.state == "FOLLOW":
            entry_allowed = entry_acceptance_enabled(
                now, getattr(self, "entry_accept_after", 0.0)
            )
            if entry_allowed:
                entry_candidate = cross.candidate
            else:
                entry_candidate = False
            self.stop_hits = follow_entry_hits(
                entry_candidate, self.stop_hits
            )
            if self.stop_hits >= STOP_STABLE_FRAMES:
                self.stop_hits = 0
                self.crosswalk.lock_current_bar()
                self.bridge.reset(self.lane_width)
                self._set_state("APPROACH")
            if observation.valid:
                self._control(observation.center_x, frame.shape[1], FOLLOW_SPEED,
                              FOLLOW_CENTER_BIAS_PIXELS)
            else:
                self.publish(0, 0)

        elif self.state == "APPROACH":
            bridge_binary = mask_crosswalk(binary, cross, include_loose=True)
            self._update_bridge(bridge_binary, frame.shape[1] * 0.5)
            # tracking_polygon 只是 Hough 跟踪结果，未必通过纯横条几何校验。
            visible = cross.candidate
            self.lost_hits = 0 if visible else self.lost_hits + 1
            bottom = polygon_bottom_in_center_band(
                cross.stop_polygon, frame.shape[1]
            ) if visible else 0
            next_state = approach_next_state(
                visible, bottom, frame.shape[0], self.lost_hits
            )
            if next_state == "FOLLOW":
                self.stop_hits = 0
                self._set_state("FOLLOW")
                if observation.valid:
                    self._control(observation.center_x, frame.shape[1], FOLLOW_SPEED,
                                  FOLLOW_CENTER_BIAS_PIXELS)
                else:
                    self.publish(0, 0)
            elif next_state == "ALIGN":
                self._set_state("ALIGN")
                self.publish(0, 0)
            elif observation.valid:
                self._control(observation.center_x, frame.shape[1], APPROACH_SPEED)
            else:
                self.publish(0, 0)

        elif self.state == "ALIGN":
            bridge_binary = mask_crosswalk(binary, cross, include_loose=True)
            self._update_bridge(bridge_binary, frame.shape[1] * 0.5)
            angle = cross.stop_angle if cross.candidate else cross.tracking_angle
            visible = angle is not None
            if angle is None:
                self.lost_hits += 1
                self.publish(0, 0)
            elif abs(angle) <= ALIGN_TOLERANCE_DEG:
                self.lost_hits = 0
                self.align_hits += 1
                self.publish(0, 0)
            else:
                self.lost_hits = 0
                self.align_hits = 0
                magnitude = clamp(abs(angle) * ALIGN_KP, ALIGN_MIN_ANGULAR, ALIGN_MAX_ANGULAR)
                self.publish(0, -magnitude if angle > 0 else magnitude)
            next_state = alignment_next_state(
                angle, visible, self.align_hits, self.lost_hits,
                now - self.state_started, len(cross.stripe_polygons),
            )
            if next_state is not None:
                self._set_state(next_state)

        elif self.state == "EXIT_ALIGN":
            angle = cross.stop_angle if cross.candidate else cross.tracking_angle
            visible = angle is not None
            if angle is None:
                self.lost_hits += 1
                self.publish(0, 0)
            elif abs(angle) <= ALIGN_TOLERANCE_DEG:
                self.lost_hits = 0
                self.align_hits += 1
                self.publish(0, 0)
            else:
                self.lost_hits = 0
                self.align_hits = 0
                magnitude = clamp(abs(angle) * ALIGN_KP,
                                  ALIGN_MIN_ANGULAR, ALIGN_MAX_ANGULAR)
                self.publish(0, -magnitude if angle > 0 else magnitude)
            next_state = exit_alignment_next_state(
                self.align_hits, self.lost_hits, now - self.state_started
            )
            if next_state is not None:
                self._set_state(next_state)
                if observation.valid:
                    self._control(
                        observation.center_x, frame.shape[1], FOLLOW_SPEED,
                        FOLLOW_CENTER_BIAS_PIXELS,
                    )
                else:
                    self.publish(0, 0)

        elif self.state == "WAIT":
            angle = cross.stop_angle if cross.candidate else cross.tracking_angle
            visible = angle is not None
            safe_visible = (
                visible and len(cross.stripe_polygons) >= ALIGN_ENTRY_MIN_STRIPES
                and abs(float(angle)) <= ALIGN_ENTRY_MAX_ANGLE
            )
            self.wait_recover_hits = self.wait_recover_hits + 1 if safe_visible else 0
            next_state = wait_recovery_state(
                angle, visible, self.wait_recover_hits,
                len(cross.stripe_polygons),
            )
            self.publish(0, 0)
            if next_state is not None:
                self._set_state(next_state)

        elif self.state == "MANEUVER":
            lane_binary = mask_crosswalk(binary, cross, include_loose=True)
            side = maneuver_follow_side(self.turn_cmd)
            if side is None:
                target_y = int(frame.shape[0] * MANEUVER_LOOKAHEAD_RATIO)
                center, left_model, right_model = self._update_bridge(
                    lane_binary, frame.shape[1] * 0.5, target_y
                )
                if center is None:
                    center = frame.shape[1] * 0.5
                self._control(center, frame.shape[1], MANEUVER_SPEED)
            else:
                self.last_binary = lane_binary
                self._run_timed_turn_phase(now)

            cross_visible = cross.candidate or len(cross.stripe_polygons) >= 3
            self.clear_hits = 0 if cross_visible else self.clear_hits + 1
            if self.clear_hits >= ENTRY_CLEAR_FRAMES:
                self.entry_cleared = True
            exit_ready = (side is None
                          or self.maneuver_phase == "EXIT_STRAIGHT")
            stop_bottom = polygon_bottom_in_center_band(
                cross.stop_polygon, frame.shape[1]
            ) if cross.candidate else 0
            exit_visible = exit_ready and self.entry_cleared and cross.candidate \
                and stop_bottom >= frame.shape[0] * BAR_TRACK_MIN_BOTTOM_RATIO
            self.exit_hits, exit_near = maneuver_exit(
                self.entry_cleared, self.exit_hits, exit_visible,
                stop_bottom, frame.shape[0]
            )
            if exit_near and now - self.state_started >= MANEUVER_MIN_TIME:
                self.crosswalk.lock_current_bar()
                self._set_state("EXIT_ALIGN")
            elif maneuver_timeout_exits_to_follow(now - self.state_started):
                rospy.logwarn("maneuver timeout, resume normal lane following")
                self.maneuver_timeout_warned = True
                self._set_state("FOLLOW")
                if observation.valid:
                    self._control(observation.center_x, frame.shape[1], FOLLOW_SPEED,
                                  FOLLOW_CENTER_BIAS_PIXELS)
                else:
                    self.publish(0, 0)

        else:
            self.publish(0, 0)

        if self.debug_view:
            self.draw_debug(frame)

    def draw_debug(self, frame):
        height, width = frame.shape[:2]
        observation = self.last_observation
        target_x = observation.center_x if self.last_control_target is None \
            else self.last_control_target
        virtual_display = clip_points_for_display(
            observation.virtual_left_points + observation.virtual_right_points, width
        )
        center_path = sorted(observation.center_points, key=lambda point: point[1])
        side = observation.follow_side or ("dual" if observation.dual_rows else "none")
        if self.last_crosswalk.stop_polygon is not None:
            bar_state = "stop"
        elif self.last_crosswalk.tracking_polygon is not None:
            bar_state = "track"
        else:
            bar_state = "none"
        bar_angle = (self.last_crosswalk.stop_angle
                     if self.last_crosswalk.stop_angle is not None
                     else self.last_crosswalk.tracking_angle)
        angle_text = "--" if bar_angle is None else "{:.1f}".format(bar_angle)
        text = ("state={} cmd={} phase={} side={} lane={:.0f} dual={} stripes={} "
                "bar={} angle={} hits={} cross={:.2f}").format(
            self.state, self.turn_cmd, self.maneuver_phase,
            side, self.lane_width,
            observation.dual_rows, len(self.last_crosswalk.stripe_polygons),
            bar_state, angle_text, self.crosswalk.bar_only_hits,
            self.last_crosswalk.confidence)
        try:
            cv2.imshow(WINDOW_NAME, frame)
            processed = cv2.cvtColor(self.last_binary, cv2.COLOR_GRAY2BGR)
            top, bottom = int(height * ROI_TOP), int(height * ROI_BOTTOM)
            cv2.rectangle(processed, (0, top), (width - 1, bottom), (0, 180, 0), 2)
            cv2.line(processed, (width // 2, top), (width // 2, bottom), (100, 100, 100), 1)
            stop_half_width = width * clamp(STOP_CENTER_WIDTH_RATIO, 0.0, 1.0) * 0.5
            stop_left = int(round(width * 0.5 - stop_half_width))
            stop_right = int(round(width * 0.5 + stop_half_width))
            stop_top = int(round(height * STOP_NEAR_RATIO))
            cv2.rectangle(processed, (stop_left, stop_top),
                          (stop_right, height - 1), (0, 165, 255), 2)
            for x, y in observation.left_points + observation.right_points:
                cv2.circle(processed, (x, y), 4, (255, 255, 0), -1)
            for x, y in virtual_display:
                cv2.circle(processed, (x, y), 5, (255, 0, 255), 2)
            if center_path:
                for x, y in center_path:
                    cv2.circle(processed, (int(x), int(y)), 4, (0, 255, 0), -1)
            target_display = int(clamp(target_x, 0, width - 1))
            cv2.line(processed, (target_display, bottom - 15),
                     (target_display, bottom + 15), (0, 255, 0), 3)
            for polygon in self.last_crosswalk.stripe_polygons:
                cv2.polylines(processed, [np.asarray(polygon, np.int32)], True,
                              (0, 255, 255), 2)
            if self.last_crosswalk.stop_polygon is not None:
                cv2.polylines(processed,
                              [np.asarray(self.last_crosswalk.stop_polygon, np.int32)],
                              True, (0, 0, 255), 3)
            elif (self.state in ("APPROACH", "ALIGN", "EXIT_ALIGN")
                  and self.last_crosswalk.tracking_polygon is not None):
                cv2.polylines(
                    processed,
                    [np.asarray(self.last_crosswalk.tracking_polygon, np.int32)],
                    True, (0, 128, 255), 2,
                )
            if self.state == "MANEUVER" and self.turn_cmd == "straight":
                y1, y2 = int(height * ROI_TOP), int(height * ROI_BOTTOM)
                models = (
                    (self.bridge.left_model, (255, 128, 0), 3),
                    (self.bridge.right_model, (0, 255, 0), 3),
                    (self.bridge.center_model, (255, 0, 255), 2),
                )
                for model, color, thickness in models:
                    if model is None:
                        continue
                    x1 = int(clamp(model.x_at(y1), 0, width - 1))
                    x2 = int(clamp(model.x_at(y2), 0, width - 1))
                    cv2.line(processed, (x1, y1), (x2, y2), color, thickness)
                lookahead_y = int(height * MANEUVER_LOOKAHEAD_RATIO)
                cv2.circle(processed, (target_display, lookahead_y),
                           7, (0, 0, 255), 2)
            cv2.putText(processed, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 255), 2)
            cv2.imshow(PROCESSED_WINDOW_NAME, processed)
            cv2.waitKey(1)
        except cv2.error:
            self.debug_view = False

    def run(self):
        rate = rospy.Rate(20)
        try:
            while not rospy.is_shutdown():
                ok, frame = self.camera.read(1.0)
                if ok:
                    self.process(frame)
                else:
                    self.publish(0, 0)
                rate.sleep()
        finally:
            self.cleanup()

    def cleanup(self):
        if self.cleaned:
            return
        self.cleaned = True
        try:
            self.publish(0, 0)
            self.camera.release()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        LaneFollower().run()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        pass
