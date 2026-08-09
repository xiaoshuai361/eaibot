#!/usr/bin/env python
# coding=utf-8
"""九路口循迹任务：右、直、右、左、直、左、右、直、右。"""
import threading
import time
import os
import shutil

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Twist

try:
    from traffic_light_vision import (
        TrafficLightDetector,
        configure_traffic_camera,
        draw_traffic_light,
        set_capture_resolution,
        update_green_hits,
    )
    TRAFFIC_LIGHT_MODULE_AVAILABLE = True
except ImportError as traffic_light_import_error:
    # 允许只部署主程序；未部署红绿灯模块时仍可运行原巡线任务。
    TRAFFIC_LIGHT_MODULE_AVAILABLE = False
    TRAFFIC_LIGHT_IMPORT_ERROR = traffic_light_import_error
    TrafficLightDetector = None

    def configure_traffic_camera(_camera_index):
        return False

    def set_capture_resolution(capture, width, height):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

    def draw_traffic_light(frame, _detections, _color, _green_hits,
                           _green_required):
        return frame

    def update_green_hits(_detections, _current_hits, _required_hits):
        return 0, False, None


# ===== 现场启动、摄像头与功能开关（运行前优先检查） =====
CAMERA_INDEX = 4          # 巡线摄像头：/dev/video4。
YOLO_CAMERA_INDEX = 2     # 人偶、垃圾桶和楼宇识别摄像头：/dev/video2。
TRAFFIC_LIGHT_CAMERA_INDEX = 0 # 红绿灯识别摄像头：/dev/video0，仅停止线等待时使用。
PROCESS_WIDTH = 640       # 巡线处理图宽度；改动后 PID 和像素距离需要重调。
TRAFFIC_LIGHT_FRAME_WIDTH = 320 # 红绿灯摄像头采集宽度。
TRAFFIC_LIGHT_FRAME_HEIGHT = 240 # 红绿灯摄像头采集高度。
DRY_RUN = False           # False 发布速度；True 只识别不动车。
DEBUG_VIEW = True         # True 显示巡线、任务 YOLO 和红绿灯调试窗口。
YOLO_ENABLED = True       # 是否启用人偶、垃圾桶和楼宇任务识别。
YOLO_STOP_ENABLED = True  # True 时任务目标进入中央区域会停车并记录。
TRAFFIC_LIGHT_ENABLED = True # True 时每个入口横条摆正后必须确认绿灯。
TRAFFIC_LIGHT_CONFIDENCE = 0.55 # 红绿灯单帧最低置信度；漏检可降，误检可加。

# ===== 模型路径与路线切换（部署前检查） =====
YOLO_STREET_MODEL_PATH = "/home/eaibot/handeye-calib/src/model/yolov5/rub_roll_new_yolov5n_320_best.onnx"
YOLO_BUILDING_MODEL_PATH = "/home/eaibot/handeye-calib/src/model/yolov5/building_new_yolov5n_320_best.onnx"
YOLO_MODEL_PATH = YOLO_STREET_MODEL_PATH # 兼容旧 ROS 参数；启动时加载人偶和垃圾桶模型。
YOLO_BUILDING_SWITCH_INDEX = 3 # 第三个右转完成后、第四个左转前切换楼宇模型。
TRAFFIC_LIGHT_MODEL_PATH = "/home/eaibot/handeye-calib/src/model/yolov5/traffic_lights_yolov5n_320_best.onnx"

# ===== 场地光照快速调参（固定曝光后再调整） =====
BLACK_V_MAX = 160         # 黑线断裂就加；阴影和地面杂物变多就降。
ADAPTIVE_BLOCK_SIZE = 31  # 局部阈值窗口，必须为大于 1 的奇数；光照变化范围大可加。
ADAPTIVE_C = 5            # 自适应阈值偏移；噪声多可加，细黑线漏检可减。
MORPH_KERNEL_SIZE = 3     # 开闭运算核；噪点多可加，细线被吃掉就减。
STRIPE_MIN_AREA = 25      # 斑马条断裂或远处漏检就减；噪点多就加。
STRIPE_MIN_FILL = 0.25    # 斑马条破碎漏检就减；空心杂物误识别就加。
BAR_HOUGH_THRESHOLD_RATIO = 0.055 # 横条断裂漏检就减；杂线多就加。
BAR_HOUGH_MAX_GAP_RATIO = 0.05 # 横条受反光断开就加；杂线被乱连就减。
BAR_THICKNESS_MIN_OCCUPANCY = 0.35 # 横条断裂就减；噪声使横条虚增厚就加。

# ===== 固定任务路线 =====
TASK_TURN_COMMANDS = (
    "right", "straight", "right",
    "left", "straight", "left",
    "right", "straight", "right",
)

# ===== 路口时序快速调参 =====
STOP_STABLE_FRAMES = 3    # 入口横条确认帧数；误触发就加。
ENTRY_MIN_STRIPES = 1     # 入口最少斑马条数；边线误报就加。
ALIGN_STABLE_FRAMES = 8   # 摆正稳定帧数；容易误通过就加。
ALIGN_LOCK_SETTLE_TIME = 0.15 # 横条丢失补转后的稳定时间(s)。
ALIGN_OPEN_LOOP_TIME_SCALE = 1.0 # 丢失后按最后角度补转；不额外超调。
ALIGN_OPEN_LOOP_MIN_TIME = 0.05 # 横条丢失后最短补转时间(s)。
ALIGN_OPEN_LOOP_MAX_TIME = 5.0 # 横条丢失后最长补转时间(s)。
ALIGN_TIMEOUT = 8.0       # 入口摆正最长等待时间(s)。
LOST_LIMIT = 7            # 入口横条允许丢失帧数；偶发丢线就加。
EXIT_ALIGN_LOST_FRAMES = 5 # 出口横条丢失多少帧后恢复巡线。
WAIT_RECOVER_FRAMES = 3   # 等待状态重新确认入口的帧数。
TURN_ENTRY_TIME = 6.5     # 摆正后盲区直行时间(s)；起转太早加，太晚减。
TURN_TIME = 3.6           # 固定转弯时间(s)；转不够加，转过头减。
MANEUVER_MIN_TIME = 1.0   # 路口最短通过时间(s)；过早识别出口就加。
MANEUVER_MAX_TIME = 14.0  # 路口最长通过时间(s)；出口漏检后恢复巡线。
ENTRY_CLEAR_FRAMES = 5    # 入口横条消失确认帧数；入口被当出口就加。
EXIT_BAR_FRAMES = 1       # 出口横条确认帧数；误触发就加。
EXIT_ENTRY_IGNORE_TIME = 2.0 # 出口完成后忽略横条时间(s)；重复触发就加。
FINAL_EXIT_TIME = 6.0     # 第九次出口摆正后继续直行时间(s)。

# ===== 速度与转向 =====
FOLLOW_SPEED = 0.16       # 普通巡线 linear.x 前进速度(m/s)；整车过弯慢可小幅加。
APPROACH_SPEED = 0.16     # 靠近横条 linear.x 前进速度(m/s)；冲过横条就降。
MANEUVER_SPEED = 0.16     # 路口内 linear.x 前进速度(m/s)；路口通过慢可小幅加。
MANEUVER_CENTER_BIAS_PIXELS = 40.0 # 直行路口避障量；只填正数，数值越大避让越多。
MAX_ANGULAR = 0.50       # angular.z 偏航角速度上限(rad/s)；只影响转头快慢，不提高前进速度。
FOLLOW_LEFT_ANGULAR_SCALE = 0.74 # 所有巡线左弯力度；左弯太猛就减小。
FOLLOW_RIGHT_ANGULAR_SCALE = 1.00 # 巡线右弯力度；右弯正常保持 1.0。

# 左右转固定控制；直行路口和普通巡线不使用这两项。
TURN_SPEED = 0.16         # 盲区直行和固定转弯线速度(m/s)。
TURN_ANGULAR = 0.58        # 固定转弯角速度绝对值(rad/s)；越大转弯半径越小。

KP = 0.0015              # 小误差比例；直线摆动就降，轻微修正不够就加。
KD = 0.0008              # 小误差阻尼；直线摆动就加，反应迟钝或尖峰大时降。
LARGE_ERROR_THRESHOLD_PIXELS = 120.0 # 误差达到此值切换急转 PD；太晚切换就减小。
LARGE_ERROR_KP = 0.0028  # 大误差比例；急弯拐不过就加，转得过猛就减。
LARGE_ERROR_KD = 0.01  # 大误差阻尼；急弯摆动就加，响应尖峰过大就减。
ANGULAR_SMOOTH = 0.80    # 转向保留比例；加大更平稳但迟钝，减小更灵敏。

# ===== 车道边线 =====
ROI_TOP = 0.2           # 识别区域上边界；减小看得更远，但更容易收到远处干扰。
ROI_BOTTOM = 0.92       # 识别区域下边界；增大看得更近，车头遮挡或噪声多就减小。
LANE_WIDTH_PIXELS = 620.0 # 车道内边缘间距；按当前 640 宽处理图估算，实测后可微调。
FILL_WIDTH_PIXELS = 620.0 # 路口直行模型补线间距；路口内偏移时再调。
LEFT_FILL_WIDTH_PIXELS = 620.0 # 只看到左边线时使用；增大会让目标向右移。
RIGHT_FILL_WIDTH_PIXELS = 620.0 # 只看到右边线时使用；减小会让目标向右移。
FOLLOW_CENTER_BIAS_PIXELS = 0.0 # 巡线目标横向偏置；正数向右、负数向左，固定偏航先调这里。
SCAN_ROWS = 9                 # 水平扫描行数；加大更稳但稍慢，过少容易漏线。
LANE_CENTER_NEAR_WEIGHT = 3.0 # 最下方中心点权重；弯道切内线就加。
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
STRIPE_RATIO_MIN = 1.7        # 竖条最小长宽比；方块误识别就加。
STRIPE_RATIO_MAX = 5.8        # 竖条最大长宽比；细长真实竖条漏检就加。
STRIPE_SHORT_MIN_RATIO = 0.025 # 竖条短边最小图宽比例；小噪声多就加。
STRIPE_SHORT_MAX_RATIO = 0.13 # 竖条短边最大图宽比例；大块误识别就减。
STRIPE_LONG_MIN_RATIO = 0.055 # 竖条长边最小图高比例；远处竖条漏检就减。
STRIPE_LONG_MAX_RATIO = 0.45  # 竖条长边最大图高比例；近景斑马条漏检就加。
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
BAR_HOUGH_MIN_LENGTH_RATIO = 0.16 # 单个横条片段最小图宽比例；短杂线多就加。

BAR_MAX_ABS_ANGLE = 45.0      # 横条相对画面水平最大角度；急斜拍漏检就加。
BAR_LANE_PARALLEL_ANGLE = 15.0 # 横条与边线方向接近到此值时判为边线。
BAR_LANE_DISTANCE_RATIO = 0.07 # 横条中心距边线最大图宽比例；边线误报就加，误排横条就减。
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
BAR_DEFAULT_THICKNESS_RATIO = 0.025 # 无法测厚时默认图宽比例。
BAR_ONLY_MIN_THICKNESS_RATIO = 0.010 # 纯横条最小厚度；细线误报就加。
BAR_ONLY_MAX_THICKNESS_RATIO = 0.075 # 纯横条最大厚度；大块误报就减。

# ===== 横条时间防抖与纯横条通道 =====
STOP_NEAR_RATIO = 0.8       # 横条接近画面底部比例；停得太早加，太晚减。
STOP_CENTER_WIDTH_RATIO = 0.12 # 底部停车区占画面中央宽度比例；减小可排除两侧干扰。
BAR_ONLY_STABLE_FRAMES = 1   # 无竖纹时横条连续确认帧数；误识别就加，出现太慢就减。
BAR_ONLY_MIN_LENGTH_RATIO = 0.2 # 无竖纹横条最小宽度比例；误识别就加，近景短条漏检就减。
BAR_ONLY_MAX_ABS_ANGLE = 20.0 # 无竖纹横条最大倾角；斜车道线误报就减，真实斜横条漏检就加。
BAR_TRACK_MAX_Y_RATIO = 0.18 # 前后帧横条最大纵向跳变；抖动串线就减，车速快就加。
BAR_TRACK_MAX_X_RATIO = 0.24 # 前后帧横条最大横向跳变；串线就减，急转跟丢就加。
BAR_TRACK_MAX_ANGLE = 10.0   # 前后帧横条最大角度跳变；串线就减，转动车身快就加。
BAR_TRACK_HOLD_FRAMES = 1    # 横条短暂丢失保持帧数；闪烁就加，残影久就减。
BAR_TRACK_SMOOTH = 0.25      # 横条位置历史保留比例；当前帧占 75%，框跟随迟钝就继续减。

# ===== 停车摆正 =====
ALIGN_TOLERANCE_DEG = 2.0    # 横条水平容差；难以完成摆正就加，要求更正就减。
ALIGN_KP = 0.018             # 摆正转向比例；摆正太慢就加，来回过冲就减。
ALIGN_MIN_ANGULAR = 0.08     # 摆正最小角速度；小误差转不动就加。
ALIGN_MAX_ANGULAR = 0.20     # 摆正最大角速度；转得太猛就减，太慢可加。
ALIGN_LOST_FALLBACK_ENABLED = True # 横条丢失时用最后角度完成补转。
ALIGN_ENTRY_MAX_ANGLE = 10.0 # 超时进入路口允许的最大横条角度；入场太斜就减。
ALIGN_ENTRY_MIN_STRIPES = 3  # 超时进入至少需要的竖条数；误进入就加。

# ===== 路口通过与双边透视补线 =====
MANEUVER_LOOKAHEAD_RATIO = 0.60 # 路口中心线前视控制行；减小看得更远，增大看得更近。
RANSAC_RESIDUAL_PIXELS = 12.0 # 直线内点容差像素；线断/抖就加，圆角混入就减。
RANSAC_MIN_INLIERS = 4       # 直线最少内点数；误拟合就加，难锁定就减。
MODEL_HOLD_FRAMES = 8        # 边线丢失后保持帧数；短暂丢线就加，旧线残留就减。
MODEL_MAX_SHIFT_RATIO = 0.08 # 新旧直线最大位置跳变；误换圆角就减，转向变化大就加。
MODEL_MAX_SLOPE_DELTA = 0.35 # 新旧直线最大斜率变化；误换线就减，允许急变就加。
MODEL_CENTER_CONSISTENCY_RATIO = 0.20 # 双边补出的中心最大差异；圆角抢线时优先连续模型。
MODEL_MAX_ABS_SLOPE = 0.85  # 新锁边线最大横向斜率；斑马条拟合成贯穿画面的斜线时减小。
MODEL_CENTER_CROSS_MARGIN_RATIO = 0.08 # 边线在 ROI 内越过画面中心的容差；误锁对侧杂线时减小。
WINDOW_NAME = "line_cy_task" # 调试窗口名称；不影响算法。
PROCESSED_WINDOW_NAME = "line_cy_task_processed" # 二值处理结果窗口名称。

# ===== YOLO 模型识别 =====
YOLO_FRAME_INTERVAL = 1      # YOLO 线程每隔多少个任务摄像头新帧做一次模型推理。
YOLO_PEOPLE_STABLE_FRAMES = 3 # 人群多数类别连续确认帧数；误识别多就加。
YOLO_CONFIDENCE = 0.60       # 人群 YOLO 置信度阈值。
YOLO_TRASH_CONFIDENCE = 0.65 # 垃圾桶 YOLO 置信度阈值。
YOLO_BUILDING_CONFIDENCE = 0.65 # 楼宇 YOLO 置信度阈值。
YOLO_CENTER_BAND_RATIO = 0.650 # 目标框中心位于画面中间此比例时才触发停车。
YOLO_STOP_TIME = 1.0         # 模型触发后停车等待时间(s)。
YOLO_EVENT_IGNORE_TIME = 4.0 # 每次任务识别后忽略新目标时间(s)。
YOLO_IMAGE_SIZE = 320        # 模型训练和导出使用的输入尺寸。
YOLO_NMS_THRESHOLD = 0.45    # ONNX 后处理 NMS 阈值；重叠框重复就减，漏框就加。
YOLO_SAVE_DIR = "/home/eaibot/zcy/保存图片" # 任务识别图片保存目录，启动时清空。
YOLO_STREET_CLASS_NAMES = (
    "General population",
    "Medical population",
    "hazardous waste",
    "recyclable material",
)
YOLO_BUILDING_CLASS_NAMES = (
    "Collapsed Building",
    "Electrical Fault Building",
    "Fire Building",
    "Toxic Gas-contaminated Building",
)
YOLO_CLASS_NAMES = (
    "Collapsed Building",
    "Electrical Fault Building",
    "Fire Building",
    "General population",
    "Medical population",
    "Toxic Gas-contaminated Building",
    "hazardous waste",
    "recyclable material",
)
YOLO_TARGET_CLASS_NAMES = (
    "Collapsed Building",
    "Electrical Fault Building",
    "Fire Building",
    "Toxic Gas-contaminated Building",
    "General population",
    "Medical population",
    "hazardous waste",
    "recyclable material",
)
YOLO_STREET_MESSAGES = {
    "Medical population": ("people", "医疗人群"),
    "General population": ("people", "普通人群"),
    "recyclable material": ("trash", "可回收垃圾"),
    "hazardous waste": ("trash", "有害垃圾"),
}
YOLO_PEOPLE_CLASS_NAMES = (
    "Medical population",
    "General population",
)
YOLO_BUILDING_MESSAGES = (
    ("Fire Building", "火灾楼宇"),
    ("Collapsed Building", "坍塌楼宇"),
    ("Toxic Gas-contaminated Building", "有毒气体楼宇"),
    ("Electrical Fault Building", "电力故障楼宇"),
)
YOLO_BUILDING_MESSAGE_BY_CLASS = dict(YOLO_BUILDING_MESSAGES)
YOLO_STREET_ROUTE_AREAS = {
    1: ("C区", "P区"),
    2: ("A区", "S区"),
}
YOLO_BUILDING_ROUTE_AREAS = {
    4: "楼宇B",
    5: "楼宇C",
    7: "楼宇A",
    8: "楼宇D",
}
YOLO_MODEL_PREFERRED_FILES = (
    "rub_roll_new_yolov5n_320_best.onnx",
    "rubbish_doll_yolov5n_320_best.onnx",
    "building_new_yolov5n_320_best.onnx",
    "merge_new_yolov5n_320_best.onnx",
    "best.onnx",
)
YOLO_WINDOW_NAME = "line_cy_task_yolo" # 任务识别调试窗口。

# ===== 红绿灯等待 =====
TRAFFIC_GREEN_STABLE_FRAMES = 2 # 连续绿灯确认帧数，防止单帧误放行。
TRAFFIC_LIGHT_RETRY_TIME = 2.0 # 摄像头或模型失败后的重试间隔(s)。
TRAFFIC_LIGHT_WINDOW_NAME = "line_cy_task_traffic_light"


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


def ensure_clean_directory(path):
    path = os.path.expanduser(str(path))
    if os.path.isdir(path):
        for name in os.listdir(path):
            item = os.path.join(path, name)
            if os.path.isdir(item):
                shutil.rmtree(item)
            else:
                os.remove(item)
    else:
        os.makedirs(path)


def safe_filename_text(text):
    return str(text).replace("/", "_").replace("\\", "_").replace(" ", "")


def draw_yolo_boxes(frame, detections, center_band_ratio, draw_center_band=True):
    output = frame.copy()
    height, width = output.shape[:2]
    if draw_center_band:
        ratio = clamp(float(center_band_ratio), 0.0, 1.0)
        left = int(round(width * (1.0 - ratio) * 0.5))
        right = int(round(width - left))
        cv2.line(output, (left, 0), (left, height - 1), (255, 255, 0), 1)
        cv2.line(output, (right, 0), (right, height - 1), (255, 255, 0), 1)
    for item in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in item.box]
        color = (0, 255, 0) if item.target and item.in_center else (0, 255, 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = "{} {:.2f}".format(item.class_name, item.confidence)
        cv2.putText(output, label, (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(output, (int(round(item.center_x)), int(round(item.center_y))),
                   3, color, -1)
    return output


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


def follow_entry_hits(candidate, current_hits):
    return current_hits + 1 if candidate else max(0, current_hits - 1)


def entry_acceptance_enabled(now, accept_after):
    return float(now) >= float(accept_after)


def yolo_route_context(task_index, state):
    if state in ("FINAL_EXIT", "DONE"):
        return {"kind": "off"}
    index = int(task_index)
    if index in YOLO_STREET_ROUTE_AREAS:
        return {"kind": "street", "areas": YOLO_STREET_ROUTE_AREAS[index]}
    if index in YOLO_BUILDING_ROUTE_AREAS:
        return {"kind": "building", "area": YOLO_BUILDING_ROUTE_AREAS[index]}
    return {"kind": "off"}


def yolo_model_profile(task_index):
    """返回当前路线应使用的任务识别模型。"""
    return "building" if int(task_index) >= YOLO_BUILDING_SWITCH_INDEX \
        else "street"


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


def resolve_yolo_model_path(model_path):
    path = os.path.expanduser(str(model_path))
    if os.path.isfile(path):
        if not path.lower().endswith(".onnx"):
            raise IOError("YOLO model must be an .onnx file: %s" % path)
        return path
    if not os.path.isdir(path):
        raise IOError("YOLO model path does not exist: %s" % path)
    onnx_files = [
        os.path.join(path, name)
        for name in os.listdir(path)
        if name.lower().endswith(".onnx")
    ]
    if not onnx_files:
        raise IOError("No .onnx YOLO model found in: %s" % path)
    by_name = {os.path.basename(item).lower(): item for item in onnx_files}
    for name in YOLO_MODEL_PREFERRED_FILES:
        preferred = by_name.get(name.lower())
        if preferred is not None:
            return preferred
    return sorted(onnx_files)[0]


class YoloDetection(object):
    def __init__(self, class_id, class_name, confidence, box,
                 frame_shape, center_band_ratio):
        self.class_id = int(class_id)
        self.class_name = str(class_name)
        self.confidence = float(confidence)
        self.box = tuple(float(value) for value in box)
        self.frame_shape = frame_shape
        self.center_band_ratio = float(center_band_ratio)
        height, width = frame_shape[:2]
        self.center_x = (self.box[0] + self.box[2]) * 0.5
        self.center_y = (self.box[1] + self.box[3]) * 0.5
        ratio = clamp(self.center_band_ratio, 0.0, 1.0)
        left = float(width) * (1.0 - ratio) * 0.5
        right = float(width) - left
        self.in_center = left <= self.center_x <= right
        self.target = self.class_name in YOLO_TARGET_CLASS_NAMES


class YoloTaskEvent(object):
    def __init__(self, kind, area, class_name, display_name, detection):
        self.kind = str(kind)
        self.area = str(area)
        self.class_name = str(class_name)
        self.display_name = str(display_name)
        self.detection = detection


class YoloTaskLedger(object):
    def __init__(self):
        self.street_results = dict(
            (area, None) for area in ("C区", "P区", "A区", "S区")
        )
        self.street_seen_classes = set()
        self.building_results = dict(
            (area, None) for area in ("楼宇A", "楼宇B", "楼宇C", "楼宇D")
        )
        self.building_seen_classes = set()
        self.people_stable_key = None
        self.people_stable_hits = 0
        self.pending_event = None
        self.save_index = 0

    def _target_candidates(self, detections, confidence):
        return [
            item for item in detections
            if item.target and item.in_center
            and item.confidence >= float(confidence)
        ]

    def _next_street_area(self, areas):
        for area in areas:
            if self.street_results.get(area) is None:
                return area
        return None

    def _reset_people_stability(self):
        self.people_stable_key = None
        self.people_stable_hits = 0

    def _stable_people_candidate(self, area, candidates, stable_frames):
        grouped = dict((name, []) for name in YOLO_PEOPLE_CLASS_NAMES)
        for item in candidates:
            if item.class_name in grouped:
                grouped[item.class_name].append(item)
        largest = max([len(items) for items in grouped.values()] or [0])
        winners = [
            name for name, items in grouped.items()
            if largest > 0 and len(items) == largest
        ]
        if len(winners) != 1:
            self._reset_people_stability()
            return None
        class_name = winners[0]
        key = (str(area), class_name)
        if key == self.people_stable_key:
            self.people_stable_hits += 1
        else:
            self.people_stable_key = key
            self.people_stable_hits = 1
        if self.people_stable_hits < max(1, int(stable_frames)):
            return None
        return max(grouped[class_name], key=lambda item: item.confidence)

    def select_event(self, context, detections, confidence,
                     building_confidence=None, people_stable_frames=1,
                     trash_confidence=None):
        kind = context.get("kind")
        if kind == "street":
            trash_threshold = confidence if trash_confidence is None \
                else float(trash_confidence)
            candidates = self._target_candidates(
                detections, min(float(confidence), trash_threshold)
            )
            area = self._next_street_area(context.get("areas", ()))
            if area is None:
                return None
            street = []
            for item in candidates:
                if (item.class_name not in YOLO_STREET_MESSAGES
                        or item.class_name in self.street_seen_classes):
                    continue
                target_kind, _ = YOLO_STREET_MESSAGES[item.class_name]
                threshold = trash_threshold if target_kind == "trash" \
                    else float(confidence)
                if item.confidence >= threshold:
                    street.append(item)
            if not street:
                self._reset_people_stability()
                return None
            people = [
                item for item in street
                if item.class_name in YOLO_PEOPLE_CLASS_NAMES
            ]
            if people:
                selected = self._stable_people_candidate(
                    area, people, people_stable_frames
                )
                if selected is None:
                    return None
            else:
                self._reset_people_stability()
                selected = max(street, key=lambda item: item.confidence)
            _, display_name = YOLO_STREET_MESSAGES[selected.class_name]
            return YoloTaskEvent(
                "street", area, selected.class_name, display_name, selected
            )
        if kind == "building":
            self._reset_people_stability()
            threshold = confidence if building_confidence is None \
                else building_confidence
            candidates = self._target_candidates(detections, threshold)
            area = context.get("area")
            if self.building_results.get(area) is not None:
                return None
            buildings = [
                item for item in candidates
                if item.class_name in YOLO_BUILDING_MESSAGE_BY_CLASS
                and item.class_name not in self.building_seen_classes
            ]
            if not buildings:
                return None
            selected = max(buildings, key=lambda item: item.confidence)
            return YoloTaskEvent(
                "building", area, selected.class_name,
                YOLO_BUILDING_MESSAGE_BY_CLASS[selected.class_name],
                selected,
            )
        return None

    def accept(self, event):
        self.pending_event = event
        if event is None:
            return
        if event.kind == "street":
            self.street_results[event.area] = event
            self.street_seen_classes.add(event.class_name)
            self._reset_people_stability()
        elif event.kind == "building":
            self.building_results[event.area] = event
            self.building_seen_classes.add(event.class_name)


class YoloObstacleDetector(object):
    def __init__(self, model_path, confidence=YOLO_CONFIDENCE,
                 center_band_ratio=YOLO_CENTER_BAND_RATIO,
                 image_size=YOLO_IMAGE_SIZE,
                 nms_threshold=YOLO_NMS_THRESHOLD,
                 class_names=YOLO_CLASS_NAMES):
        self.model_path = resolve_yolo_model_path(model_path)
        self.confidence = float(confidence)
        self.center_band_ratio = float(center_band_ratio)
        self.image_size = int(image_size)
        self.nms_threshold = float(nms_threshold)
        self.model = None
        self.names = self._normalize_names(class_names)
        self.backend_name = "opencv-dnn-onnx"

    def load(self):
        if self.model is not None:
            return
        try:
            self.model = cv2.dnn.readNetFromONNX(self.model_path)
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except Exception as exc:
            raise RuntimeError("Could not load ONNX YOLO model: %s" % exc)

    def close(self):
        """释放 OpenCV DNN 模型内存，供路线中途切换模型。"""
        self.model = None

    def _normalize_names(self, names):
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        if isinstance(names, (list, tuple)):
            return {index: str(value) for index, value in enumerate(names)}
        return {}

    def _letterbox(self, frame):
        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("YOLO frame has invalid shape")
        size = int(self.image_size)
        scale = min(float(size) / float(width), float(size) / float(height))
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(frame, (resized_width, resized_height),
                             interpolation=cv2.INTER_LINEAR)
        padded = np.full((size, size, 3), 114, dtype=np.uint8)
        pad_left = (size - resized_width) // 2
        pad_top = (size - resized_height) // 2
        padded[pad_top:pad_top + resized_height,
               pad_left:pad_left + resized_width] = resized
        return padded, scale, pad_left, pad_top

    def _predictions_from_output(self, output):
        if isinstance(output, (list, tuple)):
            if not output:
                return np.empty((0, 5 + len(self.names)), dtype=np.float32)
            output = output[0]
        data = np.asarray(output)
        if data.ndim == 3 and data.shape[0] == 1:
            data = data[0]
        if data.ndim != 2:
            return np.empty((0, 5 + len(self.names)), dtype=np.float32)
        field_counts = (4 + len(self.names), 5 + len(self.names))
        if data.shape[0] in field_counts:
            data = data.T
        elif data.shape[1] not in field_counts:
            return np.empty((0, 5 + len(self.names)), dtype=np.float32)
        return np.asarray(data, dtype=np.float32)

    def _nms_indices(self, boxes, confidences):
        kept = []
        class_ids = sorted(set(item[0] for item in boxes))
        for class_id in class_ids:
            local = [index for index, item in enumerate(boxes)
                     if item[0] == class_id]
            local_boxes = [boxes[index][1] for index in local]
            local_scores = [confidences[index] for index in local]
            indices = cv2.dnn.NMSBoxes(
                local_boxes, local_scores, self.confidence, self.nms_threshold
            )
            for item in np.asarray(indices).reshape(-1):
                kept.append(local[int(item)])
        return sorted(kept, key=lambda index: confidences[index], reverse=True)

    def _decode(self, output, frame_shape, scale, pad_left, pad_top):
        predictions = self._predictions_from_output(output)
        if len(predictions) == 0:
            return []
        coords = predictions[:, :4].copy()
        if np.isfinite(coords).any() and np.nanmax(np.abs(coords)) <= 2.0:
            coords *= float(self.image_size)
        if predictions.shape[1] == 5 + len(self.names):
            objectness = predictions[:, 4]
            scores = predictions[:, 5:]
        else:
            objectness = None
            scores = predictions[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(scores.shape[0]), class_ids]
        if objectness is not None:
            confidences = confidences * objectness
        height, width = frame_shape[:2]
        boxes = []
        selected_confidences = []
        for index, confidence in enumerate(confidences):
            confidence = float(confidence)
            if confidence < self.confidence:
                continue
            center_x, center_y, box_width, box_height = [
                float(value) for value in coords[index]
            ]
            x1 = (center_x - box_width * 0.5 - pad_left) / scale
            y1 = (center_y - box_height * 0.5 - pad_top) / scale
            x2 = (center_x + box_width * 0.5 - pad_left) / scale
            y2 = (center_y + box_height * 0.5 - pad_top) / scale
            x1 = clamp(x1, 0.0, float(width - 1))
            y1 = clamp(y1, 0.0, float(height - 1))
            x2 = clamp(x2, 0.0, float(width - 1))
            y2 = clamp(y2, 0.0, float(height - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append((
                int(class_ids[index]),
                [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            ))
            selected_confidences.append(confidence)

        detections = []
        for index in self._nms_indices(boxes, selected_confidences):
            class_id, box = boxes[index]
            x, y, box_width, box_height = box
            class_name = self.names.get(class_id, str(class_id))
            detections.append(YoloDetection(
                class_id, class_name, selected_confidences[index],
                (x, y, x + box_width, y + box_height),
                frame_shape, self.center_band_ratio,
            ))
        return detections

    def detect(self, frame):
        self.load()
        padded, scale, pad_left, pad_top = self._letterbox(frame)
        blob = cv2.dnn.blobFromImage(
            padded, 1.0 / 255.0, (self.image_size, self.image_size),
            swapRB=True, crop=False,
        )
        self.model.setInput(blob)
        output = self.model.forward()
        return self._decode(output, frame.shape, scale, pad_left, pad_top)


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
                 fill_width=0.0, left_fill_width=None,
                 right_fill_width=None,
                 center_near_weight=LANE_CENTER_NEAR_WEIGHT):
        self.roi_top = float(roi_top)
        self.roi_bottom = float(roi_bottom)
        self.scan_rows = int(scan_rows)
        self.fill_width = float(fill_width)
        self.left_fill_width = (self.fill_width if left_fill_width is None
                                else float(left_fill_width))
        self.right_fill_width = (self.fill_width if right_fill_width is None
                                 else float(right_fill_width))
        self.center_near_weight = max(1.0, float(center_near_weight))

    def _weighted_center_x(self, center_points, frame_height):
        top_y = float(frame_height) * self.roi_top
        bottom_y = float(frame_height) * self.roi_bottom
        span = max(1.0, bottom_y - top_y)
        weights = [
            1.0 + (self.center_near_weight - 1.0)
            * clamp((float(y) - top_y) / span, 0.0, 1.0)
            for _, y in center_points
        ]
        return float(np.average(
            [x for x, _ in center_points], weights=weights
        ))

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
        left_fill_width = (self.left_fill_width
                           if self.left_fill_width > 0 else expected)
        right_fill_width = (self.right_fill_width
                            if self.right_fill_width > 0 else expected)
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
                    offset = left_fill_width * 0.5
                    if follow_side == "left" and side_center_transform is not None:
                        offset = (side_center_transform[0] * y
                                  + side_center_transform[1])
                    candidates.append((left_x + offset, "left", offset))
                if right_x is not None and follow_side in (None, "right"):
                    offset = -right_fill_width * 0.5
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
            self._weighted_center_x(center_points, height), True, len(widths),
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
        self.selected_side = None

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
        self.selected_side = None

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
            self.selected_side = None
            return None, None, None

        if len(candidates) == 1:
            center = candidates[0][0]
            center_model = candidates[0][2]
            self.selected_side = candidates[0][1]
        else:
            self.selected_side = None
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
                    self.selected_side = chosen[1]
        self.last_center = float(center)
        if center_model is not None:
            self.center_model = center_model
        return self.last_center, self.left_model, self.right_model


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
                                         cv2.THRESH_BINARY_INV,
                                         ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C)
        binary = cv2.bitwise_and(color, adaptive)
        kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
        return cv2.morphologyEx(cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel),
                               cv2.MORPH_CLOSE, kernel)


class CameraReader(object):
    def __init__(self, index, frame_width=None, frame_height=None):
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
            if frame_width is not None and frame_height is not None:
                set_capture_resolution(self.cap, frame_width, frame_height)
            else:
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
        rospy.init_node("line_cy_task", anonymous=True)
        self.camera_index = int(rospy.get_param("~camera_index", CAMERA_INDEX))
        self.process_width = int(rospy.get_param("~process_width", PROCESS_WIDTH))
        self.dry_run = bool(rospy.get_param("~dry_run", DRY_RUN))
        self.debug_view = bool(rospy.get_param("~debug_view", DEBUG_VIEW))
        requested_traffic_light = bool(rospy.get_param(
            "~traffic_light_enabled", TRAFFIC_LIGHT_ENABLED
        ))
        self.traffic_light_enabled = requested_traffic_light
        if requested_traffic_light and not TRAFFIC_LIGHT_MODULE_AVAILABLE:
            message = "未找到 traffic_light_vision，禁止绕过红绿灯：%s" % \
                TRAFFIC_LIGHT_IMPORT_ERROR
            rospy.logerr("line_cy_task %s", message)
            rospy.signal_shutdown(message)
        self.traffic_light_camera_index = int(rospy.get_param(
            "~traffic_light_camera_index", TRAFFIC_LIGHT_CAMERA_INDEX
        ))
        self.traffic_light_model_path = str(rospy.get_param(
            "~traffic_light_model_path", TRAFFIC_LIGHT_MODEL_PATH
        ))
        self.traffic_light_confidence = clamp(float(rospy.get_param(
            "~traffic_light_confidence", TRAFFIC_LIGHT_CONFIDENCE
        )), 0.01, 1.0)
        self.traffic_green_stable_frames = max(1, int(rospy.get_param(
            "~traffic_green_stable_frames", TRAFFIC_GREEN_STABLE_FRAMES
        )))
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
        self.final_exit_time = max(0.0, float(rospy.get_param(
            "~final_exit_time", FINAL_EXIT_TIME
        )))
        self.yolo_enabled = bool(rospy.get_param("~yolo_enabled", YOLO_ENABLED))
        self.yolo_stop_enabled = bool(rospy.get_param(
            "~yolo_stop_enabled", YOLO_STOP_ENABLED
        ))
        self.yolo_debug_view = bool(rospy.get_param(
            "~yolo_debug_view", self.debug_view
        ))
        self.yolo_camera_index = int(rospy.get_param(
            "~yolo_camera_index", YOLO_CAMERA_INDEX
        ))
        legacy_yolo_model_path = str(rospy.get_param(
            "~yolo_model_path", YOLO_STREET_MODEL_PATH
        ))
        self.yolo_street_model_path = str(rospy.get_param(
            "~yolo_street_model_path", legacy_yolo_model_path
        ))
        self.yolo_building_model_path = str(rospy.get_param(
            "~yolo_building_model_path", YOLO_BUILDING_MODEL_PATH
        ))
        self.yolo_model_path = self.yolo_street_model_path
        self.yolo_frame_interval = max(1, int(rospy.get_param(
            "~yolo_frame_interval", YOLO_FRAME_INTERVAL
        )))
        self.yolo_people_stable_frames = max(1, int(rospy.get_param(
            "~yolo_people_stable_frames", YOLO_PEOPLE_STABLE_FRAMES
        )))
        self.yolo_confidence = clamp(float(rospy.get_param(
            "~yolo_confidence", YOLO_CONFIDENCE
        )), 0.001, 1.0)
        self.yolo_trash_confidence = YOLO_TRASH_CONFIDENCE
        self.yolo_building_confidence = YOLO_BUILDING_CONFIDENCE
        self.yolo_center_band_ratio = clamp(float(rospy.get_param(
            "~yolo_center_band_ratio", YOLO_CENTER_BAND_RATIO
        )), 0.01, 1.0)
        self.yolo_image_size = max(32, int(rospy.get_param(
            "~yolo_image_size", YOLO_IMAGE_SIZE
        )))
        self.yolo_nms_threshold = clamp(float(rospy.get_param(
            "~yolo_nms_threshold", YOLO_NMS_THRESHOLD
        )), 0.0, 1.0)
        legacy_class_names = rospy.get_param("~yolo_class_names", None)
        street_class_names = rospy.get_param(
            "~yolo_street_class_names", legacy_class_names
        )
        building_class_names = rospy.get_param(
            "~yolo_building_class_names", None
        )
        self.yolo_street_class_names = self._normalize_ros_class_names(
            street_class_names, YOLO_STREET_CLASS_NAMES
        )
        self.yolo_building_class_names = self._normalize_ros_class_names(
            building_class_names, YOLO_BUILDING_CLASS_NAMES
        )
        self.yolo_class_names = self.yolo_street_class_names
        self.yolo_save_dir = str(rospy.get_param(
            "~yolo_save_dir", YOLO_SAVE_DIR
        ))
        self.yolo_stop_time = max(0.0, float(rospy.get_param(
            "~yolo_stop_time", YOLO_STOP_TIME
        )))
        self.yolo_event_ignore_time = max(0.0, float(rospy.get_param(
            "~yolo_event_ignore_time", YOLO_EVENT_IGNORE_TIME
        )))
        self.task_index = 0
        self.turn_cmd = TASK_TURN_COMMANDS[self.task_index]
        rospy.loginfo(
            "line_cy_task route=%s entry=%.2f speed=%.2f angular=%.2f "
            "turn_time=%.2f final_exit=%.2f",
            ",".join(TASK_TURN_COMMANDS),
            self.turn_entry_time, self.turn_speed,
            self.turn_angular, self.turn_time, self.final_exit_time,
        )
        rospy.loginfo(
            "line_cy_task intersection %d/%d command=%s",
            self.task_index + 1, len(TASK_TURN_COMMANDS), self.turn_cmd,
        )
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.vision = BinaryVision()
        self.lanes = LaneDetector(
            fill_width=FILL_WIDTH_PIXELS,
            left_fill_width=LEFT_FILL_WIDTH_PIXELS,
            right_fill_width=RIGHT_FILL_WIDTH_PIXELS,
        )
        self.crosswalk = CrosswalkDetector()
        self.camera = CameraReader(self.camera_index)
        self.yolo_camera = None
        self.traffic_camera = None
        self.traffic_camera_owned = False
        self.traffic_camera_configured = False
        self.traffic_detector = None
        self.traffic_green_hits = 0
        self.traffic_last_color = None
        self.traffic_retry_after = 0.0
        self.yolo_detector = None
        self.yolo_counter = 0
        self.yolo_lock = threading.Lock()
        self.yolo_switch_lock = threading.Lock()
        self.yolo_thread = None
        self.yolo_running = False
        self.yolo_worker_active = False
        self.yolo_latest_seq = 0
        self.yolo_read_seq = 0
        self.yolo_latest_detections = []
        self.yolo_latest_frame = None
        self.yolo_ready = False
        self.yolo_active_profile = None
        self.yolo_stop_detection = None
        self.yolo_stop_reported = False
        self.yolo_stop_report_seq = 0
        self.yolo_segment_key = None
        self.yolo_segment_start_seq = 0
        self.yolo_accept_after = 0.0
        self.task_ledger = YoloTaskLedger()
        self.pid = PID(KP, KD, MAX_ANGULAR)
        self.lane_width = LANE_WIDTH_PIXELS if LANE_WIDTH_PIXELS > 0 else PROCESS_WIDTH * DEFAULT_LANE_WIDTH_RATIO
        self.bridge = DualLineBridge(self.lane_width, fill_width=FILL_WIDTH_PIXELS)
        self.state = "FOLLOW"
        self.state_started = rospy.get_time()
        self.stop_hits = self.lost_hits = self.align_hits = 0
        self.wait_recover_hits = 0
        self.clear_hits = self.exit_hits = 0
        self.entry_cleared = False
        self.maneuver_phase = "NONE"
        self.maneuver_phase_started = self.state_started
        self.entry_accept_after = 0.0
        self.align_lock = None
        self.align_last_angle = None
        self.last_angular = 0.0
        self.last_command_angular = 0.0
        self.last_control_target = None
        self.last_observation = None
        self.last_crosswalk = CrosswalkResult()
        self.last_binary = None
        self.cleaned = False
        if not self.camera.cap.isOpened():
            rospy.signal_shutdown("cannot open lane camera")
        self._prepare_yolo_save_dir()
        if self.yolo_enabled:
            self._init_yolo()
        rospy.on_shutdown(self.cleanup)

    def _normalize_ros_class_names(self, value, defaults):
        if not value:
            return tuple(defaults)
        if isinstance(value, str):
            value = [item.strip() for item in value.split(",")
                     if item.strip()]
        return tuple(value)

    def _yolo_profile_settings(self, profile):
        if profile == "building":
            return (
                self.yolo_building_model_path,
                self.yolo_building_class_names,
                self.yolo_building_confidence,
            )
        return (
            self.yolo_street_model_path,
            self.yolo_street_class_names,
            min(self.yolo_confidence, self.yolo_trash_confidence),
        )

    def _create_yolo_detector(self, profile):
        model_path, class_names, confidence = \
            self._yolo_profile_settings(profile)
        return YoloObstacleDetector(
            model_path,
            confidence=confidence,
            center_band_ratio=self.yolo_center_band_ratio,
            image_size=self.yolo_image_size,
            nms_threshold=self.yolo_nms_threshold,
            class_names=class_names,
        )

    def _init_yolo(self):
        initial_profile = yolo_model_profile(self.task_index)
        _, initial_class_names, _ = self._yolo_profile_settings(initial_profile)
        try:
            self.yolo_detector = self._create_yolo_detector(initial_profile)
            rospy.loginfo(
                "line_cy_task 正在加载并预热%s模型：%s",
                initial_profile,
                self.yolo_detector.model_path,
            )
            self.publish(0, 0)
            # 启动时完成加载，避免进入识别路段后再承担首次加载延迟。
            self.yolo_detector.load()
            self.yolo_active_profile = initial_profile
            self.yolo_model_path = self.yolo_detector.model_path
            self.yolo_class_names = tuple(initial_class_names)
        except Exception as exc:
            rospy.logwarn("line_cy_task YOLO disabled: %s", exc)
            self.yolo_enabled = False
            self.yolo_detector = None
            self.yolo_active_profile = None
            self.yolo_ready = False
            return
        traffic_camera_index = getattr(
            self, "traffic_light_camera_index", TRAFFIC_LIGHT_CAMERA_INDEX
        )
        if self.yolo_camera_index == traffic_camera_index:
            try:
                configure_traffic_camera(self.yolo_camera_index)
                self.traffic_camera_configured = True
            except Exception as exc:
                self.traffic_camera_configured = False
                rospy.logwarn(
                    "line_cy_task 摄像头 %d 参数设置失败：%s",
                    self.yolo_camera_index, exc,
                )
        self.yolo_camera = CameraReader(
            self.yolo_camera_index,
            TRAFFIC_LIGHT_FRAME_WIDTH,
            TRAFFIC_LIGHT_FRAME_HEIGHT,
        )
        if not self.yolo_camera.cap.isOpened():
            rospy.logwarn(
                "line_cy_task YOLO camera %d cannot open, YOLO disabled",
                self.yolo_camera_index,
            )
            self.yolo_enabled = False
            self.yolo_ready = False
            self.yolo_camera.release()
            self.yolo_camera = None
            return
        rospy.loginfo("line_cy_task waiting for YOLO first frame warmup")
        ok, frame = self.yolo_camera.read(3.0)
        if not ok:
            rospy.logwarn(
                "line_cy_task YOLO camera %d has no warmup frame, YOLO disabled",
                self.yolo_camera_index,
            )
            self.yolo_enabled = False
            self.yolo_ready = False
            self.yolo_camera.release()
            self.yolo_camera = None
            return
        try:
            detections = self.yolo_detector.detect(frame)
        except Exception as exc:
            rospy.logwarn("line_cy_task YOLO warmup inference failed: %s", exc)
            self.yolo_enabled = False
            self.yolo_ready = False
            self.yolo_camera.release()
            self.yolo_camera = None
            return
        self._store_yolo_result(frame, detections)
        self.yolo_ready = True
        rospy.loginfo(
            "line_cy_task YOLO %s模型加载和预热完成，"
            "enabled camera=%d backend=%s model=%s interval=%d imgsz=%d "
            "people_stable=%d people_conf=%.2f trash_conf=%.2f "
            "building_conf=%.2f nms=%.2f "
            "stop=%s debug=%s",
            self.yolo_active_profile,
            self.yolo_camera_index, self.yolo_detector.backend_name,
            self.yolo_detector.model_path, self.yolo_frame_interval,
            self.yolo_image_size, self.yolo_people_stable_frames,
            self.yolo_confidence,
            self.yolo_trash_confidence, self.yolo_building_confidence,
            self.yolo_nms_threshold,
            self.yolo_stop_enabled,
            self.yolo_debug_view,
        )
        self.yolo_running = True
        self.yolo_thread = threading.Thread(target=self._yolo_loop)
        self.yolo_thread.daemon = True
        self.yolo_thread.start()

    def _clear_yolo_cache(self):
        with self.yolo_lock:
            self.yolo_latest_detections = []
            self.yolo_latest_frame = None
            self.yolo_read_seq = self.yolo_latest_seq
        self.yolo_segment_key = None
        self.yolo_segment_start_seq = self._latest_yolo_seq()
        self.yolo_counter = 0

    def _switch_yolo_profile_if_needed(self):
        """在第三个右转完成后释放街道模型并加载楼宇模型。"""
        if not getattr(self, "yolo_enabled", False):
            return True
        desired_profile = yolo_model_profile(self.task_index)
        if desired_profile == getattr(self, "yolo_active_profile", None):
            return True
        if self.yolo_camera is None:
            rospy.logwarn("line_cy_task YOLO 摄像头不可用，无法切换模型")
            self.yolo_enabled = False
            self.yolo_ready = False
            return False

        model_path, class_names, _ = self._yolo_profile_settings(
            desired_profile
        )
        rospy.loginfo(
            "line_cy_task 到达物资点切换位置，停车并切换为%s模型：%s",
            desired_profile, model_path,
        )
        self.publish(0, 0)
        self.yolo_ready = False
        detector = None
        try:
            with self.yolo_switch_lock:
                old_detector = self.yolo_detector
                self.yolo_detector = None
                if old_detector is not None:
                    old_detector.close()
                detector = self._create_yolo_detector(desired_profile)
                detector.load()
                ok, frame = self.yolo_camera.read(3.0)
                if not ok:
                    raise RuntimeError("模型切换后摄像头没有新画面")
                detections = detector.detect(frame)
                self.yolo_detector = detector
        except Exception as exc:
            if detector is not None:
                detector.close()
            self.yolo_detector = None
            self.yolo_active_profile = None
            self.yolo_enabled = False
            self.yolo_ready = False
            rospy.logwarn("line_cy_task YOLO 模型切换失败，已关闭任务识别：%s",
                          exc)
            return False

        self.yolo_active_profile = desired_profile
        self.yolo_model_path = self.yolo_detector.model_path
        self.yolo_class_names = tuple(class_names)
        self._clear_yolo_cache()
        self._store_yolo_result(frame, detections)
        self.yolo_ready = True
        rospy.loginfo(
            "line_cy_task YOLO 已切换为%s模型并完成预热：%s",
            desired_profile, self.yolo_detector.model_path,
        )
        return True

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
        self.last_command_angular = angular
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
        direction_scale = FOLLOW_LEFT_ANGULAR_SCALE \
            if raw > 0.0 else FOLLOW_RIGHT_ANGULAR_SCALE
        raw *= direction_scale
        angular = ANGULAR_SMOOTH * self.last_angular + (1.0 - ANGULAR_SMOOTH) * raw
        self.last_angular = angular
        self.publish(speed, angular)

    def _set_maneuver_phase(self, phase, now=None):
        if phase == self.maneuver_phase:
            return
        rospy.loginfo("line_cy_task maneuver phase: %s -> %s",
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

    def _entry_ready_state(self):
        enabled = getattr(self, "traffic_light_enabled", TRAFFIC_LIGHT_ENABLED)
        return "TRAFFIC_WAIT" if enabled else "MANEUVER"

    def _close_traffic_light(self):
        detector = getattr(self, "traffic_detector", None)
        if detector is not None:
            detector.close()
        self.traffic_detector = None
        camera = getattr(self, "traffic_camera", None)
        if camera is not None and getattr(self, "traffic_camera_owned", False):
            camera.release()
        self.traffic_camera = None
        self.traffic_camera_owned = False
        self.traffic_green_hits = 0
        self.traffic_last_color = None
        try:
            cv2.destroyWindow(TRAFFIC_LIGHT_WINDOW_NAME)
        except cv2.error:
            pass

    def _open_traffic_light(self):
        shared_camera = getattr(self, "yolo_camera", None)
        use_shared = (
            shared_camera is not None
            and shared_camera.cap.isOpened()
            and self.yolo_camera_index == self.traffic_light_camera_index
        )
        if use_shared:
            if not getattr(self, "traffic_camera_configured", False):
                configure_traffic_camera(self.traffic_light_camera_index)
                self.traffic_camera_configured = True
            camera = shared_camera
            camera_owned = False
        else:
            camera = CameraReader(
                self.traffic_light_camera_index,
                TRAFFIC_LIGHT_FRAME_WIDTH,
                TRAFFIC_LIGHT_FRAME_HEIGHT,
            )
            camera_owned = True
            if not camera.cap.isOpened():
                camera.release()
                raise RuntimeError("无法打开红绿灯摄像头 %d" %
                                   self.traffic_light_camera_index)
            try:
                configure_traffic_camera(self.traffic_light_camera_index)
                self.traffic_camera_configured = True
            except Exception:
                camera.release()
                raise
        detector = TrafficLightDetector(
            self.traffic_light_model_path,
            confidence=self.traffic_light_confidence,
        )
        try:
            detector.load()
        except Exception:
            if camera_owned:
                camera.release()
            raise
        self.traffic_camera = camera
        self.traffic_camera_owned = camera_owned
        self.traffic_detector = detector
        rospy.loginfo(
            "line_cy_task 红绿灯模型已在停止线加载：camera=%d model=%s",
            self.traffic_light_camera_index,
            self.traffic_light_model_path,
        )

    def _handle_traffic_light_wait(self, now):
        self.publish(0, 0)
        # 原任务模型可能刚开始一帧推理，等它退出后再独占摄像头和 CPU。
        if getattr(self, "yolo_worker_active", False):
            return
        if self.traffic_detector is None or self.traffic_camera is None:
            if float(now) < self.traffic_retry_after:
                return
            try:
                self._open_traffic_light()
            except Exception as exc:
                self._close_traffic_light()
                self.traffic_retry_after = float(now) + TRAFFIC_LIGHT_RETRY_TIME
                rospy.logwarn("line_cy_task 红绿灯识别启动失败，保持停车：%s", exc)
                return
        ok, frame = self.traffic_camera.read(0.2)
        if not ok:
            return
        try:
            detections = self.traffic_detector.detect(frame)
        except Exception as exc:
            self._close_traffic_light()
            self.traffic_retry_after = float(now) + TRAFFIC_LIGHT_RETRY_TIME
            rospy.logwarn("line_cy_task 红绿灯推理失败，保持停车：%s", exc)
            return
        self.traffic_green_hits, green_ready, color = update_green_hits(
            detections, self.traffic_green_hits,
            self.traffic_green_stable_frames,
        )
        self.traffic_last_color = color
        if self.debug_view:
            try:
                cv2.imshow(
                    TRAFFIC_LIGHT_WINDOW_NAME,
                    draw_traffic_light(
                        frame, detections, color, self.traffic_green_hits,
                        self.traffic_green_stable_frames,
                    ),
                )
                cv2.waitKey(1)
            except cv2.error:
                pass
        if green_ready:
            rospy.loginfo("line_cy_task 连续识别到绿灯，释放模型并进入路口")
            self._set_state("MANEUVER")

    def _lock_entry_alignment(self, now=None, angle=None):
        if angle is None:
            cross = getattr(self, "last_crosswalk", None)
            if cross is None:
                self.align_lock = None
                return False
            if getattr(cross, "candidate", False) and cross.stop_angle is not None:
                angle = cross.stop_angle
            else:
                angle = cross.tracking_angle
        if angle is None:
            self.align_lock = None
            return False
        now = rospy.get_time() if now is None else float(now)
        angle = float(angle)
        magnitude = clamp(abs(angle) * ALIGN_KP,
                          ALIGN_MIN_ANGULAR, ALIGN_MAX_ANGULAR)
        angular = 0.0 if abs(angle) <= ALIGN_TOLERANCE_DEG \
            else (-magnitude if angle > 0.0 else magnitude)
        rotate_time = 0.0 if angular == 0.0 else clamp(
            np.radians(abs(angle)) / max(magnitude, 1e-6)
            * ALIGN_OPEN_LOOP_TIME_SCALE,
            ALIGN_OPEN_LOOP_MIN_TIME,
            ALIGN_OPEN_LOOP_MAX_TIME,
        )
        self.align_lock = {
            "angle": angle,
            "angular": angular,
            "rotate_until": now + rotate_time,
            "settle_until": now + rotate_time + ALIGN_LOCK_SETTLE_TIME,
        }
        rospy.loginfo(
            "line_cy_task lost-bar align angle=%.1f angular=%.2f "
            "rotate=%.2fs settle=%.2fs",
            self.align_lock["angle"], self.align_lock["angular"],
            rotate_time, ALIGN_LOCK_SETTLE_TIME,
        )
        return True

    def _run_locked_entry_alignment(self, now):
        if self.align_lock is None:
            return False, None
        now = float(now)
        if now < self.align_lock["rotate_until"]:
            self.publish(0, self.align_lock["angular"])
            return True, None
        if now < self.align_lock["settle_until"]:
            self.publish(0, 0)
            return True, None
        self.publish(0, 0)
        return True, "MANEUVER"

    def _set_state(self, state):
        if state == self.state:
            return
        previous_state = self.state
        if previous_state == "TRAFFIC_WAIT" and state != "TRAFFIC_WAIT":
            self._close_traffic_light()
        rospy.loginfo("line_cy_task state: %s -> %s", previous_state, state)
        self.state = state
        self.state_started = rospy.get_time()
        self.pid.reset()
        self.last_angular = 0.0
        self.last_control_target = None
        self.lost_hits = self.align_hits = 0
        self.align_lock = None
        self.align_last_angle = None
        if (state == "FOLLOW"
                and previous_state in ("EXIT_ALIGN", "MANEUVER")):
            self.stop_hits = 0
            self.entry_accept_after = (
                self.state_started + EXIT_ENTRY_IGNORE_TIME
            )
        if state == "FOLLOW" and previous_state == "YOLO_STOP":
            ignore_time = max(0.0, float(getattr(
                self, "yolo_event_ignore_time", YOLO_EVENT_IGNORE_TIME
            )))
            self.yolo_accept_after = self.state_started + ignore_time
            rospy.loginfo(
                "line_cy_task 任务识别保护 %.1f 秒",
                ignore_time,
            )
        if state in ("FOLLOW", "MANEUVER", "FINAL_EXIT", "YOLO_STOP",
                     "TRAFFIC_WAIT"):
            self.crosswalk.unlock_bar()
        if state == "TRAFFIC_WAIT":
            self.traffic_retry_after = self.state_started
            self.traffic_green_hits = 0
            self.traffic_last_color = None
        if state == "MANEUVER":
            self.entry_cleared = False
            self.clear_hits = self.exit_hits = 0
            self.maneuver_phase = (
                "ENTRY" if maneuver_follow_side(self.turn_cmd) is not None
                else "STRAIGHT"
            )
            self.maneuver_phase_started = self.state_started
        else:
            self.maneuver_phase = "NONE"
    def _complete_intersection(self):
        completed = self.task_index + 1
        rospy.loginfo(
            "line_cy_task intersection %d/%d completed command=%s",
            completed, len(TASK_TURN_COMMANDS), self.turn_cmd,
        )
        if completed >= len(TASK_TURN_COMMANDS):
            self._set_state("FINAL_EXIT")
            return

        self.task_index += 1
        self.turn_cmd = TASK_TURN_COMMANDS[self.task_index]
        self._switch_yolo_profile_if_needed()
        self._set_state("FOLLOW")
        rospy.loginfo(
            "line_cy_task intersection %d/%d command=%s",
            self.task_index + 1, len(TASK_TURN_COMMANDS), self.turn_cmd,
        )

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

    def _prepare_yolo_save_dir(self):
        ensure_clean_directory(self.yolo_save_dir)

    def _save_yolo_event_image(self, event, detections):
        with self.yolo_lock:
            frame = None if self.yolo_latest_frame is None \
                else self.yolo_latest_frame.copy()
        if frame is None:
            frame = np.zeros((1, 1, 3), dtype=np.uint8)
        event_confidence = self.yolo_confidence
        if event.kind == "building":
            event_confidence = self.yolo_building_confidence
        elif (event.kind == "street"
              and YOLO_STREET_MESSAGES[event.class_name][0] == "trash"):
            event_confidence = self.yolo_trash_confidence
        event_detections = [
            item for item in detections
            if item.class_name == event.class_name
            and item.confidence >= event_confidence
        ]
        if not event_detections and event.detection is not None:
            event_detections = [event.detection]
        boxed = draw_yolo_boxes(
            frame, event_detections, self.yolo_center_band_ratio,
            draw_center_band=False,
        )
        self.task_ledger.save_index += 1
        result = event.display_name
        filename = "%02d_%s_%s.jpg" % (
            self.task_ledger.save_index,
            safe_filename_text(event.area),
            safe_filename_text(result),
        )
        path = os.path.join(self.yolo_save_dir, filename)
        cv2.imwrite(path, boxed)
        return path

    def _report_yolo_task_event(self, detections):
        event = self.task_ledger.pending_event
        if event is None:
            return
        if event.kind == "street":
            target_kind, _ = YOLO_STREET_MESSAGES[event.class_name]
            if target_kind == "people":
                rospy.loginfo("%s识别到%s", event.area, event.display_name)
            else:
                rospy.loginfo(
                    "%s检测到垃圾桶：%s",
                    event.area, event.display_name,
                )
        elif event.kind == "building":
            rospy.loginfo("%s检测到%s", event.area, event.display_name)
        self._save_yolo_event_image(event, detections)
        self.task_ledger.pending_event = None

    def _store_yolo_result(self, frame, detections):
        display_frame = None if frame is None else frame.copy()
        with self.yolo_lock:
            self.yolo_latest_seq += 1
            self.yolo_latest_detections = list(detections)
            self.yolo_latest_frame = display_frame

    def _yolo_loop(self):
        while self.yolo_running and not rospy.is_shutdown():
            if not self.yolo_enabled or self.yolo_detector is None \
                    or self.yolo_camera is None:
                time.sleep(0.05)
                continue
            if not self._yolo_inference_allowed():
                time.sleep(0.05)
                continue
            ok, frame = self.yolo_camera.read(0.2)
            if not ok:
                continue
            self.yolo_counter += 1
            if self.yolo_counter % self.yolo_frame_interval != 0:
                continue
            try:
                with self.yolo_switch_lock:
                    detector = self.yolo_detector
                    if detector is None:
                        continue
                    self.yolo_worker_active = True
                    detections = detector.detect(frame)
            except Exception as exc:
                rospy.logwarn("line_cy_task YOLO inference failed: %s", exc)
                detections = []
            finally:
                self.yolo_worker_active = False
            self._store_yolo_result(frame, detections)

    def _poll_yolo_detections(self):
        if not self.yolo_enabled or self.yolo_detector is None \
                or self.yolo_camera is None:
            return False, []
        with self.yolo_lock:
            if self.yolo_latest_seq == self.yolo_read_seq:
                return False, []
            self.yolo_read_seq = self.yolo_latest_seq
            return True, list(self.yolo_latest_detections)

    def _latest_yolo_seq(self):
        with self.yolo_lock:
            return self.yolo_latest_seq

    def _current_yolo_context(self):
        return yolo_route_context(
            getattr(self, "task_index", 0),
            getattr(self, "state", "FOLLOW"),
        )

    def _yolo_context_key(self, context):
        if context.get("kind") == "street":
            return ("street", tuple(context.get("areas", ())))
        if context.get("kind") == "building":
            return ("building", context.get("area"))
        return ("off", None)

    def _mark_yolo_segment_if_needed(self):
        context = self._current_yolo_context()
        key = self._yolo_context_key(context)
        if key != self.yolo_segment_key:
            self.yolo_segment_key = key
            self.yolo_segment_start_seq = self._latest_yolo_seq()
        return context

    def _yolo_inference_allowed(self):
        if not self.yolo_enabled:
            return False
        if getattr(self, "state", None) not in ("FOLLOW", "YOLO_STOP"):
            return False
        return (
            getattr(self, "state", None) == "YOLO_STOP"
            or self._current_yolo_context().get("kind") != "off"
        )

    def _yolo_segment_has_fresh_result(self):
        if self._current_yolo_context().get("kind") == "off":
            return True
        return self._latest_yolo_seq() > self.yolo_segment_start_seq

    def _wait_for_yolo_ready_if_needed(self):
        context = self._mark_yolo_segment_if_needed()
        if context.get("kind") == "off" or not self.yolo_enabled:
            return True
        if self.yolo_ready and self._yolo_segment_has_fresh_result():
            return True
        self.publish(0, 0)
        return False

    def _select_yolo_stop_event(self, detections, now=None):
        return self.task_ledger.select_event(
            self._current_yolo_context(), detections, self.yolo_confidence,
            getattr(self, "yolo_building_confidence", self.yolo_confidence),
            getattr(self, "yolo_people_stable_frames",
                    YOLO_PEOPLE_STABLE_FRAMES),
            getattr(self, "yolo_trash_confidence", YOLO_TRASH_CONFIDENCE),
        )

    def _maybe_enter_yolo_stop(self, observation):
        if self.state != "FOLLOW" or not self.yolo_enabled:
            return False
        if rospy.get_time() < getattr(self, "yolo_accept_after", 0.0):
            return False
        if self._current_yolo_context().get("kind") == "off":
            return False
        if not self._wait_for_yolo_ready_if_needed():
            return False
        sampled, detections = self._poll_yolo_detections()
        if not sampled:
            return False
        event = self._select_yolo_stop_event(detections)
        if event is None or not self.yolo_stop_enabled:
            return False
        self.task_ledger.accept(event)
        self.yolo_stop_detection = event.detection
        self.yolo_stop_reported = False
        self.yolo_stop_report_seq = self._latest_yolo_seq()
        self._set_state("YOLO_STOP")
        self.publish(0, 0)
        return True

    def _handle_yolo_stop(self, now):
        if self.state != "YOLO_STOP":
            return False
        self.publish(0, 0)
        if not self.yolo_stop_reported:
            sampled, detections = self._poll_yolo_detections()
            if sampled and self.yolo_read_seq > self.yolo_stop_report_seq:
                self._report_yolo_task_event(detections)
                self.yolo_stop_reported = True
        if (self.yolo_stop_reported
                and float(now) - self.state_started >= self.yolo_stop_time):
            self._set_state("FOLLOW")
        return True

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
            entry_candidate = (
                entry_allowed
                and cross.candidate
                and len(cross.stripe_polygons) >= ENTRY_MIN_STRIPES
            )
            self.stop_hits = follow_entry_hits(
                entry_candidate, self.stop_hits
            )
            if self.stop_hits >= STOP_STABLE_FRAMES:
                self.stop_hits = 0
                self.crosswalk.lock_current_bar()
                self.bridge.reset(self.lane_width)
                self._set_state("APPROACH")
            if (self.state == "FOLLOW"
                    and not self._wait_for_yolo_ready_if_needed()):
                pass
            elif self.state == "FOLLOW" and self._maybe_enter_yolo_stop(observation):
                pass
            elif self.state != "FOLLOW":
                self.publish(0, 0)
            elif observation.valid:
                self._control(observation.center_x, frame.shape[1], FOLLOW_SPEED,
                              FOLLOW_CENTER_BIAS_PIXELS)
            else:
                self.publish(0, 0)

        elif self.state == "YOLO_STOP":
            self._handle_yolo_stop(now)

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
            if self.align_lock is not None:
                _, next_state = self._run_locked_entry_alignment(now)
                if next_state is not None:
                    self._set_state(self._entry_ready_state())
            else:
                angle = cross.stop_angle if cross.candidate else cross.tracking_angle
                if angle is None:
                    self.lost_hits += 1
                    if (ALIGN_LOST_FALLBACK_ENABLED
                            and self.align_last_angle is not None
                            and self._lock_entry_alignment(
                                now, self.align_last_angle
                            )):
                        _, next_state = self._run_locked_entry_alignment(now)
                        if next_state is not None:
                            self._set_state(self._entry_ready_state())
                    else:
                        self.publish(0, 0)
                elif abs(angle) <= ALIGN_TOLERANCE_DEG:
                    self.align_last_angle = float(angle)
                    self.lost_hits = 0
                    self.align_hits += 1
                    self.publish(0, 0)
                else:
                    self.align_last_angle = float(angle)
                    self.lost_hits = 0
                    self.align_hits = 0
                    magnitude = clamp(abs(angle) * ALIGN_KP,
                                      ALIGN_MIN_ANGULAR, ALIGN_MAX_ANGULAR)
                    self.publish(0, -magnitude if angle > 0 else magnitude)
                if self.align_hits >= ALIGN_STABLE_FRAMES:
                    self._set_state(self._entry_ready_state())

        elif self.state == "TRAFFIC_WAIT":
            self._handle_traffic_light_wait(now)

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
                self._complete_intersection()
                if self.state == "FOLLOW" and observation.valid:
                    self._control(
                        observation.center_x, frame.shape[1], FOLLOW_SPEED,
                        FOLLOW_CENTER_BIAS_PIXELS,
                    )
                elif self.state == "FINAL_EXIT":
                    self.publish(FOLLOW_SPEED, 0.0)
                else:
                    self.publish(0, 0)

        elif self.state == "FINAL_EXIT":
            if now - self.state_started >= self.final_exit_time:
                self.publish(0, 0)
                rospy.loginfo(
                    "line_cy_task route completed, final exit %.2f seconds",
                    self.final_exit_time,
                )
                self._set_state("DONE")
                rospy.signal_shutdown("line_cy_task completed")
            else:
                self.publish(self.turn_speed, 0.0)

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
                self._set_state(self._entry_ready_state())

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
                bias = 0.0
                if self.bridge.selected_side == "left":
                    bias = abs(MANEUVER_CENTER_BIAS_PIXELS)
                elif self.bridge.selected_side == "right":
                    bias = -abs(MANEUVER_CENTER_BIAS_PIXELS)
                self._control(
                    center, frame.shape[1], MANEUVER_SPEED, bias,
                )
            else:
                self.last_binary = lane_binary
                self._run_timed_turn_phase(now)

            if self.state == "MANEUVER":
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
                    rospy.logwarn(
                        "line_cy_task maneuver timeout, complete current intersection"
                    )
                    self._complete_intersection()
                    if self.state == "FOLLOW" and observation.valid:
                        self._control(observation.center_x, frame.shape[1], FOLLOW_SPEED,
                                      FOLLOW_CENTER_BIAS_PIXELS)
                    elif self.state == "FINAL_EXIT":
                        self.publish(self.turn_speed, 0.0)
                    else:
                        self.publish(0, 0)

        else:
            self.publish(0, 0)

        if self.debug_view:
            self.draw_debug(frame)
        if getattr(self, "yolo_debug_view", False):
            self.draw_yolo_debug()

    def draw_yolo_debug(self):
        with self.yolo_lock:
            if self.yolo_latest_frame is None:
                return
            frame = self.yolo_latest_frame.copy()
            detections = list(self.yolo_latest_detections)

        frame = draw_yolo_boxes(
            frame, detections,
            getattr(self, "yolo_center_band_ratio", YOLO_CENTER_BAND_RATIO),
        )
        status = "YOLO frame_interval={} detections={}".format(
            getattr(self, "yolo_frame_interval", YOLO_FRAME_INTERVAL),
            len(detections)
        )
        cv2.putText(frame, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
        try:
            cv2.imshow(YOLO_WINDOW_NAME, frame)
            cv2.waitKey(1)
        except cv2.error:
            self.yolo_debug_view = False

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
        align_lock = getattr(self, "align_lock", None)
        align_angle = (getattr(self, "align_last_angle", None)
                       if align_lock is None else align_lock["angle"])
        lock_text = "--" if align_angle is None else "{:.1f}".format(align_angle)
        text = ("task={}/{} state={} cmd={} phase={} side={} lane={:.0f} dual={} "
                "ctrl={:+.2f} stripes={} bar={} angle={} lock={} hits={} "
                "cross={:.2f}").format(
            self.task_index + 1, len(TASK_TURN_COMMANDS),
            self.state, self.turn_cmd, self.maneuver_phase,
            side, self.lane_width,
            observation.dual_rows,
            getattr(self, "last_command_angular", 0.0),
            len(self.last_crosswalk.stripe_polygons),
            bar_state, angle_text, lock_text, self.crosswalk.bar_only_hits,
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
            self.yolo_running = False
            if self.yolo_thread is not None:
                self.yolo_thread.join(1.0)
            if self.yolo_detector is not None:
                self.yolo_detector.close()
                self.yolo_detector = None
            self._close_traffic_light()
            self.camera.release()
            if self.yolo_camera is not None:
                self.yolo_camera.release()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        LaneFollower().run()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        pass
