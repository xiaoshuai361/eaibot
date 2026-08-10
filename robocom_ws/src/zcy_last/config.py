#!/usr/bin/env python3
# coding=utf-8
"""比赛现场参数；优先修改本文件顶部开关。"""

# ===== 机械臂抓取与投递开关（调试时优先修改） =====
ENABLE_TAG_PICK = False        # B 点有 Tag 抓取。
TAG_PICK_COUNT = 1
ENABLE_TAG_DELIVERY = True     # B 点抓取开启时，第一圈按街区识别结果投递。
ENABLE_UNTAGGED_PICK = False   # A 点无 Tag 抓取。
UNTAGGED_PICK_COUNT = 1
ENABLE_UNTAGGED_DELIVERY = True # A 点抓取开启时，后两圈按楼宇识别结果投递。
UNTAGGED_TRIGGER_INTERSECTION = 3
PICK_CANDIDATE_IDS = (1, 2, 3, 4)

# ===== 现场启动、摄像头与功能开关（运行前优先检查） =====
LANE_CAMERA_INDEX = 4          # 巡线摄像头：/dev/video4。
SHARED_OBJECT_CAMERA_INDEX = 2
YOLO_CAMERA_INDEX = SHARED_OBJECT_CAMERA_INDEX     # 人偶、垃圾桶和楼宇识别摄像头：/dev/video2。
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
TURN_TIME = 4          # 固定转弯时间(s)；转不够加，转过头减。
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
FOLLOW_LEFT_ANGULAR_SCALE = 0.76 # 所有巡线左弯力度；左弯太猛就减小。
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
STOP_NEAR_RATIO = 0.75       # 横条接近画面底部比例；停得太早加，太晚减。
STOP_CENTER_WIDTH_RATIO = 0.08 # 底部停车区占画面中央宽度比例；减小可排除两侧干扰。
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
    "Recyclable waste",
    "other waste",
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
    "Recyclable waste",
    "other waste",
)
YOLO_TARGET_CLASS_NAMES = (
    "Collapsed Building",
    "Electrical Fault Building",
    "Fire Building",
    "Toxic Gas-contaminated Building",
    "General population",
    "Medical population",
    "Recyclable waste",
    "other waste",
)
YOLO_STREET_MESSAGES = {
    "Medical population": ("people", "医疗人群"),
    "General population": ("people", "普通人群"),
    "Recyclable waste": ("trash", "可回收垃圾"),
    "other waste": ("trash", "其他垃圾"),
}
TAG_DELIVERY_ID_BY_STREET_CLASS = {
    "General population": 1,   # 普通人群：基本生活物资。
    "Medical population": 2,   # 医疗人群：医疗包。
    "Recyclable waste": 3,     # 可回收垃圾：常规消杀剂。
    "other waste": 4,          # 其他垃圾：生物危害专用消杀剂。
}
UNTAGGED_DELIVERY_ID_BY_BUILDING_CLASS = {
    "Electrical Fault Building": 1,       # 电力故障：应急电源。
    "Fire Building": 2,                   # 火灾：灭火装置。
    "Toxic Gas-contaminated Building": 3, # 有毒气体：气体净化装置。
    "Collapsed Building": 4,              # 坍塌：结构支撑装置。
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

CAMERA_INDEX = LANE_CAMERA_INDEX

# ===== 抓取进程与真机路径 =====
DEPLOY_HOME = "/home/eaibot"
GRASP_SETTLE_TIME = 1.5
PICK_RECOVER_STABLE_FRAMES = 5
PICK_RECOVER_TIMEOUT = 8.0
PROCESS_START_TIMEOUT = 35.0
PROCESS_STOP_TIMEOUT = 5.0
MANAGE_ROS_PROCESSES = True
PICK_BASE_FRAME = "base"
PICK_CAMERA_FRAME = "camera_link"

TAG_ALIGN_SCRIPT = DEPLOY_HOME + "/handeye-calib/src/tag_chassis_align_pick_sequence.py"
TAG_DELIVERY_SCRIPT = DEPLOY_HOME + "/handeye-calib/src/mirobot_delivery.py"
UNTAGGED_PICK_SCRIPT = DEPLOY_HOME + "/handeye-calib/src/block_pick_main.py"
TAG_PRESET_FILE = DEPLOY_HOME + "/handeye-calib/config/tag_pick_place_presets.json"
TAG_DELIVERY_PRESET_FILE = DEPLOY_HOME + "/handeye-calib/config/delivery_presets.json"
UNTAGGED_CONFIG_FILE = DEPLOY_HOME + "/handeye-calib/src/config/block_mono_grasp.yaml"
UNTAGGED_PRESET_FILE = DEPLOY_HOME + "/handeye-calib/config/block_mono_pick_place_presets.json"
UNTAGGED_DELIVERY_PRESET_FILE = DEPLOY_HOME + "/handeye-calib/config/untagged_delivery_presets.json"
ASTRA_CAMERA_INFO_FILE = DEPLOY_HOME + "/handeye-calib/config/astra_rgb_640x480.yaml"
PROCESS_LOG_ROOT = DEPLOY_HOME + "/logs/zcy_last"
