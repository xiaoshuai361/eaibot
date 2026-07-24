#!/usr/bin/env python
# coding=utf-8

import rospy
import cv2
import numpy as np
from geometry_msgs.msg import Twist
import time
import os
import torch
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from collections import deque
import math  # 用于计算距离


# ---------------------- 人群识别模块 ----------------------
class PeopleConfig:
    # 摄像头配置
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CONFIDENCE_THRESHOLD = 0.7
    FONT_PATH = "/home/eaibot/robocom_ws/src/ziti.ttf"  # 字体路径
    PT_MODEL_PATH = "/home/eaibot/robocom_ws/src/people_best.pt"  # 人群识别模型路径
    # 去重参数
    DEDUPLICATION_THRESHOLD = 10  # 中心点距离阈值（像素）


# 创建输出目录
os.makedirs('people_out', exist_ok=True)
os.makedirs('result', exist_ok=True)
os.makedirs('saved_people_images', exist_ok=True)  # 确保人群识别图片目录存在

# 人群识别全局变量
people_model = None
people_cached_font = None

# 人员类别映射
people_class_info = {
    'zhiye': {'name': '职业人员', 'color': (0, 255, 0)},
    'putong': {'name': '普通人员', 'color': (0, 0, 255)}
}


def init_people_model_once():
    """初始化YOLOv5人群识别模型（单例模式）"""
    global people_model
    if people_model is not None:
        return people_model

    # 检查模型文件是否存在
    if not os.path.exists(PeopleConfig.PT_MODEL_PATH):
        rospy.logerr(f"【错误】人群识别模型文件不存在: {PeopleConfig.PT_MODEL_PATH}")
        rospy.logerr(f"【提示】当前工作目录: {os.getcwd()}")
        return None

    try:
        # 检查YOLOv5目录是否存在
        if not os.path.exists("yolov5-master"):
            rospy.logwarn("【警告】未找到yolov5-master目录，尝试从GitHub加载")
            people_model = torch.hub.load(
                "ultralytics/yolov5",
                'custom',
                path=PeopleConfig.PT_MODEL_PATH,
                device='cpu',
                force_reload=True
            )
        else:
            # 从本地加载YOLOv5
            people_model = torch.hub.load(
                "yolov5-master",
                'custom',
                path=PeopleConfig.PT_MODEL_PATH,
                source='local',
                device='cpu',
                force_reload=True
            )

        people_model.conf = PeopleConfig.CONFIDENCE_THRESHOLD
        people_model.imgsz = (PeopleConfig.CAMERA_HEIGHT, PeopleConfig.CAMERA_WIDTH)
        rospy.loginfo("【成功】人群识别模型加载完成")
        return people_model
    except Exception as e:
        rospy.logerr(f"【错误】人群识别模型加载失败: {str(e)}")
        rospy.logerr("【可能原因】")
        rospy.logerr("1. YOLOv5目录不存在或路径错误")
        rospy.logerr("2. 模型文件损坏或不是有效的YOLOv5模型")
        rospy.logerr("3. PyTorch环境未正确配置")
        return None


def get_people_font(font_size=10):
    """获取中文字体对象（缓存机制）"""
    global people_cached_font
    if people_cached_font and people_cached_font[0] == font_size:
        return people_cached_font[1]

    try:
        if os.path.exists(PeopleConfig.FONT_PATH):
            font = ImageFont.truetype(PeopleConfig.FONT_PATH, font_size)
            people_cached_font = (font_size, font)
            return font
        else:
            raise FileNotFoundError(f"字体不存在: {PeopleConfig.FONT_PATH}")
    except Exception as e:
        # 字体加载失败的兼容处理
        class DummyFont:
            def getbbox(self, text):
                return (0, 0, len(text) * 6, font_size)

        people_cached_font = (font_size, DummyFont())
        return people_cached_font[1]


def cv2_add_people_text(img, text, position, font_size=10, color=(255, 255, 255)):
    """在图像上添加中文文本（人群识别专用）"""
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text(position, text, font=get_people_font(font_size), fill=tuple(reversed(color)))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                           font_size / 40, color, 1)


def deduplicate_people_detections(detections):
    """对人员检测结果进行去重处理"""
    if detections.empty:
        return detections

    # 转换为列表以便处理
    detection_list = []
    for _, row in detections.iterrows():
        # 计算中心点
        cx = (row['xmin'] + row['xmax']) / 2
        cy = (row['ymin'] + row['ymax']) / 2
        detection_list.append({
            'row': row,
            'cx': cx,
            'cy': cy,
            'used': False
        })

    # 筛选结果
    filtered = []
    for i in range(len(detection_list)):
        if detection_list[i]['used']:
            continue

        current = detection_list[i]
        current_row = current['row']
        duplicates = [current]

        # 查找同类别且距离近的检测框
        for j in range(i + 1, len(detection_list)):
            if detection_list[j]['used']:
                continue

            other = detection_list[j]
            other_row = other['row']

            # 只对同一类别的进行去重
            if current_row['name'] != other_row['name']:
                continue

            # 计算欧氏距离
            distance = math.sqrt(
                (current['cx'] - other['cx']) ** 2 +
                (current['cy'] - other['cy']) ** 2
            )

            if distance < PeopleConfig.DEDUPLICATION_THRESHOLD:
                duplicates.append(other)
                detection_list[j]['used'] = True

        # 选择置信度最高的
        best = max(duplicates, key=lambda x: x['row']['confidence'])
        filtered.append(best['row'])

    # 转换回DataFrame
    if filtered:
        return detections[detections.index.isin([row.name for row in filtered])]
    return detections.iloc[0:0]  # 返回空DataFrame


def detect_and_visualize_people(img):
    """检测并可视化人群，返回处理结果和数量统计"""
    try:
        # 调整图像尺寸
        if img.shape[1] != PeopleConfig.CAMERA_WIDTH or img.shape[0] != PeopleConfig.CAMERA_HEIGHT:
            img = cv2.resize(img, (PeopleConfig.CAMERA_WIDTH, PeopleConfig.CAMERA_HEIGHT))

        # 确保模型已加载
        global people_model
        if people_model is None:
            people_model = init_people_model_once()
            if not people_model:
                return None, {'zhiye': 0, 'putong': 0}

        results = people_model(img)
        detections = results.pandas().xyxy[0]

        # 对检测结果进行去重
        detections = deduplicate_people_detections(detections)

        class_counts = {'zhiye': 0, 'putong': 0}
        vis_img = img.copy()

        if not detections.empty:
            font = get_people_font()  # 预加载字体

            for _, row in detections.iterrows():
                class_name = row['name']
                if class_name not in class_counts or row['confidence'] < PeopleConfig.CONFIDENCE_THRESHOLD:
                    continue

                class_counts[class_name] += 1
                x1, y1, x2, y2 = map(int, row[['xmin', 'ymin', 'xmax', 'ymax']])
                info = people_class_info[class_name]

                # 绘制边界框
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), info['color'], 1)

                # 绘制标签
                label = f"{info['name']}:{row['confidence']:.1f}"
                text_bbox = font.getbbox(label)
                text_pos = (x1, max(15, y1 - (text_bbox[3] - text_bbox[1]) - 2))

                # 绘制标签背景和文本
                img_pil = Image.fromarray(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                draw.rectangle([
                    (text_pos[0] - 1, text_pos[1] - 1),
                    (text_pos[0] + text_bbox[2] - text_bbox[0] + 1, text_pos[1] + text_bbox[3] - text_bbox[1] + 1)
                ], fill=info['color'] + (200,))
                draw.text(text_pos, label, font=font, fill=(255, 255, 255))
                vis_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # 显示统计信息
        total = class_counts['zhiye'] + class_counts['putong']
        stats_text = f"总人数: {total} | 职业: {class_counts['zhiye']} | 普通: {class_counts['putong']}"
        vis_img = cv2_add_people_text(vis_img, stats_text, (15, 15), 14, (0, 255, 255))

        return vis_img, class_counts

    except Exception as e:
        rospy.logerr(f"人群检测错误: {e}")
        return None, {'zhiye': 0, 'putong': 0}


# ---------------------- 火灾检测模块 ----------------------
class FireConfig:
    # 摄像头配置
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480

    # 检测阈值配置
    CONFIDENCE_THRESHOLD = 0.2
    FIRE_CONFIDENCE_THRESHOLD = 0.2
    BUILDING_CONFIDENCE_BOOST = 0.25  # 电子超市置信度补偿

    # 平滑窗口配置
    FIRE_SMOOTH_WINDOW = 3
    BUILDING_SMOOTH_WINDOW = 5

    # 文件路径配置
    FONT_PATH = "/home/eaibot/robocom_ws/src/ziti.ttf"  # 共享字体路径
    PT_MODEL_PATH = "/home/eaibot/robocom_ws/src/fire_best.pt"

    # 楼层配置（6层逻辑）
    FLOOR_INTERVALS = [
        (0.833, 1.0),  # 1层（最下方）
        (0.666, 0.833),  # 2层
        (0.5, 0.666),  # 3层
        (0.333, 0.5),  # 4层
        (0.166, 0.333),  # 5层
        (0.0, 0.166)  # 6层（最上方）
    ]


# 创建火灾检测输出目录
os.makedirs('saved_fire_images', exist_ok=True)

# 火灾检测全局变量
fire_model = None
fire_cached_font = None
building_history = deque(maxlen=FireConfig.BUILDING_SMOOTH_WINDOW)

# 类别信息映射
fire_class_info = {
    'Building': {'name': '建筑物', 'color': (255, 255, 0)},  # 黄色
    'Fire': {'name': '火灾', 'color': (0, 0, 255)},  # 红色
    'Meili': {'name': '美丽商场', 'color': (0, 255, 255)},  # 青色
    'DianZi': {'name': '电子超市', 'color': (0, 255, 0)}  # 绿色
}


def init_fire_model_once():
    """初始化YOLOv5火灾检测模型（单例模式）"""
    global fire_model
    if fire_model is not None:
        return fire_model

    if not os.path.exists(FireConfig.PT_MODEL_PATH):
        rospy.logerr(f"【错误】火灾模型文件不存在: {FireConfig.PT_MODEL_PATH}")
        return None

    try:
        # 加载模型
        if not os.path.exists("yolov5-master"):
            fire_model = torch.hub.load(
                "ultralytics/yolov5",
                'custom',
                path=FireConfig.PT_MODEL_PATH,
                device='cpu',
                force_reload=True
            )
        else:
            fire_model = torch.hub.load(
                "yolov5-master",
                'custom',
                path=FireConfig.PT_MODEL_PATH,
                source='local',
                device='cpu',
                force_reload=True
            )

        # 配置模型参数
        fire_model.conf = FireConfig.CONFIDENCE_THRESHOLD
        fire_model.nms = 0.2
        fire_model.imgsz = (FireConfig.CAMERA_HEIGHT, FireConfig.CAMERA_WIDTH)
        return fire_model
    except Exception as e:
        rospy.logerr(f"【错误】火灾模型加载失败: {str(e)}")
        return None


def get_fire_font(font_size=10):
    """获取中文字体对象（缓存机制）"""
    global fire_cached_font
    if fire_cached_font and fire_cached_font[0] == font_size:
        return fire_cached_font[1]

    try:
        if os.path.exists(FireConfig.FONT_PATH):
            font = ImageFont.truetype(FireConfig.FONT_PATH, font_size)
            fire_cached_font = (font_size, font)
            return font
        else:
            raise FileNotFoundError(f"字体不存在: {FireConfig.FONT_PATH}")
    except Exception as e:
        # 字体加载失败的兼容处理
        class DummyFont:
            def getbbox(self, text):
                return (0, 0, len(text) * 6, font_size)

        fire_cached_font = (font_size, DummyFont())
        return fire_cached_font[1]


def cv2_add_fire_text(img, text, position, font_size=10, color=(255, 255, 255)):
    """在图像上添加中文文本（火灾检测专用）"""
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text(position, text, font=get_fire_font(font_size), fill=tuple(reversed(color)))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                           font_size / 40, color, 1)


def determine_floor(building_box, fire_boxes_with_conf):
    """判断火灾所在楼层（6层逻辑）"""
    if not building_box or not fire_boxes_with_conf:
        return {}

    x1_b, y1_b, x2_b, y2_b = building_box
    building_height = y2_b - y1_b
    if building_height <= 0:
        return {}

    floor_confidences = {i + 1: [] for i in range(6)}

    for fire_box, conf in fire_boxes_with_conf:
        x1_f, y1_f, x2_f, y2_f = fire_box
        fire_center_y = (y1_f + y2_f) / 2
        relative_pos = (fire_center_y - y1_b) / building_height
        relative_pos = max(0.0, min(1.0, relative_pos))

        # 匹配楼层（高楼层增加缓冲）
        for floor_idx in range(6):
            lower, upper = FireConfig.FLOOR_INTERVALS[floor_idx]
            if floor_idx >= 4:  # 5、6层增加边界缓冲
                adj_lower = max(0.0, lower - 0.01)
                adj_upper = min(1.0, upper + 0.01)
                if adj_lower <= relative_pos <= adj_upper:
                    floor_confidences[floor_idx + 1].append(conf)
                    break
            else:
                if lower <= relative_pos <= upper:
                    floor_confidences[floor_idx + 1].append(conf)
                    break

    return floor_confidences


def detect_and_visualize_fire(img, model):
    """检测并绘制识别框，返回处理结果"""
    try:
        # 调整图像大小
        if img.shape[1] != FireConfig.CAMERA_WIDTH or img.shape[0] != FireConfig.CAMERA_HEIGHT:
            img = cv2.resize(img, (FireConfig.CAMERA_WIDTH, FireConfig.CAMERA_HEIGHT))

        # 模型检测
        results = model(img)
        detections = results.pandas().xyxy[0]

        vis_img = img.copy()
        building_name = '未知楼宇'
        building_box = None
        fire_boxes_with_conf = []
        building_types = []

        if not detections.empty:
            # 遍历检测结果
            for _, row in detections.iterrows():
                class_name = row['name']
                conf = float(row['confidence'])

                # 过滤低置信度目标
                if class_name == 'Fire' and conf < FireConfig.FIRE_CONFIDENCE_THRESHOLD:
                    continue
                if class_name != 'Fire' and conf < FireConfig.CONFIDENCE_THRESHOLD:
                    continue

                # 计算边界框坐标
                x1, y1, x2, y2 = map(int, row[['xmin', 'ymin', 'xmax', 'ymax']])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(FireConfig.CAMERA_WIDTH, x2), min(FireConfig.CAMERA_HEIGHT, y2)

                # 处理楼宇类型
                if class_name == 'Meili':
                    building_types.append(('美丽商场', conf, (x1, y1, x2, y2)))
                elif class_name == 'DianZi':
                    adjusted_conf = conf + FireConfig.BUILDING_CONFIDENCE_BOOST
                    building_types.append(('电子超市', adjusted_conf, (x1, y1, x2, y2)))
                elif class_name == 'Building':
                    building_box = (x1, y1, x2, y2)
                    # 绘制建筑物框
                    info = fire_class_info[class_name]
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), info['color'], 2)
                elif class_name == 'Fire':
                    fire_boxes_with_conf.append(((x1, y1, x2, y2), conf))
                    # 绘制火灾框
                    info = fire_class_info[class_name]
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), info['color'], 2)
                    # 绘制火灾标签
                    label = f"火灾:{conf:.2f}"
                    text_pos = (x1, y1 - 25) if y1 > 25 else (x1, y2 + 20)
                    vis_img = cv2_add_fire_text(vis_img, label, text_pos, 16, info['color'])

            # 确定楼宇类型并绘制标签
            if building_types:
                building_types.sort(key=lambda x: x[1], reverse=True)
                best_name, best_conf, (x1, y1, x2, y2) = building_types[0]

                # 平滑结果（历史投票）
                building_history.append(best_name)
                building_name = max(set(building_history), key=building_history.count)

                # 绘制楼宇框和标签
                info = next(v for k, v in fire_class_info.items() if v['name'] == building_name)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), info['color'], 2)
                label = f"{building_name}:{best_conf:.2f}"
                text_pos = (x1, y1 - 25) if y1 > 25 else (x1, y2 + 20)
                vis_img = cv2_add_fire_text(vis_img, label, text_pos, 16, info['color'])

        # 判断楼层并终端输出
        floor_fire_confidences = determine_floor(building_box, fire_boxes_with_conf)
        if building_name != '未知楼宇':
            rospy.loginfo(f"【楼宇类型】{building_name}")
            if floor_fire_confidences:
                for floor, confidences in sorted(floor_fire_confidences.items()):
                    if confidences:
                        avg_conf = round(sum(confidences) / len(confidences), 2)
                        rospy.loginfo(f"【火灾位置】第{floor}层, 置信度: {avg_conf}")
            else:
                rospy.loginfo("【火灾位置】未检测到火灾")
            rospy.loginfo("-" * 30)  # 分隔线

        return vis_img, len(fire_boxes_with_conf), building_name, floor_fire_confidences

    except Exception as e:
        rospy.logerr(f"检测错误: {e}")
        return None, None, None, None


def fire_detection(frame, save_dir="saved_fire_images"):
    """火灾检测主函数（适配ROS调用）"""
    global fire_model
    if fire_model is None:
        fire_model = init_fire_model_once()
        if not fire_model:
            rospy.logerr("【错误】模型加载失败，程序终止")
            return None

    os.makedirs(save_dir, exist_ok=True)
    results = []
    start_time = time.time()
    fire_history = deque(maxlen=FireConfig.FIRE_SMOOTH_WINDOW)
    floor_total_counts = {i: 0 for i in range(1, 7)}  # 1-6层统计

    try:
        # 处理单帧图像（适配停车检测逻辑）
        img_name = f"fire_{int(time.time() * 1000)}.jpg"
        result = detect_and_visualize_fire(frame, fire_model)

        if result[0] is not None:
            vis_img, fire_count, building_name, floor_counts = result

            fire_history.append(fire_count)
            smoothed_fire_count = sum(fire_history) // len(fire_history)

            # 累加楼层统计
            for floor in range(1, 7):
                floor_total_counts[floor] += len(floor_counts.get(floor, []))

            # 保存检测结果
            save_path = os.path.join(save_dir, img_name)
            cv2.imwrite(save_path, vis_img)
            results.append((vis_img, img_name, smoothed_fire_count, building_name, floor_counts))
            rospy.loginfo(f"火灾检测结果已保存: {save_path}")

        # 打印总结
        rospy.loginfo("\n【检测总结】")
        rospy.loginfo(f"检测时长: {time.time() - start_time:.2f}秒")
        rospy.loginfo(f"检测到的火灾总数: {sum(fire_history)}")

        rospy.loginfo("\n【楼层火灾总计】")
        has_fire = False
        for floor in range(1, 7):
            count = floor_total_counts[floor]
            if count > 0:
                has_fire = True
                rospy.loginfo(f"第{floor}层累计火点: {count}个")
        if not has_fire:
            rospy.loginfo("未检测到任何火点")

        return results

    except Exception as e:
        rospy.logerr(f"火灾检测异常: {e}")
        return None


# ---------------------- 车道跟随与其他检测模块 ----------------------

# 车道方向计算函数
def calculate_lane_direction(image, kalman_filter, last_mid_point_x, failed_count, reset_threshold=5):
    h, w, _ = image.shape

    # 黑色线提取：颜色过滤 + 自适应阈值
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    color_mask = cv2.inRange(hsv, lower_black, upper_black)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 21, 5)
    binary = cv2.bitwise_and(adaptive_thresh, color_mask)

    search_top = int(h * 0.32)
    search_bot = int(h * 0.5)
    binary[:search_top, :] = 0
    binary[search_bot:, :] = 0

    center_points = []
    mid_point_x = last_mid_point_x
    current_mid_raw = -1  # 默认无法检测到

    for y in range(search_bot, search_top - 1, -1):
        row_pixels = np.where(binary[y, :] == 255)[0]

        if len(row_pixels) > 2:
            pixel_groups = []
            current_group = [row_pixels[0]]
            for i in range(1, len(row_pixels)):
                if row_pixels[i] - row_pixels[i - 1] < 20:
                    current_group.append(row_pixels[i])
                else:
                    if len(current_group) > 2:
                        pixel_groups.append(current_group)
                    current_group = [row_pixels[i]]
            if len(current_group) > 2:
                pixel_groups.append(current_group)

            for group in pixel_groups:
                cx = int(np.mean(group))
                cv2.circle(image, (cx, y), 3, (255, 0, 0), -1)  # 红色圆点画每个分组中心

            # 过滤水平方向的线
            valid_groups = []
            for group in pixel_groups:
                left, right = group[0], group[-1]
                if (right - left) / (search_bot - y) < 1.5:  # 斜率小于1.5，认为不是水平线
                    valid_groups.append(group)

            if len(valid_groups) >= 3:
                # 三条或更多，只使用最右侧线
                rightmost_group = valid_groups[-1]
                right = rightmost_group[-1]
                current_mid_raw = right - 10  # 可选偏移，防止贴边太近
                rospy.logwarn("3+ lines detected, following rightmost line at x=%d", current_mid_raw)
                break

            elif len(valid_groups) == 2:
                # 取中线
                left, right = valid_groups[0][-1], valid_groups[1][0]
                if (abs(right - left) >= 300):
                    current_mid_raw = (left + right) // 2
                else:
                    left, right = valid_groups[0][0], valid_groups[0][-1]
                    if (left + right) / 2 < w / 2:
                        current_mid_raw = (right + w) // 2
                    else:
                        current_mid_raw = left // 2
                break

            elif len(valid_groups) == 1:
                left, right = valid_groups[0][0], valid_groups[0][-1]
                if (left + right) / 2 < w / 2:
                    current_mid_raw = (right + w) // 2
                else:
                    current_mid_raw = left // 2
                break

    predicted = kalman_filter.predict()

    if current_mid_raw != -1 and abs(current_mid_raw - mid_point_x) < w * 0.4:
        measurement = np.array([[np.float32(current_mid_raw)]])
        mid_point_x = int(kalman_filter.correct(measurement)[0])
        failed_count = 0  # 成功检测到，重置失败计数
    else:
        mid_point_x = int(predicted[0])
        failed_count += 1  # 检测失败次数增加

        # 若连续多次失败，重置kalman滤波器
        if failed_count >= reset_threshold:
            kalman_filter.statePost = np.array([[w // 2], [0]], np.float32)
            mid_point_x = w // 2
            failed_count = 0  # 重置失败计数

    center_points.append((mid_point_x, search_bot))
    # 偏移量计算
    deviation = mid_point_x - (w / 2)
    return deviation, center_points, failed_count


class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0.0, output_limits=(-1.0, 1.0)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = None

    def update(self, measured_value):
        current_time = rospy.get_time()
        if self.last_time is None:
            self.last_time = current_time
            dt = 0.1
        else:
            dt = current_time - self.last_time
            dt = max(dt, 0.001)

        error = self.setpoint - measured_value
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = max(self.output_limits[0], min(self.output_limits[1], output))

        self.last_error = error
        self.last_time = current_time
        return output


# 垃圾桶检测器类
class RubbishDetector:
    def __init__(self, model_path="rubbish_best.pt", device='cpu', conf_thres=0.4, save_dir="saved_rubbish_images"):
        self.device = device
        self.conf_thres = conf_thres
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 垃圾类别映射
        self.class_info = {
            0: '有害垃圾_未投放',
            1: '厨余垃圾_已投放',
            2: '厨余垃圾_未投放',
            3: '可回收物_已投放',
            4: '可回收物_未投放',
            5: '其他垃圾_已投放',
            6: '其他垃圾_未投放',
            7: '有害垃圾_已投放'
        }

        # 加载模型
        try:
            self.model = torch.hub.load(
                "yolov5-master",
                'custom',
                path=model_path,
                source='local',
                device=device,
                force_reload=True
            )
            self.model.conf = conf_thres
            rospy.loginfo(f"垃圾桶检测器初始化成功，使用设备: {device}")
        except Exception as e:
            rospy.logerr(f"垃圾桶检测器初始化失败: {e}")
            self.model = None

    def detect(self, frame):
        """检测图片中的垃圾桶，返回检测结果和统计信息"""
        if self.model is None:
            rospy.logwarn("垃圾桶检测器未初始化，跳过检测")
            return [], {}, True

        try:
            # 执行检测
            results = self.model(frame)
            detections = results.pandas().xyxy[0]

            detected_objects = []
            class_counts = {cls_name: 0 for cls_name in self.class_info.values()}

            # 处理检测结果
            for _, row in detections.iterrows():
                class_id = int(row['class'])
                conf = float(row['confidence'])

                if conf < self.conf_thres or class_id not in self.class_info:
                    continue

                cls_name = self.class_info[class_id]
                class_counts[cls_name] += 1

                detected_objects.append({
                    'class_name': cls_name,
                    'confidence': conf,
                    'bbox': [int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])]
                })

            # 计算统计信息
            total_number = sum(class_counts.values())

            stats = {
                'total': total_number,
                'class_counts': class_counts
            }

            # 使用模型自带的渲染方法
            rendered_frame = results.render()[0] if detected_objects else None

            if rendered_frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.save_dir, f"rubbish_{timestamp}.jpg")
                cv2.imwrite(save_path, rendered_frame)
                rospy.loginfo(f"垃圾桶检测结果已保存: {save_path}")

            return detected_objects, stats, False

        except Exception as e:
            rospy.logerr(f"垃圾桶检测过程中出错: {e}")
            return [], {}, True


# 主控制器类
class LaneFollower:
    def __init__(self):
        rospy.init_node("lane_follower", anonymous=True)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.twist = Twist()

        self.pid = PIDController(kp=0.005, ki=0.0001, kd=0.0015, output_limits=(-1.0, 1.0))
        self.kalman = cv2.KalmanFilter(2, 1)
        self.kalman.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
        self.kalman.measurementMatrix = np.array([[1, 0]], np.float32)
        self.kalman.processNoiseCov = np.eye(2, dtype=np.float32) * 1e-4
        self.kalman.measurementNoiseCov = np.array([[1]], np.float32) * 1e-1
        self.initial_state_set = False
        self.last_mid = 320
        self.failed_count = 0

        # 打开摄像头 - 索引2用于车道线检测（寻线）
        self.cap_lane = cv2.VideoCapture(2)
        if not self.cap_lane.isOpened():
            rospy.logerr("无法打开车道线检测摄像头（索引2）")
            rospy.signal_shutdown("无法打开车道线检测摄像头")

        # 打开摄像头 - 索引0用于人群识别和垃圾桶识别
        self.cap_doll = cv2.VideoCapture(0)
        if not self.cap_doll.isOpened():
            rospy.logerr("无法打开识别摄像头（索引0）")
            rospy.signal_shutdown("无法打开识别摄像头")

        rospy.loginfo("成功打开两个摄像头：索引0用于目标识别，索引2用于车道线检测")

        # 添加计时器
        self.start_time = rospy.get_time()
        self.straight_mode = False

        # 添加检测相关状态
        self.detection_intervals = [
            {"start": 32.5, "end": 33.5, "done": False, "type": "fire"},  # 第一次火灾检测
            {"start": 57, "end": 58, "done": False, "type": "people"},
            {"start": 68.5, "end": 69.5, "done": False, "type": "people"},
            {"start": 91, "end": 92, "done": False, "type": "fire"},  # 第二次火灾检测
            {"start": 104, "end": 105, "done": False, "type": "people"},
            {"start": 117, "end": 118, "done": False, "type": "rubbish"},  # 垃圾桶检测时段
            {"start": 129.5, "end": 130.5, "done": False, "type": "people"}
        ]
        self.program_start_time = time.time()  # 记录程序开始运行的时间
        self.paused_time = 0  # 累计暂停的时间

        # 初始化人群检测器（使用新的人群识别模块）
        rospy.loginfo("初始化人群检测器...")
        # 预加载人群识别模型
        global people_model
        people_model = init_people_model_once()
        rospy.loginfo(f"人群检测器初始化完成，使用设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

        # 初始化垃圾桶检测器
        self.rubbish_detector = RubbishDetector(
            model_path="/home/eaibot/robocom_ws/src/rubbish_best.pt",
            device="cuda" if torch.cuda.is_available() else "cpu",
            conf_thres=0.4,
            save_dir="/home/eaibot/robocom_ws/src/saved_rubbish_images"
        )
        rospy.loginfo(f"垃圾桶检测器初始化完成，使用设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

        # 新增状态变量
        self.people_count = 0  # 已检测到的总人数
        self.people_detection_count = 0  # 已完成的人群检测次数（用于区分A/B/C/D区域）
        self.slow_mode = False  # 是否进入慢速模式
        self.ramp_mode = False  # 是否进入坡道模式
        self.ramp_start_time = None  # 坡道模式开始时间
        self.center_points = []  # 初始化center_points
        self.lines_detected = False  # 是否检测到双黑线
        self.ramp_mode_entered = False  # 坡道模式是否已经进入过
        self.ramp_mode_completed = False  # 坡道模式是否已经完成

    def get_elapsed_time(self):
        """获取经过的时间（排除暂停时间）"""
        return time.time() - self.program_start_time - self.paused_time

    def process_frame(self):
        # 从车道线检测摄像头读取一帧（用于寻线）
        ret_lane, frame_lane = self.cap_lane.read()
        if not ret_lane:
            rospy.logerr("无法读取车道线检测摄像头图像")
            return

        # 从识别摄像头读取一帧（用于人群和垃圾桶识别）
        ret_doll, frame_doll = self.cap_doll.read()
        if not ret_doll:
            rospy.logerr("无法读取识别摄像头图像")
            # 即使无法读取识别摄像头，仍然可以继续车道线检测

        h, w, _ = frame_lane.shape
        elapsed_time = self.get_elapsed_time()  # 使用修正后的经过时间

        # 检查是否进入检测区间
        for interval in self.detection_intervals:
            if interval["start"] <= elapsed_time < interval["end"] and not interval["done"]:
                # 停车
                self.twist.linear.x = 0
                self.twist.angular.z = 0
                self.cmd_pub.publish(self.twist)

                rospy.loginfo(
                    f"在{elapsed_time:.2f}秒开始{interval['type']}检测 (区间 {interval['start']}-{interval['end']}秒)")

                # 等待1秒确保小车完全停止
                rospy.sleep(1.0)

                if ret_doll:
                    # 执行检测
                    detect_start_time = time.time()

                    if interval["type"] == "people":
                        # 递增检测次数，确定区域（A/B/C/D）
                        self.people_detection_count += 1
                        area = chr(64 + self.people_detection_count)  # 64是ASCII码中'A'的前一位，1→A，2→B...

                        # 人群检测（使用新模块）
                        vis_img, counts = detect_and_visualize_people(frame_doll)
                        zhiye_count = counts.get('zhiye', 0)
                        putong_count = counts.get('putong', 0)

                        # 终端打印区域信息
                        rospy.loginfo(f"{area}区域职业人员{zhiye_count}名，普通人员{putong_count}名")

                        # 计算总人数
                        total_people = zhiye_count + putong_count
                        self.people_count += total_people  # 累计人群数量

                        # 保存检测结果（按区域命名）
                        if vis_img is not None:
                            img_name = f"{area}区域职业人员{zhiye_count}名，普通人员{putong_count}名.jpg"
                            save_path = os.path.join("saved_people_images", img_name)
                            cv2.imwrite(save_path, vis_img)
                            rospy.loginfo(f"人群检测结果已保存: {save_path}")

                        # 第四次检测后进入慢速模式
                        if self.people_detection_count == 4:
                            self.slow_mode = True
                            rospy.loginfo("第四次人群检测完成，进入慢速模式")

                    elif interval["type"] == "rubbish":
                        # 垃圾桶检测
                        detected_objects, stats, task_end = self.rubbish_detector.detect(frame_doll)

                        # 打印统计信息
                        rospy.loginfo(f"检测到垃圾桶总数: {stats.get('total', 0)}")
                        class_counts = stats.get('class_counts', {})
                        # 确保所有八个类别都被打印，即使计数为0
                        for class_id in range(8):
                            cls_name = self.rubbish_detector.class_info.get(class_id)
                            if cls_name:
                                count = class_counts.get(cls_name, 0)
                                rospy.loginfo(f"{cls_name}: {count}")
                    elif interval["type"] == "fire":
                        # 火灾检测
                        fire_result = fire_detection(frame_doll,
                                                     save_dir="/home/eaibot/robocom_ws/src/saved_fire_images")
                        if fire_result:
                            rospy.loginfo(f"火灾检测完成")

                    # 计算实际检测耗时
                    actual_detect_time = time.time() - detect_start_time
                    remaining_pause = max(0, 2.0 - actual_detect_time)
                    rospy.sleep(remaining_pause)
                else:
                    rospy.logwarn("无法读取识别摄像头，跳过检测")
                    rospy.sleep(2.0)  # 仍然等待2秒

                # 更新暂停时间（固定增加2秒）
                self.paused_time += 2.0

                interval["done"] = True
                rospy.loginfo(
                    f"{interval['type']}检测完成 (区间 {interval['start']}-{interval['end']}秒), 累计暂停时间: {self.paused_time:.2f}秒")
                return  # 跳过本次控制循环

        # 检查是否进入慢速模式
        if self.slow_mode and not self.ramp_mode and not self.ramp_mode_completed:
            self.twist.linear.x = 0.1  # 减慢速度
            deviation, self.center_points, self.failed_count = calculate_lane_direction(
                frame_lane, self.kalman, self.last_mid, self.failed_count
            )
            self.last_mid = self.center_points[-1][0]
            angular_z = self.pid.update(deviation)
            self.twist.angular.z = angular_z
            self.cmd_pub.publish(self.twist)

            # 检查是否检测到双黑线
            if abs(deviation) < 3:  # 偏差小于10像素，认为检测到双黑线
                self.lines_detected = True
                rospy.loginfo("检测到双黑线，准备进入坡道模式")
            else:
                self.lines_detected = False

            # 检查是否可以进入坡道模式
            if self.lines_detected and not self.ramp_mode and not self.ramp_mode_entered:
                self.ramp_mode = True
                self.ramp_start_time = rospy.get_time()
                self.ramp_mode_entered = True  # 标记已经进入过坡道模式
                rospy.loginfo("进入坡道模式")

        # 检查是否进入坡道模式
        if self.ramp_mode:
            self.twist.linear.x = 0.2  # 恢复原来速度
            self.twist.angular.z = 0  # 不识别黑线
            self.cmd_pub.publish(self.twist)

            # 检查坡道模式是否结束
            if rospy.get_time() - self.ramp_start_time >= 10:  # 坡道模式持续10秒
                self.ramp_mode = False
                self.ramp_mode_completed = True  # 标记坡道模式已完成
                self.lines_detected = False
                rospy.loginfo("坡道模式结束，恢复巡线模式")

        # 正常巡线模式
        if not self.ramp_mode and (not self.slow_mode or self.ramp_mode_completed):
            if not self.initial_state_set:
                self.last_mid = w // 2
                self.kalman.statePost = np.array([[self.last_mid], [0]], np.float32)
                self.initial_state_set = True

            deviation, self.center_points, self.failed_count = calculate_lane_direction(
                frame_lane, self.kalman, self.last_mid, self.failed_count
            )
            self.last_mid = self.center_points[-1][0]

            # 动态调整 PID 参数
            deviation_threshold = 100  # 偏差阈值
            if abs(deviation) < deviation_threshold:
                self.pid.kp = 0.0035  # 较小的 kp
                self.pid.kd = 0.0008  # 较小的 kd
            else:
                self.pid.kp = 0.035  # 较大的 kp
                self.pid.kd = 0.002  # 较大的 kd

            angular_z = self.pid.update(deviation)
            self.twist.linear.x = 0.2
            self.twist.angular.z = angular_z
            self.cmd_pub.publish(self.twist)

        # 调试圆点绘制
        for x, y in self.center_points:
            cv2.circle(frame_lane, (x, y), 4, (0, 255, 0), -1)
        cv2.imshow("Lane", frame_lane)
        cv2.waitKey(3)

    def run(self):
        rate = rospy.Rate(20)  # 20 Hz
        while not rospy.is_shutdown():
            self.process_frame()
            rate.sleep()

        # 释放摄像头资源
        self.cap_lane.release()
        self.cap_doll.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        lane_follower = LaneFollower()
        lane_follower.run()
    except rospy.ROSInterruptException:
        pass
