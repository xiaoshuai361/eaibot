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

        # 打开摄像头 - 索引0用于人群识别
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
            {"start": 7.0, "end": 8.0, "done": False, "type": "people"}  # 第7-8秒进行一次人群检测
        ]
        self.program_start_time = time.time()  # 记录程序开始运行的时间
        self.paused_time = 0  # 累计暂停的时间

        # 初始化人群检测器（使用新的人群识别模块）
        rospy.loginfo("初始化人群检测器...")
        # 预加载人群识别模型
        global people_model
        people_model = init_people_model_once()
        rospy.loginfo(f"人群检测器初始化完成，使用设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

        # 新增状态变量
        self.people_count = 0  # 已检测到的总人数
        self.people_detection_count = 0  # 已完成的人群检测次数
        self.slow_mode = False  # 是否进入慢速模式
        self.ramp_mode = False  # 是否进入坡道模式
        self.ramp_start_time = None  # 坡道模式开始时间
        self.center_points = []  # 初始化center_points
        self.lines_detected = False  # 是否检测到双黑线
        self.ramp_mode_entered = False  # 坡道模式是否已经进入过
        self.ramp_mode_completed = False  # 坡道模式是否已经完成
        self.people_detection_done = False  # 人群检测是否已完成

    def get_elapsed_time(self):
        """获取经过的时间（排除暂停时间）"""
        return time.time() - self.program_start_time - self.paused_time

    def process_frame(self):
        # 从车道线检测摄像头读取一帧（用于寻线）
        ret_lane, frame_lane = self.cap_lane.read()
        if not ret_lane:
            rospy.logerr("无法读取车道线检测摄像头图像")
            return

        # 从识别摄像头读取一帧（用于人群识别）
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
                        # 人群检测
                        vis_img, counts = detect_and_visualize_people(frame_doll)
                        zhiye_count = counts.get('zhiye', 0)
                        putong_count = counts.get('putong', 0)

                        # 终端打印区域信息
                        rospy.loginfo(f"职业人员{zhiye_count}名，普通人员{putong_count}名")

                        # 计算总人数
                        total_people = zhiye_count + putong_count
                        self.people_count += total_people  # 累计人群数量

                        # 保存检测结果
                        if vis_img is not None:
                            img_name = f"职业人员{zhiye_count}名，普通人员{putong_count}名.jpg"
                            save_path = os.path.join("saved_people_images", img_name)
                            cv2.imwrite(save_path, vis_img)
                            rospy.loginfo(f"人群检测结果已保存: {save_path}")

                        # 设置人群检测完成标志
                        self.people_detection_done = True
                        rospy.loginfo("人群检测完成，准备进入坡道模式")

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

        # 检查人群检测是否完成，准备进入坡道模式
        if self.people_detection_done and not self.ramp_mode and not self.ramp_mode_completed:
            rospy.loginfo("进入坡道模式，开始调整车身位置")
            self.ramp_mode = True
            self.ramp_start_time = rospy.get_time()
            self.ramp_mode_entered = True  # 标记已经进入过坡道模式

        # 坡道模式：慢速行驶调整车身位置
        if self.ramp_mode:
            # 计算车道方向
            deviation, self.center_points, self.failed_count = calculate_lane_direction(
                frame_lane, self.kalman, self.last_mid, self.failed_count
            )
            self.last_mid = self.center_points[-1][0]

            # 调整车身位置，保持在两条黑线中间
            angular_z = self.pid.update(deviation)
            self.twist.linear.x = 0.1  # 慢速行驶
            self.twist.angular.z = angular_z
            self.cmd_pub.publish(self.twist)

            # 检查是否已经调整到两条黑线中间
            if abs(deviation) < 5:  # 偏差小于5像素，认为已经居中
                self.lines_detected = True
                rospy.loginfo("车身已调整到两条黑线中间，准备直线行驶")

            # 检查是否可以退出坡道模式
            if self.lines_detected and rospy.get_time() - self.ramp_start_time >= 3:  # 保持居中3秒
                self.ramp_mode = False
                self.ramp_mode_completed = True  # 标记坡道模式已完成
                self.lines_detected = False
                rospy.loginfo("坡道模式结束，进入正常巡线模式")

        # 正常巡线模式
        if not self.ramp_mode and self.ramp_mode_completed:
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
