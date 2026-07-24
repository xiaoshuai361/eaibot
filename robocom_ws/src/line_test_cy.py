#!/usr/bin/env python
# coding=utf-8
"""
巡线测试程序（独立版）
=======================
从 code.py 中提取巡线部分，去掉所有视觉识别任务。
运行后机器人立即开始巡线，Ctrl+C 退出。

摄像头索引：
  LANE_CAM_INDEX = 0   ← 朝下拍地面黑线的摄像头，按实际情况修改
"""

import rospy
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import time

# ===================== 参数（在这里修改）=====================
LANE_CAM_INDEX  = 0      # 巡线摄像头索引
LINEAR_SPEED    = 0.2    # 直线速度 m/s
PID_KP_SMALL    = 0.0035 # 偏差小时的 Kp（直道防抖）
PID_KP_LARGE    = 0.035  # 偏差大时的 Kp（弯道快速纠正）
PID_KI          = 0.0001
PID_KD_SMALL    = 0.0008
PID_KD_LARGE    = 0.002
DEV_THRESHOLD   = 100    # 偏差阈值（像素），大于此值用大 PID 参数
KALMAN_FAIL_MAX = 5      # 连续失败多少帧后重置卡尔曼
# =============================================================


class PIDController:
    def __init__(self, kp, ki, kd, output_limits=(-1.0, 1.0)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.last_error = 0.0
        self.integral   = 0.0
        self.last_time  = None

    def reset(self):
        self.integral   = 0.0
        self.last_error = 0.0
        self.last_time  = None

    def update(self, measured_value):
        now = rospy.get_time()
        dt  = 0.1 if self.last_time is None else max(now - self.last_time, 0.001)
        error          = -measured_value          # setpoint = 0，偏差直接取负
        self.integral += error * dt
        derivative     = (error - self.last_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = max(self.output_limits[0], min(self.output_limits[1], output))
        self.last_error = error
        self.last_time  = now
        return output


def calculate_lane_direction(image, kalman_filter, last_mid_point_x, failed_count):
    """
    从图像中检测黑色引导线并计算中心偏差。
    返回：(deviation, mid_point_x, failed_count, debug_info)
      deviation    : 中心点相对图像中线的像素偏差（正=偏右，负=偏左）
      mid_point_x  : 卡尔曼平滑后的中心 x 坐标
      failed_count : 更新后的连续失败帧计数
      debug_info   : 字典，包含可视化所需信息
    """
    h, w = image.shape[:2]
    
    # ── 黑色线提取：HSV 暗色过滤 + 自适应阈值，取交集 ──
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
    )
    binary = cv2.bitwise_and(adaptive, color_mask)

    # ── 只在帧高度 32%~50% 的水平带内搜索（去掉近距离车身和远处干扰）──
    search_top = int(h * 0.32)
    search_bot = int(h * 0.50)
    binary[:search_top, :] = 0
    binary[search_bot:,  :] = 0

    mid_point_x  = last_mid_point_x
    current_mid_raw = -1   # -1 表示本帧未检测到
    detect_y     = -1      # 实际检测到的行号
    left_x       = -1      # 左边线 x
    right_x      = -1      # 右边线 x

    for y in range(search_bot, search_top - 1, -1):
        row_pixels = np.where(binary[y, :] == 255)[0]
        if len(row_pixels) <= 2:
            continue

        # 将连续像素聚成组（间距 < 20px 视为同一组）
        pixel_groups, current_group = [], [row_pixels[0]]
        for i in range(1, len(row_pixels)):
            if row_pixels[i] - row_pixels[i - 1] < 20:
                current_group.append(row_pixels[i])
            else:
                if len(current_group) > 2:
                    pixel_groups.append(current_group)
                current_group = [row_pixels[i]]
        if len(current_group) > 2:
            pixel_groups.append(current_group)

        # 过滤水平线（斜率过大的组）
        valid_groups = [
            g for g in pixel_groups
            if (g[-1] - g[0]) / max(search_bot - y, 1) < 1.5
        ]
        
        if len(valid_groups) >= 3:
            # 三条以上：跟随最右侧线（应对交叉口/斑马线）
            current_mid_raw = valid_groups[-1][-1] - 10
            detect_y = y
            right_x  = valid_groups[-1][-1]
            left_x   = valid_groups[0][0]
            break
        elif len(valid_groups) == 2:
            left_edge  = valid_groups[0][-1]
            right_edge = valid_groups[1][0]
            detect_y = y
            left_x   = valid_groups[0][0]
            right_x  = valid_groups[1][-1]
            if abs(right_edge - left_edge) >= 300:
                current_mid_raw = (left_edge + right_edge) // 2
            else:
                mid = (valid_groups[0][0] + valid_groups[0][-1]) / 2
                current_mid_raw = (valid_groups[0][-1] + w) // 2 if mid < w / 2 else valid_groups[0][0] // 2
            break
        elif len(valid_groups) == 1:
            mid = (valid_groups[0][0] + valid_groups[0][-1]) / 2
            detect_y = y
            left_x   = valid_groups[0][0]
            right_x  = valid_groups[0][-1]
            current_mid_raw = (valid_groups[0][-1] + w) // 2 if mid < w / 2 else valid_groups[0][0] // 2
            break

    # ── 卡尔曼滤波平滑 ──
    predicted = kalman_filter.predict()

    if current_mid_raw != -1 and abs(current_mid_raw - mid_point_x) < w * 0.4:
        measurement = np.array([[np.float32(current_mid_raw)]])
        mid_point_x = int(kalman_filter.correct(measurement)[0])
        failed_count = 0
    else:
        mid_point_x = int(predicted[0])
        failed_count += 1
        if failed_count >= KALMAN_FAIL_MAX:
            kalman_filter.statePost = np.array([[w // 2], [0]], np.float32)
            mid_point_x  = w // 2
            failed_count = 0

    deviation  = mid_point_x - (w / 2)
    debug_info = {
        "binary":      binary,
        "search_top":  search_top,
        "search_bot":  search_bot,
        "detect_y":    detect_y,
        "left_x":      left_x,
        "right_x":     right_x,
    }
    return deviation, mid_point_x, failed_count, debug_info


def main():
    rospy.init_node("lane_test", anonymous=True)
    cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    twist   = Twist()
    rate    = rospy.Rate(20)   # 20 Hz

    # ── 摄像头 ──
    cap = cv2.VideoCapture(LANE_CAM_INDEX)
    if not cap.isOpened():
        rospy.logerr("无法打开摄像头（索引 %d）", LANE_CAM_INDEX)
        return
    rospy.loginfo("摄像头打开成功，开始巡线...")

    # ── 卡尔曼滤波器 ──
    kalman = cv2.KalmanFilter(2, 1)
    kalman.transitionMatrix    = np.array([[1, 1], [0, 1]], np.float32)
    kalman.measurementMatrix   = np.array([[1, 0]], np.float32)
    kalman.processNoiseCov     = np.eye(2, dtype=np.float32) * 1e-4
    kalman.measurementNoiseCov = np.array([[1]], np.float32) * 1e-1

    # ── PID ──
    pid = PIDController(PID_KP_SMALL, PID_KI, PID_KD_SMALL)

    last_mid      = 320
    failed_count  = 0
    initial_set   = False

    try:
        while not rospy.is_shutdown():
            ret, frame = cap.read()
            if not ret:
                rospy.logwarn("摄像头读取失败，跳过本帧")
                rate.sleep()
                continue

            w = frame.shape[1]

            # 首帧初始化卡尔曼状态
            if not initial_set:
                last_mid = w // 2
                kalman.statePost = np.array([[last_mid], [0]], np.float32)
                initial_set = True

            deviation, last_mid, failed_count, dbg = calculate_lane_direction(
                frame, kalman, last_mid, failed_count
            )

            # 动态 PID 参数：大偏差用大增益快速纠偏，小偏差用小增益防抖
            if abs(deviation) < DEV_THRESHOLD:
                pid.kp = PID_KP_SMALL
                pid.kd = PID_KD_SMALL
            else:
                pid.kp = PID_KP_LARGE
                pid.kd = PID_KD_LARGE

            angular_z = pid.update(deviation)
            twist.linear.x   = LINEAR_SPEED
            twist.angular.z  = angular_z
            cmd_pub.publish(twist)

            # ── 调试窗口 ──
            h_f = frame.shape[0]
            # 1. 搜索区域（黄色矩形框）
            cv2.rectangle(frame,
                          (0, dbg["search_top"]), (w - 1, dbg["search_bot"]),
                          (0, 255, 255), 1)
            # 2. 检测到的左/右边线端点（蓝色圆点）
            if dbg["detect_y"] != -1:
                dy = dbg["detect_y"]
                if dbg["left_x"] != -1:
                    cv2.circle(frame, (dbg["left_x"],  dy), 5, (255, 100, 0), -1)
                if dbg["right_x"] != -1:
                    cv2.circle(frame, (dbg["right_x"], dy), 5, (255, 100, 0), -1)
                # 3. 中心点画在 **实际检测行**（绿色，大圆）
                cv2.circle(frame, (last_mid, dy), 8, (0, 255, 0), -1)
                # 4. 中心线（绿色竖线，从检测行到底部）
                cv2.line(frame, (last_mid, dy), (last_mid, h_f - 1), (0, 255, 0), 1)
            else:
                # 未检测到时用灰色点（卡尔曼预测值）
                cv2.circle(frame, (last_mid, dbg["search_bot"]), 8, (128, 128, 128), -1)
            # 5. 图像正中参考线（红色虚线用矩形近似）
            cv2.line(frame, (w // 2, dbg["search_top"]),
                            (w // 2, dbg["search_bot"]), (0, 0, 255), 1)
            # cv2.putText(frame, f"dev={deviation:.1f}  ang={angular_z:.3f}",
            #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Lane Test", frame)
            # 6. 二值化结果单独窗口（查看算法实际看到什么）
            cv2.imshow("Binary", dbg["binary"])
            if cv2.waitKey(3) & 0xFF == ord('q'):
                rospy.loginfo("按下 q，退出...")
                break

            rate.sleep()

    except rospy.ROSInterruptException:
        pass
    finally:
        # 停车
        twist.linear.x  = 0.0
        twist.angular.z = 0.0
        cmd_pub.publish(twist)
        rospy.loginfo("已停车")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
