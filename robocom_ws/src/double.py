#!/usr/bin/env python
# coding=utf-8

import rospy
import cv2
import numpy as np
from geometry_msgs.msg import Twist

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

    search_top = int(h * 0.2)
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

            # ✅ ⬇️在这里插入调试可视化代码：
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
                # ✅ 保持原逻辑不动，取中线
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

        # 打开摄像头
        self.cap = cv2.VideoCapture(2)
        if not self.cap.isOpened():
            rospy.logerr("无法打开摄像头")
            rospy.signal_shutdown("无法打开摄像头")

        # 添加计时器
        self.start_time = rospy.get_time()
        self.straight_mode = False

    def process_frame(self, frame):
        h, w, _ = frame.shape

        if not self.initial_state_set:
            self.last_mid = w // 2
            self.kalman.statePost = np.array([[self.last_mid], [0]], np.float32)
            self.initial_state_set = True

        # 检查是否进入直线行驶模式
        current_time = rospy.get_time()
        if current_time - self.start_time >= 124:
            self.straight_mode = True

        if not self.straight_mode:
            deviation, center_points, self.failed_count = calculate_lane_direction(
                frame, self.kalman, self.last_mid, self.failed_count
            )
            self.last_mid = center_points[-1][0]

            # 动态调整 PID 参数
            deviation_threshold = 100  # 偏差阈值
            if abs(deviation) < deviation_threshold:
                self.pid.kp = 0.003  # 较小的 kp
                self.pid.kd = 0.0008  # 较小的 kd
            else:
                self.pid.kp = 0.035  # 较大的 kp
                self.pid.kd = 0.002  # 较大的 kd

            angular_z = self.pid.update(deviation)
        else:
            # 直线行驶模式
            angular_z = 0.0
            center_points = [(w // 2, h // 2)]  # 提供一个默认的中心点

        self.twist.linear.x = 0.2
        self.twist.angular.z = angular_z
        self.cmd_pub.publish(self.twist)

        for x, y in center_points:
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        cv2.imshow("Lane", frame)
        cv2.waitKey(3)

    def run(self):
        rate = rospy.Rate(20)  # 20 Hz
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                rospy.logerr("无法读取摄像头图像")
                break

            self.process_frame(frame)
            rate.sleep()

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        lane_follower = LaneFollower()
        lane_follower.run()
    except rospy.ROSInterruptException:
        pass
