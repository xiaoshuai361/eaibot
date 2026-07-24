#!/usr/bin/env python
# coding=utf-8

import rospy
from sensor_msgs.msg import Image
import cv2, cv_bridge
import numpy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import threading
import math
import numpy as np
from time import sleep

# Define constants
RAD2DEG = 180 / math.pi

class LaserAvoid():
    def __init__(self):
        
        # Create subscribers and publishers
        self.sub_laser = rospy.Subscriber(LaserScan, "/scan", self.registerScan, 1)
        self.pub_vel = rospy.Publisher(Twist, '/cmd_vel', 1)

        self.x_pid = PID(0.3, 0.0, 0.05, (-0.5, 0.5))
        self.z_pid = PID(0.08, 0.0, 0.05, (-1.0, 1.0))

        global follower

        global pid_distance
        global pid_angle
        global flag_detected

        # Declare parameters
        self.declare_parameter("linear", 0.3)
        self.declare_parameter("angular", 1.0)
        self.declare_parameter("LaserAngle", 40.0)
        self.declare_parameter("ResponseDist", 0.55)
        self.declare_parameter("Switch", False)

        self.linear = self.get_parameter('linear').get_parameter_value().double_value
        self.angular = self.get_parameter('angular').get_parameter_value().double_value
        self.LaserAngle = self.get_parameter('LaserAngle').get_parameter_value().double_value
        self.ResponseDist = self.get_parameter('ResponseDist').get_parameter_value().double_value
        self.Switch = self.get_parameter('Switch').get_parameter_value().bool_value

        # Initialize state
        self.Right_warning = 0
        self.Left_warning = 0
        self.front_warning = 0
        self.Joy_active = False
        self.Moving = False
        self.original_orientation = 0.0  # New variable for original orientation

        # Create a timer
        self.timer = self.create_timer(0.01, self.on_timer)

    def on_timer(self):
        self.Switch = self.get_parameter('Switch').get_parameter_value().bool_value
        self.angular = self.get_parameter('angular').get_parameter_value().double_value
        self.linear = self.get_parameter('linear').get_parameter_value().double_value
        self.LaserAngle = self.get_parameter('LaserAngle').get_parameter_value().double_value
        self.ResponseDist = self.get_parameter('ResponseDist').get_parameter_value().double_value

    def JoyStateCallback(self, msg):
        if not isinstance(msg, Bool): return
        self.Joy_active = msg.data

    def registerScan(self, scan_data):
        if not isinstance(scan_data, LaserScan): return


        ranges = np.array(scan_data.ranges)
        self.Right_warning = 0
        self.Left_warning = 0
        self.front_warning = 0

        if follower.switch_on_off == False:
            return

        for i in range(len(ranges)):
            angle = (scan_data.angle_min + scan_data.angle_increment * i) * RAD2DEG
            if 160 > angle > 180 - self.LaserAngle:
                if ranges[i] < self.ResponseDist * 1.5:
                    self.Right_warning += 1
            if -160 < angle < self.LaserAngle - 180:
                if ranges[i] < self.ResponseDist * 1.5:
                    self.Left_warning += 1
            if abs(angle) > 160:
                if ranges[i] <= self.ResponseDist * 1.5:
                    self.front_warning += 1

        if self.Joy_active or self.Switch:
            if self.Moving:
                self.pub_vel.publish(Twist())
                self.Moving = not self.Moving
            return

        twist = Twist()
        if self.front_warning > 10 and self.Left_warning > 10 and self.Right_warning > 10:
            print('1, there are obstacles in the left and right, turn right')
            twist.linear.x = self.linear
            twist.angular.z = -self.angular
            self.pub_vel.publish(twist)
            sleep(0.2)

        elif self.front_warning > 10 and self.Left_warning <= 10 and self.Right_warning > 10:
            print('2, there is an obstacle in the middle right, turn left')
            twist.linear.x = 0.0
            twist.angular.z = self.angular
            self.pub_vel.publish(twist)
            sleep(0.2)
            if self.Left_warning > 10 and self.Right_warning <= 10:
                twist.linear.x = 0.0
                twist.angular.z = -self.angular
                self.pub_vel.publish(twist)
                sleep(0.5)

        elif self.front_warning > 10 and self.Left_warning > 10 and self.Right_warning <= 10:
            print('4. There is an obstacle in the middle left, turn right')
            twist.linear.x = 0.0
            twist.angular.z = -self.angular
            self.pub_vel.publish(twist)
            sleep(0.2)
            if self.Left_warning <= 10 and self.Right_warning > 10:
                twist.linear.x = 0.0
                twist.angular.z = self.angular
                self.pub_vel.publish(twist)
                sleep(0.5)

        elif self.front_warning > 10 and self.Left_warning < 10 and self.Right_warning < 10:
            print('6, there is an obstacle in the middle, turn left')
            twist.linear.x = 0.0
            twist.angular.z = self.angular
            self.pub_vel.publish(twist)
            sleep(0.2)

        elif self.front_warning < 10 and self.Left_warning > 10 and self.Right_warning > 10:
            print('7. There are obstacles on the left and right, turn right')
            twist.linear.x = 0.0
            twist.angular.z = -self.angular
            self.pub_vel.publish(twist)
            sleep(0.4)

        elif self.front_warning < 10 and self.Left_warning > 10 and self.Right_warning <= 10:
            print('8, there is an obstacle on the left, turn right')
            twist.linear.x = 0.0
            twist.angular.z = -self.angular
            self.pub_vel.publish(twist)
            sleep(0.2)

        elif self.front_warning < 10 and self.Left_warning <= 10 and self.Right_warning > 10:
            print('9, there is an obstacle on the right, turn left')
            twist.linear.x = 0.0
            twist.angular.z = self.angular
            self.pub_vel.publish(twist)
            sleep(0.2)

        elif self.front_warning <= 10 and self.Left_warning <= 10 and self.Right_warning <= 10:
            twist.linear.x = self.linear
            twist.angular.z = 0.0
            self.pub_vel.publish(twist)

        # Restore original orientation
        if self.Moving:
            twist = Twist()
            twist.linear.x = self.linear
            twist.angular.z = 0.0
            self.pub_vel.publish(twist)
            self.Moving = False

class Follower:
    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.twist = Twist()

        self.switch_on_off = False

        #PID参数定义
        self.Kp = 0.02
        self.Ki = 0.0
        self.Kd = 0.0
        self.num = 0
        self.data = 0.0
        self.data1 = 0.0

        self.PIDOutput =0.0         #PID控制器输出

        self.Error = 0.0
        self.LastError = 0.0
        self.LastLastError = 0.0

    def image_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_black = numpy.array([0, 0, 0])
        upper_black = numpy.array([85, 85, 85])
        mask = cv2.inRange(hsv, lower_black, upper_black)
        masked = cv2.bitwise_and(image, image, mask=mask)

        # 在图像某处绘制一个指示，因为只考虑20行宽的图像，所以使用numpy切片将以外的空间区域清空
        h, w, d = image.shape
        search_top = 1*h/2
        search_bot = search_top + 15
        mask[0:search_top, 0:w] = 0
        mask[search_bot:h, 0:w] = 0
        # 计算mask图像的重心，即几何中心
        M = cv2.moments(mask)
        #print M
        if M['m00'] > 0:
            self.switch_on_off = False
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.circle(image, (cx, cy), 20, (0, 0, 255), -1)
            self.LastLastError = self.LastError
            self.LastError = self.Error
            #erro = cx - w/2
            self.Error = w/2 - cx
            #计算增量
            IncrementalValue = self.Kp*(self.Error - self.LastError) + self.Ki * self.Error +self.Kd *(self.Error -2*self.LastError +self.LastLastError)
            #计算输出
            self.PIDOutput += IncrementalValue
            # print(self.PIDOutput
            self.twist.linear.x = 0.28
            self.twist.angular.z = float(self.PIDOutput)/8
            #self.twist.angular.z = 0
            self.cmd_vel_pub.publish(self.twist)
        else:
            self.switch_on_off = True
        cv2.imshow("window", image)
        cv2.waitKey(3)


rospy.init_node("opencv")
global follower

laser_avoid = LaserAvoid()
follower = Follower()

thread1 = threading.Thread(target=rospy.spin, args=())
thread1.start()

