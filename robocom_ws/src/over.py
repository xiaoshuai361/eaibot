#!/usr/bin/env python
# coding=utf-8

import rospy
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
import cv2
import cv_bridge
import numpy as np
import threading
import math
from time import sleep

# Define constants
RAD2DEG = 180 / math.pi


class LaserAvoid():
    def __init__(self):
        # Create subscribers and publishers
        self.sub_laser = rospy.Subscriber("/scan", LaserScan, self.registerScan)
        self.pub_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # PID controllers for x and z velocities
        self.x_pid = PID(0.3, 0.0, 0.05, (-0.5, 0.5))
        self.z_pid = PID(0.08, 0.0, 0.05, (-1.0, 1.0))

        global follower

        # Declare parameters (default values)
        self.linear = rospy.get_param("~linear", 0.3)
        self.angular = rospy.get_param("~angular", 1.0)
        self.LaserAngle = rospy.get_param("~LaserAngle", 40.0)
        self.ResponseDist = rospy.get_param("~ResponseDist", 0.55)
        self.Switch = rospy.get_param("~Switch", False)

        # Initialize state
        self.Right_warning = 0
        self.Left_warning = 0
        self.front_warning = 0
        self.Joy_active = False
        self.Moving = False

        # Create a timer
        self.timer = rospy.Timer(rospy.Duration(0.01), self.on_timer)

    def on_timer(self, event):
        # Update parameters from ROS parameter server
        self.linear = rospy.get_param("~linear", self.linear)
        self.angular = rospy.get_param("~angular", self.angular)
        self.LaserAngle = rospy.get_param("~LaserAngle", self.LaserAngle)
        self.ResponseDist = rospy.get_param("~ResponseDist", self.ResponseDist)
        self.Switch = rospy.get_param("~Switch", self.Switch)

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

        # Restore original orientation if moving
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

        # PID parameters
        self.Kp = 0.02
        self.Ki = 0.0
        self.Kd = 0.0
        self.PIDOutput = 0.0

        self.Error = 0.0
        self.LastError = 0.0
        self.LastLastError = 0.0

    def image_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([85, 85, 85])
        mask = cv2.inRange(hsv, lower_black, upper_black)
        masked = cv2.bitwise_and(image, image, mask=mask)

        # Define search area
        h, w, d = image.shape
        search_top = int(h / 2)
        search_bot = search_top + 15
        mask[0:search_top, 0:w] = 0
        mask[search_bot:h, 0:w] = 0

        # Calculate the center of the mask
        M = cv2.moments(mask)
        if M['m00'] > 0:
            self.switch_on_off = False
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(image, (cx, cy), 20, (0, 0, 255), -1)
            self.LastLastError = self.LastError
            self.LastError = self.Error
            self.Error = w / 2 - cx

            # Incremental PID control
            IncrementalValue = self.Kp * (self.Error - self.LastError) + self.Ki * self.Error + self.Kd * (self.Error - 2 * self.LastError + self.LastLastError)
            self.PIDOutput += IncrementalValue

            self.twist.linear.x = 0.28
            self.twist.angular.z = float(self.PIDOutput) / 8
            self.cmd_vel_pub.publish(self.twist)
        else:
            # Turn on laser scan avoidance if the line is lost
            print('Cannot find line')
            sleep(1.5)
            self.switch_on_off = True

        ##cv2.imshow("window", image)
        ##cv2.waitKey(3)

class PID:
    def __init__(self, kp, ki, kd, output_limits=(None, None)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.integral = 0
        self.previous_error = 0

    def compute(self, error, dt):
        # PID calculation
        p = error
        self.integral += error * dt
        d = (error - self.previous_error) / dt if dt > 0 else 0.0

        output = self.kp * p + self.ki * self.integral + self.kd * d

        # Apply output limits
        if self.output_limits:
            output = max(min(output, self.output_limits[1]), self.output_limits[0])

        self.previous_error = error
        return output

def main():
    rospy.init_node('follower')
    global follower
    follower = Follower()
    laser_avoid = LaserAvoid()
    rospy.spin()

if __name__ == '__main__':
    main()
