#!/usr/bin/env python
# coding=utf-8
"""Low-speed keyboard teleop for the real robot's ROS Melodic setup."""
from __future__ import print_function

import select
import sys
import termios
import time
import tty

import rospy
from geometry_msgs.msg import Twist


# ===== 可直接修改的速度参数 =====
DEFAULT_LINEAR_SPEED = 0.16   # 初始前进/后退速度，单位 m/s。
DEFAULT_ANGULAR_SPEED = 0.50  # 初始原地旋转速度，单位 rad/s。
SPEED_STEP = 1.10             # 每次按调速键变为原来的 1.10 倍或其倒数。
MIN_LINEAR_SPEED = 0.02       # 线速度调节下限。
MAX_LINEAR_SPEED = 0.60       # 线速度调节上限。
MIN_ANGULAR_SPEED = 0.05      # 角速度调节下限。
MAX_ANGULAR_SPEED = 1.50      # 角速度调节上限。


HELP = """
低速键盘控制（按住按键移动，松开后自动停车）
------------------------------------------------
W / I : 前进
S / , : 后退
A / J : 原地左转
D / L : 原地右转
空格 / K / X : 立即停车

Q / Z : 线速度和角速度同时增加/降低 10%
R / F : 只增加/降低线速度 10%
T / G : 只增加/降低角速度 10%
Ctrl-C : 退出并停车
"""


def read_key(timeout, settings):
    tty.setraw(sys.stdin.fileno())
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    key = sys.stdin.read(1) if ready else ""
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key.lower()


def command_for_key(key, linear_speed, angular_speed):
    if key in ("w", "i"):
        return linear_speed, 0.0
    if key in ("s", ","):
        return -linear_speed, 0.0
    if key in ("a", "j"):
        return 0.0, angular_speed
    if key in ("d", "l"):
        return 0.0, -angular_speed
    if key in (" ", "k", "x"):
        return 0.0, 0.0
    return None


def clamp(value, low, high):
    return max(low, min(high, value))


def adjust_speeds(key, linear_speed, angular_speed):
    linear_scale = angular_scale = 1.0
    if key == "q":
        linear_scale = angular_scale = SPEED_STEP
    elif key == "z":
        linear_scale = angular_scale = 1.0 / SPEED_STEP
    elif key == "r":
        linear_scale = SPEED_STEP
    elif key == "f":
        linear_scale = 1.0 / SPEED_STEP
    elif key == "t":
        angular_scale = SPEED_STEP
    elif key == "g":
        angular_scale = 1.0 / SPEED_STEP
    else:
        return None
    return (
        clamp(linear_speed * linear_scale,
              MIN_LINEAR_SPEED, MAX_LINEAR_SPEED),
        clamp(angular_speed * angular_scale,
              MIN_ANGULAR_SPEED, MAX_ANGULAR_SPEED),
    )


def make_twist(linear, angular):
    message = Twist()
    message.linear.x = float(linear)
    message.linear.y = message.linear.z = 0.0
    message.angular.x = message.angular.y = 0.0
    message.angular.z = float(angular)
    return message


def main():
    settings = termios.tcgetattr(sys.stdin)
    rospy.init_node("slow_keyboard_teleop")
    publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    linear_speed = clamp(
        float(rospy.get_param("~linear_speed", DEFAULT_LINEAR_SPEED)),
        MIN_LINEAR_SPEED, MAX_LINEAR_SPEED,
    )
    angular_speed = clamp(
        float(rospy.get_param("~angular_speed", DEFAULT_ANGULAR_SPEED)),
        MIN_ANGULAR_SPEED, MAX_ANGULAR_SPEED,
    )
    key_timeout = max(0.05, float(rospy.get_param("~key_timeout", 0.25)))
    publish_rate = max(5.0, float(rospy.get_param("~publish_rate", 20.0)))
    poll_timeout = 1.0 / publish_rate
    linear = angular = 0.0
    last_motion_key = 0.0

    print(HELP)
    print("线速度: %.3f m/s，角速度: %.3f rad/s" %
          (linear_speed, angular_speed))
    try:
        while not rospy.is_shutdown():
            key = read_key(poll_timeout, settings)
            if key == "\x03":
                break
            command = command_for_key(key, linear_speed, angular_speed)
            if command is not None:
                linear, angular = command
                last_motion_key = time.time() if command != (0.0, 0.0) else 0.0
            else:
                adjusted = adjust_speeds(key, linear_speed, angular_speed)
                if adjusted is not None:
                    linear_speed, angular_speed = adjusted
                    linear = angular = 0.0
                    last_motion_key = 0.0
                    print("\r线速度: %.3f m/s，角速度: %.3f rad/s" %
                          (linear_speed, angular_speed))
                elif key:
                    linear = angular = 0.0
                    last_motion_key = 0.0
                elif last_motion_key and time.time() - last_motion_key >= key_timeout:
                    linear = angular = 0.0
                    last_motion_key = 0.0
            publisher.publish(make_twist(linear, angular))
    finally:
        stop = make_twist(0.0, 0.0)
        for _ in range(3):
            publisher.publish(stop)
            rospy.sleep(0.03)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


if __name__ == "__main__":
    main()
