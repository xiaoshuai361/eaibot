#!/usr/bin/env python3
# coding=utf-8
"""九路口比赛任务唯一启动入口。"""

import rospy

from .task.competition import LaneFollower


def main():
    follower = LaneFollower()
    follower.run()


if __name__ == "__main__":
    try:
        main()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        pass
