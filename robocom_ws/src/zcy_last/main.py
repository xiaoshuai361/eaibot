#!/usr/bin/env python3
# coding=utf-8
"""九路口比赛任务唯一启动入口。"""

import argparse
import sys

import rospy

from . import config
from .control.grasp import GraspCoordinator
from .control.processes import ProcessSupervisor
from .task.competition import LaneFollower


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="九路口循迹与机械臂比赛任务")
    tag = parser.add_mutually_exclusive_group()
    tag.add_argument("--tag-pick", dest="tag_pick", action="store_true")
    tag.add_argument("--no-tag-pick", dest="tag_pick", action="store_false")
    untagged = parser.add_mutually_exclusive_group()
    untagged.add_argument(
        "--untagged-pick", dest="untagged_pick", action="store_true")
    untagged.add_argument(
        "--no-untagged-pick", dest="untagged_pick", action="store_false")
    parser.set_defaults(tag_pick=None, untagged_pick=None)
    parser.add_argument("--tag-pick-count", type=int,
                        default=config.TAG_PICK_COUNT)
    parser.add_argument("--untagged-pick-count", type=int,
                        default=config.UNTAGGED_PICK_COUNT)
    parser.add_argument(
        "--external-ros", action="store_true",
        help="不启动或关闭 ROS 依赖，仅用于已经手动启动依赖的调试环境。")
    options, ros_args = parser.parse_known_args(argv)
    if options.tag_pick is None:
        options.tag_pick = config.ENABLE_TAG_PICK
    if options.untagged_pick is None:
        options.untagged_pick = config.ENABLE_UNTAGGED_PICK
    for name, value in (
            ("--tag-pick-count", options.tag_pick_count),
            ("--untagged-pick-count", options.untagged_pick_count)):
        if not 1 <= int(value) <= len(config.PICK_CANDIDATE_IDS):
            parser.error("%s 必须在 1 到 4 之间" % name)
    return options, ros_args


def main(argv=None):
    options, ros_args = parse_args(argv)
    sys.argv = [sys.argv[0]] + list(ros_args)
    supervisor = ProcessSupervisor(
        enabled=config.MANAGE_ROS_PROCESSES and not options.external_ros,
        python3=sys.executable,
    )
    try:
        supervisor.start_base()
        if options.tag_pick or options.untagged_pick:
            supervisor.start_arm_common()
        if options.tag_pick:
            supervisor.start_astra()
            supervisor.start_tag_stack()
        coordinator = GraspCoordinator(
            supervisor,
            keep_arm_after_tag=options.untagged_pick,
            python3=sys.executable,
        )
        follower = LaneFollower(
            grasp_coordinator=coordinator,
            process_supervisor=supervisor,
            enable_tag_pick=options.tag_pick,
            tag_pick_count=options.tag_pick_count,
            enable_untagged_pick=options.untagged_pick,
            untagged_pick_count=options.untagged_pick_count,
        )
        follower.run()
    finally:
        supervisor.shutdown()


if __name__ == "__main__":
    try:
        main()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        pass
