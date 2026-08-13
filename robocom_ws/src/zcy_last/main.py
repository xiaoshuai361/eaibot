#!/usr/bin/env python3
# coding=utf-8
"""九路口比赛任务入口；底盘可由外部启动或由 launch.py 托管。"""

import argparse
import inspect
import sys

import rospy

from . import config
from .control.grasp import GraspCoordinator
from .control.processes import ProcessSupervisor
from .task.competition import LaneFollower


def validate_runtime_interfaces():
    """在启动相机和机械臂前检查 zcy_last 是否混入旧模块。"""
    required = {
        "grasp_coordinator",
        "process_supervisor",
        "enable_tag_pick",
        "tag_pick_count",
        "enable_tag_delivery",
        "enable_untagged_pick",
        "untagged_pick_count",
        "enable_untagged_delivery",
    }
    signature = inspect.signature(LaneFollower.__init__)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD
           for parameter in signature.parameters.values()):
        return
    parameters = set(signature.parameters)
    missing = sorted(required - parameters)
    if missing:
        raise RuntimeError(
            "zcy_last 模块版本不一致："
            "task/competition.py 缺少参数 %s。"
            "请完整同步 /home/eaibot/robocom_ws/src/zcy_last/，"
            "不要只替换单个文件。" % ", ".join(missing))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="九路口循迹与机械臂比赛任务")
    tag = parser.add_mutually_exclusive_group()
    tag.add_argument("--tag-pick", dest="tag_pick", action="store_true")
    tag.add_argument("--no-tag-pick", dest="tag_pick", action="store_false")
    delivery = parser.add_mutually_exclusive_group()
    delivery.add_argument(
        "--tag-delivery", dest="tag_delivery", action="store_true")
    delivery.add_argument(
        "--no-tag-delivery", dest="tag_delivery", action="store_false")
    untagged = parser.add_mutually_exclusive_group()
    untagged.add_argument(
        "--untagged-pick", dest="untagged_pick", action="store_true")
    untagged.add_argument(
        "--no-untagged-pick", dest="untagged_pick", action="store_false")
    untagged_delivery = parser.add_mutually_exclusive_group()
    untagged_delivery.add_argument(
        "--untagged-delivery", dest="untagged_delivery",
        action="store_true")
    untagged_delivery.add_argument(
        "--no-untagged-delivery", dest="untagged_delivery",
        action="store_false")
    parser.set_defaults(
        tag_pick=None, tag_delivery=None, untagged_pick=None,
        untagged_delivery=None)
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
    if options.tag_delivery is None:
        options.tag_delivery = config.ENABLE_TAG_DELIVERY
    if options.untagged_delivery is None:
        options.untagged_delivery = config.ENABLE_UNTAGGED_DELIVERY
    options.tag_delivery = bool(options.tag_pick and options.tag_delivery)
    options.untagged_delivery = bool(
        options.untagged_pick and options.untagged_delivery)
    for name, value in (
            ("--tag-pick-count", options.tag_pick_count),
            ("--untagged-pick-count", options.untagged_pick_count)):
        if not 1 <= int(value) <= len(config.PICK_CANDIDATE_IDS):
            parser.error("%s 必须在 1 到 4 之间" % name)
    return options, ros_args


def run_competition(options, ros_args, supervisor):
    """复用 launch.py 的常驻依赖并运行比赛状态机。"""
    sys.argv = [sys.argv[0]] + list(ros_args)
    print(
        "[zcy_last] 任务配置：Tag抓取=%s 数量=%d Tag投递=%s "
        "无Tag抓取=%s 数量=%d 无Tag投递=%s" % (
            options.tag_pick, options.tag_pick_count,
            options.tag_delivery, options.untagged_pick,
            options.untagged_pick_count, options.untagged_delivery,
        ),
        flush=True,
    )
    if options.tag_pick:
        supervisor.start_astra()
        supervisor.start_tag_stack()

    coordinator = GraspCoordinator(
        supervisor,
        keep_arm_after_tag=(
            options.tag_delivery or options.untagged_pick),
        keep_arm_after_untagged=options.untagged_delivery,
        python3=sys.executable,
    )
    follower = LaneFollower(
        grasp_coordinator=coordinator,
        process_supervisor=supervisor,
        enable_tag_pick=options.tag_pick,
        tag_pick_count=options.tag_pick_count,
        enable_tag_delivery=options.tag_delivery,
        enable_untagged_pick=options.untagged_pick,
        untagged_pick_count=options.untagged_pick_count,
        enable_untagged_delivery=options.untagged_delivery,
    )
    follower.run()


def main(argv=None):
    """任务入口：复用 launch.py 常驻依赖，按任务开关托管阶段资源。"""
    options, ros_args = parse_args(argv)
    validate_runtime_interfaces()
    supervisor = ProcessSupervisor(
        enabled=config.MANAGE_ROS_PROCESSES and not options.external_ros,
        python3=sys.executable,
    )
    try:
        run_competition(options, ros_args, supervisor)
    finally:
        supervisor.shutdown()


if __name__ == "__main__":
    try:
        main()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        pass
