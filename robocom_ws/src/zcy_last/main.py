#!/usr/bin/env python3
# coding=utf-8
"""九路口比赛任务入口；底盘可由外部启动或由 launch.py 托管。"""

import argparse
import inspect
import os
import sys
import time

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
        "start_untagged_aligned",
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
    try:
        untagged_command_source = inspect.getsource(
            GraspCoordinator._untagged_command)
    except (AttributeError, IOError, OSError, TypeError) as exc:
        raise RuntimeError(
            "zcy_last 模块版本不一致：无法检查无 Tag 抓取命令：%s；"
            "请完整同步 /home/eaibot/robocom_ws/src/zcy_last/。" % exc)
    if ("--allow-partial" not in untagged_command_source
            or "--fail-on-skip" in untagged_command_source):
        raise RuntimeError(
            "zcy_last 模块版本不一致：control/grasp.py 仍在使用严格"
            "无 Tag 抓取模式。请完整同步 "
            "/home/eaibot/robocom_ws/src/zcy_last/，确认命令包含 "
            "--allow-partial 且不包含 --fail-on-skip。")


def validate_external_pick_scripts():
    """真机运动前拒绝混用旧版无 Tag 父/子抓取脚本。"""
    block_script = config.UNTAGGED_PICK_SCRIPT
    arm_script = os.path.join(
        os.path.dirname(block_script), "mirobot_pick_test.py")
    requirements = (
        (block_script, "block_pick_main.py", (
            "--allow-partial", "--search-enable-file")),
        (arm_script, "mirobot_pick_test.py", (
            "allow_partial = bool", "current_pose_reached_target",
            "wait_for_joint_state_stable")),
    )
    for path, label, markers in requirements:
        # Windows 开发机没有 /home/eaibot；真机文件存在时执行严格检查。
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                source = handle.read()
        except (IOError, OSError) as exc:
            raise RuntimeError("无法检查 %s：%s" % (path, exc))
        missing = [marker for marker in markers if marker not in source]
        if missing:
            raise RuntimeError(
                "无 Tag 抓取脚本版本不一致：%s 缺少 %s。请同步 Windows "
                "仓库中的 handeye-calib/src/%s 到 %s。" % (
                    label, ", ".join(missing), label, path))


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
        "--start-untagged-aligned", action="store_true",
        help="调试入口：车辆已在第3个右转出口摆正，直接从A点无Tag抓取开始。")
    parser.add_argument(
        "--external-ros", action="store_true",
        help="不启动或关闭 ROS 依赖，仅用于已经手动启动依赖的调试环境。")
    options, ros_args = parser.parse_known_args(argv)
    if options.start_untagged_aligned:
        if options.tag_pick is True:
            parser.error("--start-untagged-aligned 不能同时启用 --tag-pick")
        if options.untagged_pick is False:
            parser.error(
                "--start-untagged-aligned 不能同时使用 --no-untagged-pick")
        options.tag_pick = False
        options.untagged_pick = True
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
        "无Tag抓取=%s 数量=%d 无Tag投递=%s 调试起点=%s" % (
            options.tag_pick, options.tag_pick_count,
            options.tag_delivery, options.untagged_pick,
            options.untagged_pick_count, options.untagged_delivery,
            "第3个右转后已摆正" if options.start_untagged_aligned else "正常起点",
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
        start_untagged_aligned=options.start_untagged_aligned,
    )
    try:
        follower.run()
    except rospy.ROSInterruptException:
        # 正常跑完时状态机会先进入 DONE，再调用 rospy.signal_shutdown；
        # 随后的 Rate.sleep 也可能抛 ROSInterruptException，不能误报为中断。
        if getattr(follower, "state", None) != "DONE":
            raise
    return getattr(follower, "state", None) == "DONE"


def main(argv=None):
    """任务入口：复用 launch.py 常驻依赖，按任务开关托管阶段资源。"""
    options, ros_args = parse_args(argv)
    validate_runtime_interfaces()
    if options.untagged_pick:
        validate_external_pick_scripts()
    supervisor = ProcessSupervisor(
        enabled=config.MANAGE_ROS_PROCESSES and not options.external_ros,
        python3=sys.executable,
    )
    try:
        if options.tag_pick or options.untagged_pick:
            supervisor.start_arm_common()
        return run_competition(options, ros_args, supervisor)
    finally:
        supervisor.shutdown()


def format_elapsed_time(elapsed_seconds):
    """把单调时钟耗时格式化为比赛终端使用的分秒。"""
    total_seconds = max(0, int(float(elapsed_seconds)))
    minutes, seconds = divmod(total_seconds, 60)
    return "%d分%02d秒" % (minutes, seconds)


def run_timed(argv=None, clock=None):
    """运行比赛，并且只在正常完成或被中断时打印一次总耗时。"""
    clock = time.monotonic if clock is None else clock
    started_at = clock()
    try:
        completed = bool(main(argv))
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        print(
            "[zcy_last] 任务已中断，已运行：%s" %
            format_elapsed_time(clock() - started_at),
            flush=True,
        )
        return False

    elapsed_text = format_elapsed_time(clock() - started_at)
    if completed:
        print(
            "[zcy_last] 全部任务完成，总耗时：%s" % elapsed_text,
            flush=True,
        )
    else:
        # rospy 收到 Ctrl+C 时通常只设置 shutdown，不一定抛异常；
        # 只要路线还没进入 DONE，就按中断显示。
        print(
            "[zcy_last] 任务已中断，已运行：%s" % elapsed_text,
            flush=True,
        )
    return completed


if __name__ == "__main__":
    run_timed()
