#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Guide per-target RGB distance collection and fit monocular models."""

from __future__ import print_function

import argparse
import os
import subprocess
import sys


TARGETS = ("power", "fire", "gas", "support")
DEFAULT_DISTANCES = (
    280, 300, 320, 340, 360, 380,
    400, 420, 440, 460, 480,
)
DEFAULT_CONFIG = "/home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml"
DEFAULT_OUTPUT_DIR = "/home/eaibot/handeye-calib/config/block_distance_samples"


def parse_targets(value):
    targets = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in targets if item not in TARGETS]
    if not targets or unknown:
        raise argparse.ArgumentTypeError(
            "targets must be a comma-separated subset of %s" % (TARGETS,))
    return targets


def parse_distances(value):
    try:
        distances = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError("distances must be comma-separated integers")
    if len(set(distances)) < 3 or any(item <= 0 for item in distances):
        raise argparse.ArgumentTypeError(
            "provide at least three different positive distances")
    return distances


def parse_args(argv=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Guided monocular distance collection for tagless blocks")
    parser.add_argument("--targets", type=parse_targets,
                        default=list(TARGETS))
    parser.add_argument("--distances", type=parse_distances,
                        default=list(DEFAULT_DISTANCES))
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--block-pick-main",
                        default=os.path.join(script_dir, "block_pick_main.py"))
    parser.add_argument("--calibrator",
                        default=os.path.join(script_dir, "block_distance_calibrate.py"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.frames <= 0:
        parser.error("--frames must be positive")
    if not 0.0 < args.confidence <= 1.0:
        parser.error("--confidence must be in (0, 1]")
    return args


def sample_path(output_dir, target, distance_mm):
    return os.path.join(output_dir, "%s_%d.csv" % (target, distance_mm))


def sample_schedule(targets, distances):
    return [
        (target, distance_mm)
        for distance_mm in distances
        for target in targets
    ]


def build_collect_command(args, target, distance_mm):
    return [
        sys.executable,
        args.block_pick_main,
        "--target", target,
        "--calib-record",
        "--known-z-mm", str(distance_mm),
        "--frames", str(args.frames),
        "--confidence", str(args.confidence),
        "--config", args.config,
    ]


def run_and_tee(command, output_path):
    temporary_path = output_path + ".tmp"
    process = None
    try:
        output = open(temporary_path, "w")
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, bufsize=1)
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                output.write(line)
        finally:
            output.close()
        return_code = process.wait()
        if return_code == 0:
            os.replace(temporary_path, output_path)
        elif os.path.exists(temporary_path):
            os.unlink(temporary_path)
        return return_code
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def prompt_sample(target, distance_mm, frames, index, total):
    print("\n[%d/%d] 物块=%s，真实相机距离=%dmm" % (
        index, total, target, distance_mm))
    print("本次要求采集 %d 张不同时间戳的 YOLO 图像。" % frames)
    print("只摆放当前类别，把框中心放进实时预览的红色 ROI，并保持物块正面平行。")
    print("准确测量 RGB 镜头到物块正面的距离。按 Enter 开始；s 跳过；q 退出。")
    return input("> ").strip().lower()


def fit_target(args, target, paths):
    model_path = os.path.join(args.output_dir, "%s_model.yaml" % target)
    command = [sys.executable, args.calibrator, "--target", target] + paths
    return_code = run_and_tee(command, model_path)
    if return_code != 0:
        raise RuntimeError("distance fitting failed for %s" % target)
    print("已保存拟合结果：%s" % model_path)


def main(argv=None):
    args = parse_args(argv)
    print("采集配置：每个类别/距离 %d 帧，置信度 %.3f" % (
        args.frames, args.confidence))
    print("采集入口：%s" % os.path.abspath(args.block_pick_main))
    print("配置文件：%s" % os.path.abspath(args.config))
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    schedule = sample_schedule(args.targets, args.distances)
    existing_count = sum(
        os.path.isfile(sample_path(args.output_dir, target, distance_mm))
        for target, distance_mm in schedule
    )
    if args.overwrite:
        print("覆盖模式：将重新采集计划内已有的 %d 个样本文件。" % existing_count)
    else:
        print("续采模式：保留并跳过 %d 个已有样本，只采集缺失距离。" % existing_count)
    paths_by_target = dict((target, []) for target in args.targets)
    previous_distance = None
    for index, (target, distance_mm) in enumerate(schedule, 1):
        if distance_mm != previous_distance:
            print("\n=== 当前真实相机距离：%dmm ===" % distance_mm)
            previous_distance = distance_mm
        path = sample_path(args.output_dir, target, distance_mm)
        if os.path.isfile(path) and not args.overwrite:
            print("跳过已有样本：%s" % path)
            paths_by_target[target].append(path)
            continue
        choice = prompt_sample(
            target, distance_mm, args.frames, index, len(schedule))
        if choice in ("q", "quit", "exit"):
            print("采集已停止；再次运行会自动跳过已有文件并继续。")
            return 0
        if choice in ("s", "skip"):
            continue
        command = build_collect_command(args, target, distance_mm)
        if run_and_tee(command, path) != 0:
            raise RuntimeError(
                "collection failed for %s at %dmm" % (target, distance_mm))
        paths_by_target[target].append(path)

    for target in args.targets:
        paths = paths_by_target[target]
        if len(paths) >= 3:
            fit_target(args, target, paths)
    print("\n全部采集完成。把各个 *_model.yaml 参数填入 block_mono_grasp.yaml。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n采集已中断；再次运行可继续。")
        sys.exit(1)
    except Exception as exc:
        print("错误：%s" % exc)
        sys.exit(1)
