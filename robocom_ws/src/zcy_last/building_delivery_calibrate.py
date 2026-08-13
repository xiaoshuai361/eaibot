#!/usr/bin/env python3
# coding=utf-8
"""Collect real-distance building boxes and fit per-class monocular models."""

import argparse
import csv
import os
import select
import sys
import time

import cv2

from .algorithms.building_delivery import (
    build_distance_calibration_entry,
    building_box_geometry,
    empty_building_calibration,
    load_building_calibration,
    save_building_calibration,
    select_locked_building_detection,
)
from .algorithms.vision import (
    YoloObstacleDetector,
    detection_center_in_x_roi,
    draw_yolo_boxes,
)
from .config import (
    BUILDING_DELIVERY_CALIBRATION_FILE,
    BUILDING_DELIVERY_REFERENCE_DISTANCE_MM,
    UNTAGGED_DELIVERY_ID_BY_BUILDING_CLASS,
    YOLO_BUILDING_CLASS_NAMES,
    YOLO_BUILDING_CONFIDENCE,
    YOLO_BUILDING_CENTER_ROI_X_RATIO,
    YOLO_BUILDING_MODEL_PATH,
    YOLO_CAMERA_INDEX,
    YOLO_FRAME_HEIGHT,
    YOLO_FRAME_WIDTH,
    YOLO_IMAGE_SIZE,
    YOLO_NMS_THRESHOLD,
)


DEFAULT_DISTANCES = (
    350,
    400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500,
    550, 600,
)
DEFAULT_SAMPLE_DIR = (
    "/home/eaibot/handeye-calib/config/"
    "building_delivery_distance_samples_building_new_320x240")
ID_TO_BUILDING_CLASS = dict(
    (item_id, class_name)
    for class_name, item_id in UNTAGGED_DELIVERY_ID_BY_BUILDING_CLASS.items())
BUILDING_NAME_BY_ID = {
    1: "电力故障楼宇",
    2: "火灾楼宇",
    3: "有毒气体楼宇",
    4: "坍塌楼宇",
}


def parse_distances(value):
    try:
        distances = [int(item.strip()) for item in value.split(",")
                     if item.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError("距离必须是逗号分隔的整数毫米值")
    if len(set(distances)) < 3 or any(item <= 0 for item in distances):
        raise argparse.ArgumentTypeError("至少提供3个不同的正距离")
    return distances


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="采集楼宇真实距离与YOLO框，拟合单目距离模型")
    parser.add_argument("--distances", type=parse_distances,
                        default=list(DEFAULT_DISTANCES))
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--reference-distance-mm", type=float,
                        default=BUILDING_DELIVERY_REFERENCE_DISTANCE_MM)
    parser.add_argument("--camera-index", type=int, default=YOLO_CAMERA_INDEX)
    parser.add_argument("--output", default=BUILDING_DELIVERY_CALIBRATION_FILE)
    parser.add_argument("--sample-dir", default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--model", default=YOLO_BUILDING_MODEL_PATH)
    parser.add_argument("--confidence", type=float,
                        default=YOLO_BUILDING_CONFIDENCE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args(argv)
    if args.frames <= 0:
        parser.error("--frames must be positive")
    if args.reference_distance_mm <= 0.0:
        parser.error("--reference-distance-mm must be positive")
    if not 0.0 < args.confidence <= 1.0:
        parser.error("--confidence must be in (0, 1]")
    return args


def sample_path(directory, target, distance_mm):
    return os.path.join(directory, "building_%d_%dmm.csv" % (
        int(target), int(distance_mm)))


def save_samples(path, distance_mm, measurements):
    temporary = path + ".tmp"
    with open(temporary, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("distance_mm", "width_px"))
        for width_px in measurements:
            writer.writerow((distance_mm, width_px))
    os.replace(temporary, path)


def load_samples(path):
    result = []
    with open(path, "r", newline="") as stream:
        for row in csv.DictReader(stream):
            result.append((float(row["distance_mm"]),
                           float(row["width_px"])))
    return result


def configure_capture(capture):
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, YOLO_FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, YOLO_FRAME_HEIGHT)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def prompt_building(display_name, distance_mm, frames, index, total,
                    no_window):
    print("[%d/%d] 楼宇=%s，真实镜头距离=%dmm" % (
        index, total, display_name, distance_mm))
    print("车身和摄像机保持正对楼面，准确测量镜头平面到楼面的垂直距离。")
    if not no_window:
        print("点击检测窗口后按 Enter 采集%d帧；s跳过；q退出。" % frames)
        return ""
    print("无窗口模式：按 Enter 采集%d帧；s跳过；q退出。" % frames)
    return input("> ").strip().lower()


def collect_distance(capture, detector, class_name, confidence,
                     distance_mm, frame_count, no_window):
    measurements = []
    collecting = bool(no_window)
    while len(measurements) < frame_count:
        ok, frame = capture.read()
        if not ok:
            time.sleep(0.02)
            continue
        if frame.shape[:2] != (YOLO_FRAME_HEIGHT, YOLO_FRAME_WIDTH):
            raise RuntimeError("摄像头实际分辨率为%dx%d，要求%d×%d" % (
                frame.shape[1], frame.shape[0],
                YOLO_FRAME_WIDTH, YOLO_FRAME_HEIGHT))
        detections = detector.detect(frame)
        selected = select_locked_building_detection(
            detections, class_name, confidence)
        if selected is not None and collecting:
            if not detection_center_in_x_roi(
                    selected, YOLO_BUILDING_CENTER_ROI_X_RATIO):
                print("\r等待楼宇框中心进入红框...", end="")
            else:
                try:
                    geometry = building_box_geometry(selected, frame.shape)
                except RuntimeError as exc:
                    print("\r等待完整楼宇框：%s" % exc, end="")
                else:
                    measurements.append(geometry["width_px"])
                    print("\r%dmm 有效样本 %d/%d" % (
                        distance_mm, len(measurements), frame_count), end="")
        if not no_window:
            shown = draw_yolo_boxes(
                frame, detections, 1.0, draw_center_band=False,
                center_roi_x_ratio=YOLO_BUILDING_CENTER_ROI_X_RATIO)
            cv2.imshow("building_delivery_distance_calibration", shown)
            key = cv2.waitKey(1) & 0xFF
            terminal_key = None
            try:
                if select.select([sys.stdin], [], [], 0)[0]:
                    terminal_key = sys.stdin.readline().strip().lower()
            except (OSError, ValueError):
                pass
            if (key in (10, 13) or terminal_key == "") and not collecting:
                collecting = True
                print("开始记录%dmm的%d帧数据。" % (
                    distance_mm, frame_count))
            elif (key == ord("s") or terminal_key == "s") \
                    and not collecting:
                print("已跳过当前楼宇。")
                return None
            elif key in (27, ord("q")) or terminal_key == "q":
                raise KeyboardInterrupt()
    print("")
    return measurements


def main(argv=None):
    args = parse_args(argv)
    targets = sorted(ID_TO_BUILDING_CLASS)
    os.makedirs(args.sample_dir, exist_ok=True)
    detector = YoloObstacleDetector(
        args.model, confidence=args.confidence,
        image_size=YOLO_IMAGE_SIZE, nms_threshold=YOLO_NMS_THRESHOLD,
        class_names=YOLO_BUILDING_CLASS_NAMES)
    capture = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
    if not capture.isOpened():
        capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        raise RuntimeError("无法打开楼宇摄像头：%d" % args.camera_index)
    configure_capture(capture)
    try:
        for distance_index, distance_mm in enumerate(args.distances, 1):
            print("\n=== 距离 %d/%d：%dmm，依次采集四类楼宇 ===" % (
                distance_index, len(args.distances), distance_mm))
            for target_index, target in enumerate(targets, 1):
                path = sample_path(args.sample_dir, target, distance_mm)
                if os.path.isfile(path) and not args.overwrite:
                    print("[%d/%d] 跳过已有%s样本：%s" % (
                        target_index, len(targets),
                        BUILDING_NAME_BY_ID[target], path))
                    continue
                choice = prompt_building(
                    BUILDING_NAME_BY_ID[target], distance_mm, args.frames,
                    target_index, len(targets), args.no_window)
                if choice in ("q", "quit", "exit"):
                    print("采集已停止；再次运行会继续缺失距离。")
                    return 0
                if choice in ("s", "skip"):
                    continue
                measurements = collect_distance(
                    capture, detector, ID_TO_BUILDING_CLASS[target],
                    args.confidence, distance_mm, args.frames,
                    args.no_window)
                if measurements is None:
                    continue
                save_samples(path, distance_mm, measurements)
    finally:
        capture.release()
        detector.close()
        if not args.no_window:
            cv2.destroyAllWindows()

    entries = {}
    for target in targets:
        samples = []
        for distance_mm in args.distances:
            path = sample_path(args.sample_dir, target, distance_mm)
            if os.path.isfile(path):
                samples.extend(load_samples(path))
        entries[target] = build_distance_calibration_entry(
            target, ID_TO_BUILDING_CLASS[target], samples,
            YOLO_FRAME_WIDTH, YOLO_FRAME_HEIGHT,
            args.model, args.reference_distance_mm)
    existing = load_building_calibration(
        args.output, YOLO_FRAME_WIDTH, YOLO_FRAME_HEIGHT,
        args.model, allow_missing=True)
    payload = existing or empty_building_calibration(
        YOLO_FRAME_WIDTH, YOLO_FRAME_HEIGHT, args.model)
    for target, entry in entries.items():
        payload["targets"][str(target)] = entry
    save_building_calibration(args.output, payload)
    print("四类楼宇多距离模型已原子保存：%s" % args.output)
    for target, entry in entries.items():
        print("ID%d %s：框宽 RMSE=%.1fmm，距离点=%d，帧=%d" % (
                  target, BUILDING_NAME_BY_ID[target],
                  entry["width"]["rmse_mm"],
                  entry["distance_point_count"], entry["sample_count"]))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已取消楼宇距离标定；已有CSV可以续采。")
        sys.exit(1)
    except Exception as exc:
        print("楼宇距离标定失败：%s" % exc, file=sys.stderr)
        sys.exit(1)
