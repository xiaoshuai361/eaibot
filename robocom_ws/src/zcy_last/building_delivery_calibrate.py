#!/usr/bin/env python3
# coding=utf-8
"""Capture per-class 320x240 YOLO box calibration at a taught stop pose."""

import argparse
import sys
import time

import cv2

from .algorithms.building_delivery import (
    build_calibration_entry,
    building_box_measurement,
    empty_building_calibration,
    load_building_calibration,
    save_building_calibration,
    select_locked_building_detection,
)
from .algorithms.vision import YoloObstacleDetector, draw_yolo_boxes
from .config import (
    BUILDING_DELIVERY_CALIBRATION_FILE,
    UNTAGGED_DELIVERY_ID_BY_BUILDING_CLASS,
    YOLO_BUILDING_CLASS_NAMES,
    YOLO_BUILDING_CONFIDENCE,
    YOLO_BUILDING_MODEL_PATH,
    YOLO_CAMERA_INDEX,
    YOLO_FRAME_HEIGHT,
    YOLO_FRAME_WIDTH,
    YOLO_IMAGE_SIZE,
    YOLO_NMS_THRESHOLD,
)


ID_TO_BUILDING_CLASS = dict(
    (item_id, class_name)
    for class_name, item_id in UNTAGGED_DELIVERY_ID_BY_BUILDING_CLASS.items())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="在正确停车姿态采集楼宇YOLO框尺度标定")
    parser.add_argument("--target", type=int, choices=sorted(ID_TO_BUILDING_CLASS),
                        required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--camera-index", type=int, default=YOLO_CAMERA_INDEX)
    parser.add_argument("--output", default=BUILDING_DELIVERY_CALIBRATION_FILE)
    parser.add_argument("--model", default=YOLO_BUILDING_MODEL_PATH)
    parser.add_argument("--confidence", type=float,
                        default=YOLO_BUILDING_CONFIDENCE)
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args(argv)
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if not 0.0 < args.confidence <= 1.0:
        parser.error("--confidence must be in (0, 1]")
    return args


def update_calibration_payload(existing, entry, frame_width, frame_height,
                               model_path):
    payload = existing or empty_building_calibration(
        frame_width, frame_height, model_path)
    payload["targets"][str(int(entry["item_id"]))] = entry
    return payload


def configure_capture(capture):
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, YOLO_FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, YOLO_FRAME_HEIGHT)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def main(argv=None):
    args = parse_args(argv)
    class_name = ID_TO_BUILDING_CLASS[args.target]
    detector = YoloObstacleDetector(
        args.model,
        confidence=args.confidence,
        image_size=YOLO_IMAGE_SIZE,
        nms_threshold=YOLO_NMS_THRESHOLD,
        class_names=YOLO_BUILDING_CLASS_NAMES,
    )
    capture = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
    if not capture.isOpened():
        capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        raise RuntimeError("无法打开楼宇摄像头：%d" % args.camera_index)
    configure_capture(capture)
    measurements = []
    try:
        print("目标 ID%d：%s" % (args.target, class_name))
        print("请先把车辆放到正确投递停车姿态；开始采集%d个有效框。" % args.samples)
        while len(measurements) < args.samples:
            ok, frame = capture.read()
            if not ok:
                time.sleep(0.02)
                continue
            if frame.shape[1] != YOLO_FRAME_WIDTH \
                    or frame.shape[0] != YOLO_FRAME_HEIGHT:
                raise RuntimeError(
                    "楼宇摄像头实际分辨率为%dx%d，要求%d×%d" % (
                        frame.shape[1], frame.shape[0],
                        YOLO_FRAME_WIDTH, YOLO_FRAME_HEIGHT))
            detections = detector.detect(frame)
            selected = select_locked_building_detection(
                detections, class_name, args.confidence)
            if selected is not None:
                measurements.append(building_box_measurement(
                    selected, frame.shape))
                print("\r有效样本 %d/%d" %
                      (len(measurements), args.samples), end="")
            if not args.no_window:
                shown = draw_yolo_boxes(
                    frame, detections, 1.0, draw_center_band=False)
                cv2.putText(
                    shown, "%s %d/%d" % (
                        class_name, len(measurements), args.samples),
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2)
                cv2.imshow("building_delivery_calibration", shown)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    raise KeyboardInterrupt()
        print("")
    finally:
        capture.release()
        detector.close()
        if not args.no_window:
            cv2.destroyAllWindows()
    entry = build_calibration_entry(
        args.target, class_name, measurements,
        YOLO_FRAME_WIDTH, YOLO_FRAME_HEIGHT, args.model)
    existing = load_building_calibration(
        args.output, YOLO_FRAME_WIDTH, YOLO_FRAME_HEIGHT, args.model,
        allow_missing=True)
    payload = update_calibration_payload(
        existing, entry, YOLO_FRAME_WIDTH, YOLO_FRAME_HEIGHT, args.model)
    save_building_calibration(args.output, payload)
    print("ID%d楼宇标定已保存：%s" % (args.target, args.output))
    print("center=%.5f scale=%.5f samples=%d" % (
        entry["center_x_ratio"], entry["scale_ratio"],
        entry["sample_count"]))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("已取消楼宇标定\n")
        sys.exit(1)
    except Exception as exc:
        sys.stderr.write("楼宇标定失败：%s\n" % exc)
        sys.exit(1)
