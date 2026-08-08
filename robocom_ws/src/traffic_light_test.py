#!/usr/bin/env python3
# coding=utf-8
"""独立测试红绿灯摄像头和 YOLOv5 ONNX 检测框。"""
import argparse
import sys
import time

import cv2

from traffic_light_vision import (
    TrafficLightDetector,
    configure_traffic_camera,
    draw_traffic_light,
    set_capture_resolution,
    update_green_hits,
)


CAMERA_INDEX = 0
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
MODEL_PATH = (
    "/home/eaibot/handeye-calib/src/model/yolov5/"
    "traffic_lights_yolov5n_320_best.onnx"
)
CONFIDENCE = 0.55
GREEN_STABLE_FRAMES = 2
WINDOW_NAME = "traffic_light_yolo_test"


def parse_args():
    parser = argparse.ArgumentParser(
        description="显示摄像头 0 的红绿灯 YOLO 检测框"
    )
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX,
                        help="红绿灯摄像头编号，默认 0")
    parser.add_argument("--model", default=MODEL_PATH,
                        help="红绿灯 ONNX 模型路径")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE,
                        help="最低检测置信度，默认 0.55")
    parser.add_argument("--green-frames", type=int,
                        default=GREEN_STABLE_FRAMES,
                        help="连续绿灯确认帧数，默认 2")
    parser.add_argument(
        "--skip-camera-config", action="store_true",
        help="跳过 v4l2 曝光和白平衡设置，仅用于不支持这些控制项的摄像头",
    )
    return parser.parse_args()


def open_camera(camera_index):
    backend = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else 0
    capture = cv2.VideoCapture(int(camera_index), backend)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(int(camera_index))
    if not capture.isOpened():
        raise RuntimeError("无法打开红绿灯摄像头 /dev/video%d" % camera_index)
    set_capture_resolution(capture, FRAME_WIDTH, FRAME_HEIGHT)
    return capture


def draw_status(frame, green_ready, fps):
    text = "PASS" if green_ready else "WAIT"
    color = (0, 255, 0) if green_ready else (0, 0, 255)
    cv2.putText(frame, text, (8, 48), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, color, 2)
    cv2.putText(frame, "FPS %.1f" % fps, (8, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return frame


def run(args):
    confidence = max(0.01, min(1.0, float(args.confidence)))
    green_required = max(1, int(args.green_frames))
    if not args.skip_camera_config:
        configure_traffic_camera(args.camera)

    detector = TrafficLightDetector(
        args.model,
        confidence=confidence,
        image_size=320,
    )
    detector.load()
    capture = None
    green_hits = 0
    last_color = None
    last_time = time.perf_counter()
    fps = 0.0

    try:
        capture = open_camera(args.camera)
        print("红绿灯测试已启动：camera=%d model=%s" %
              (args.camera, detector.model_path))
        print("窗口中绿色框=Green，红色框=Red，黄色框=Yellow；按 q 或 Esc 退出")
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("红绿灯摄像头读取失败")

            detections = detector.detect(frame)
            green_hits, green_ready, color = update_green_hits(
                detections, green_hits, green_required
            )
            if color != last_color:
                print("当前灯色：%s" % (color or "未识别"))
                last_color = color

            now = time.perf_counter()
            elapsed = max(1e-6, now - last_time)
            instant_fps = 1.0 / elapsed
            fps = instant_fps if fps <= 0.0 else 0.85 * fps + 0.15 * instant_fps
            last_time = now

            display = draw_traffic_light(
                frame, detections, color, green_hits, green_required
            )
            cv2.imshow(WINDOW_NAME, draw_status(display, green_ready, fps))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        if capture is not None:
            capture.release()
        detector.close()
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    try:
        run(args)
    except (RuntimeError, IOError, cv2.error) as exc:
        print("红绿灯测试失败：%s" % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
