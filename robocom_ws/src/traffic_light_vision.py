#!/usr/bin/env python
# coding=utf-8
"""红绿灯摄像头配置和 YOLOv5 ONNX 推理。"""
import gc
import os
import subprocess

import cv2
import numpy as np


TRAFFIC_LIGHT_CLASS_NAMES = ("Green", "Red", "Yellow")
TRAFFIC_CAMERA_EXPOSURE = 15


def traffic_camera_command(camera_index):
    """生成真机要求的 v4l2 控制命令，便于启动前检查和测试。"""
    return [
        "v4l2-ctl", "-d", "/dev/video%d" % int(camera_index),
        "-c", "exposure_auto=1",
        "-c", "exposure_absolute=%d" % TRAFFIC_CAMERA_EXPOSURE,
        "-c", "white_balance_temperature_auto=0",
        "-c", "white_balance_temperature=4600",
        "-c", "exposure_auto_priority=0",
    ]


def configure_traffic_camera(camera_index, runner=None):
    """设置红绿灯相机曝光和白平衡；失败时由调用方决定是否重试。"""
    runner = subprocess.check_call if runner is None else runner
    command = traffic_camera_command(camera_index)
    try:
        runner(command)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("红绿灯摄像头参数设置失败：%s" % exc)
    return command


def set_capture_resolution(capture, width=320, height=240):
    """把红绿灯摄像头固定为模型采集使用的 320x240。"""
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)


class TrafficLightDetection(object):
    def __init__(self, class_id, confidence, box):
        self.class_id = int(class_id)
        self.class_name = TRAFFIC_LIGHT_CLASS_NAMES[self.class_id]
        self.confidence = float(confidence)
        self.box = tuple(float(value) for value in box)


class TrafficLightDetector(object):
    """仅在停止线等待阶段持有 ONNX 网络。"""
    def __init__(self, model_path, confidence=0.55, image_size=320,
                 nms_threshold=0.45):
        self.model_path = os.path.expanduser(str(model_path))
        self.confidence = float(confidence)
        self.image_size = int(image_size)
        self.nms_threshold = float(nms_threshold)
        self.model = None

    @property
    def loaded(self):
        return self.model is not None

    def load(self):
        if self.model is not None:
            return
        if not os.path.isfile(self.model_path):
            raise IOError("红绿灯 ONNX 模型不存在：%s" % self.model_path)
        try:
            model = cv2.dnn.readNetFromONNX(self.model_path)
            model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except Exception as exc:
            raise RuntimeError("红绿灯 ONNX 模型加载失败：%s" % exc)
        self.model = model

    def close(self):
        """绿灯放行后立即释放网络，避免占用后续路口算力。"""
        self.model = None
        gc.collect()

    def _letterbox(self, frame):
        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("红绿灯画面尺寸无效")
        size = self.image_size
        scale = min(float(size) / width, float(size) / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(
            frame, (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        padded = np.full((size, size, 3), 114, dtype=np.uint8)
        pad_left = (size - resized_width) // 2
        pad_top = (size - resized_height) // 2
        padded[pad_top:pad_top + resized_height,
               pad_left:pad_left + resized_width] = resized
        return padded, scale, pad_left, pad_top

    def _prediction_rows(self, output):
        if isinstance(output, (list, tuple)):
            if not output:
                return np.empty((0, 8), dtype=np.float32)
            output = output[0]
        rows = np.asarray(output)
        if rows.ndim == 3 and rows.shape[0] == 1:
            rows = rows[0]
        if rows.ndim != 2:
            return np.empty((0, 8), dtype=np.float32)
        expected_fields = 5 + len(TRAFFIC_LIGHT_CLASS_NAMES)
        if rows.shape[0] == expected_fields and rows.shape[1] != expected_fields:
            rows = rows.T
        if rows.shape[1] != expected_fields:
            return np.empty((0, expected_fields), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)

    def _decode(self, output, frame_shape, scale, pad_left, pad_top):
        height, width = frame_shape[:2]
        candidates = []
        for row in self._prediction_rows(output):
            objectness = float(row[4])
            class_id = int(np.argmax(row[5:]))
            confidence = objectness * float(row[5 + class_id])
            if confidence < self.confidence:
                continue
            center_x, center_y, box_width, box_height = row[:4]
            x1 = (float(center_x - box_width * 0.5) - pad_left) / scale
            y1 = (float(center_y - box_height * 0.5) - pad_top) / scale
            x2 = (float(center_x + box_width * 0.5) - pad_left) / scale
            y2 = (float(center_y + box_height * 0.5) - pad_top) / scale
            x1 = max(0.0, min(float(width - 1), x1))
            y1 = max(0.0, min(float(height - 1), y1))
            x2 = max(0.0, min(float(width - 1), x2))
            y2 = max(0.0, min(float(height - 1), y2))
            if x2 <= x1 or y2 <= y1:
                continue
            candidates.append((class_id, confidence, (x1, y1, x2, y2)))

        detections = []
        for class_id in range(len(TRAFFIC_LIGHT_CLASS_NAMES)):
            local = [item for item in candidates if item[0] == class_id]
            if not local:
                continue
            boxes = [
                [item[2][0], item[2][1],
                 item[2][2] - item[2][0], item[2][3] - item[2][1]]
                for item in local
            ]
            scores = [item[1] for item in local]
            indices = cv2.dnn.NMSBoxes(
                boxes, scores, self.confidence, self.nms_threshold
            )
            for index in np.asarray(indices).reshape(-1):
                item = local[int(index)]
                detections.append(TrafficLightDetection(*item))
        return sorted(detections, key=lambda item: item.confidence, reverse=True)

    def detect(self, frame):
        if self.model is None:
            raise RuntimeError("红绿灯模型尚未加载")
        padded, scale, pad_left, pad_top = self._letterbox(frame)
        blob = cv2.dnn.blobFromImage(
            padded, 1.0 / 255.0,
            (self.image_size, self.image_size),
            swapRB=True, crop=False,
        )
        self.model.setInput(blob)
        output = self.model.forward()
        return self._decode(output, frame.shape, scale, pad_left, pad_top)


def update_green_hits(detections, current_hits, required_hits):
    """只有最高置信度结果连续为绿灯时才允许车辆通过。"""
    selected = max(detections, key=lambda item: item.confidence) \
        if detections else None
    color = None if selected is None else selected.class_name
    hits = int(current_hits) + 1 if color == "Green" else 0
    return hits, hits >= int(required_hits), color


def draw_traffic_light(frame, detections, color=None, green_hits=0,
                       required_hits=2,
                       exposure=TRAFFIC_CAMERA_EXPOSURE):
    output = frame.copy()
    colors = {
        "Green": (0, 255, 0),
        "Red": (0, 0, 255),
        "Yellow": (0, 255, 255),
    }
    for item in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in item.box]
        box_color = colors.get(item.class_name, (255, 255, 255))
        cv2.rectangle(output, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(
            output, "%s %.2f" % (item.class_name, item.confidence),
            (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, box_color, 2,
        )
    text = "signal=%s green=%d/%d" % (
        color or "--", int(green_hits), int(required_hits)
    )
    cv2.putText(output, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2)
    cv2.putText(
        output, "exposure_absolute=%d" % int(exposure),
        (8, 44), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (255, 255, 255), 2,
    )
    return output
