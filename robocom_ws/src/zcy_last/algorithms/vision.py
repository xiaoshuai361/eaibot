#!/usr/bin/env python3
# coding=utf-8
"""不直接操作 ROS 硬件的巡线、横条和任务识别算法。"""
import os
import shutil

import cv2
import numpy as np

from ..config import *  # noqa: F401,F403

def clamp(value, low, high):
    return max(low, min(high, value))


def motion_enabled(dry_run):
    return not bool(dry_run)


def normalize_angle(angle):
    angle = float(angle)
    while angle > 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    return angle


def angle_delta(a, b):
    delta = abs(normalize_angle(a) - normalize_angle(b))
    return min(delta, 180.0 - delta)


def polygon_long_angle(points):
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    edges = np.roll(points, -1, axis=0) - points
    edge = edges[int(np.argmax(np.sum(edges * edges, axis=1)))]
    return normalize_angle(np.degrees(np.arctan2(edge[1], edge[0])))


def polygon_bottom_in_center_band(polygon, frame_width,
                                  center_width_ratio=STOP_CENTER_WIDTH_RATIO):
    """返回多边形在画面中央指定宽度内的最低点，未相交时返回 0。"""
    if polygon is None or frame_width <= 0:
        return 0.0
    ratio = clamp(float(center_width_ratio), 0.0, 1.0)
    if ratio <= 0.0:
        return 0.0
    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if len(points) < 2:
        return 0.0
    left = float(frame_width) * (1.0 - ratio) * 0.5
    right = float(frame_width) - left
    y_values = []
    for index, first in enumerate(points):
        second = points[(index + 1) % len(points)]
        if left <= first[0] <= right:
            y_values.append(float(first[1]))
        delta_x = float(second[0] - first[0])
        if abs(delta_x) < 1e-6:
            continue
        for boundary in (left, right):
            amount = (boundary - float(first[0])) / delta_x
            if 0.0 <= amount <= 1.0:
                y_values.append(float(first[1] + amount * (second[1] - first[1])))
    return max(y_values) if y_values else 0.0


def find_contours(binary):
    result = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return result[0] if len(result) == 2 else result[1]


def ensure_clean_directory(path):
    path = os.path.expanduser(str(path))
    if os.path.isdir(path):
        for name in os.listdir(path):
            item = os.path.join(path, name)
            if os.path.isdir(item):
                shutil.rmtree(item)
            else:
                os.remove(item)
    else:
        os.makedirs(path)


def safe_filename_text(text):
    return str(text).replace("/", "_").replace("\\", "_").replace(" ", "")


def detection_center_in_x_roi(detection, roi_x_ratio):
    left_ratio, right_ratio = [float(value) for value in roi_x_ratio]
    width = int(detection.frame_shape[1])
    return left_ratio * width <= detection.center_x <= right_ratio * width


def draw_yolo_boxes(frame, detections, center_band_ratio,
                    draw_center_band=True, center_roi_x_ratio=None):
    output = frame.copy()
    height, width = output.shape[:2]
    if draw_center_band:
        ratio = clamp(float(center_band_ratio), 0.0, 1.0)
        left = int(round(width * (1.0 - ratio) * 0.5))
        right = int(round(width - left))
        cv2.line(output, (left, 0), (left, height - 1), (255, 255, 0), 1)
        cv2.line(output, (right, 0), (right, height - 1), (255, 255, 0), 1)
    if center_roi_x_ratio is not None:
        left_ratio, right_ratio = center_roi_x_ratio
        left = int(round(width * float(left_ratio)))
        right = int(round(width * float(right_ratio)))
        cv2.rectangle(
            output, (left, 0), (right, height - 1), (0, 0, 255), 2)
    for item in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in item.box]
        in_roi = detection_center_in_x_roi(item, center_roi_x_ratio) \
            if center_roi_x_ratio is not None else item.in_center
        color = (0, 255, 0) if item.target and in_roi else (0, 255, 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = "{} {:.2f}".format(item.class_name, item.confidence)
        cv2.putText(output, label, (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(output, (int(round(item.center_x)), int(round(item.center_y))),
                   3, color, -1)
    return output


def row_segments(row, min_width=MIN_SEGMENT_WIDTH, max_width=None):
    pixels = np.flatnonzero(row > 0)
    if len(pixels) == 0:
        return []
    segments = []
    start = last = int(pixels[0])
    for pixel in pixels[1:]:
        pixel = int(pixel)
        if pixel == last + 1:
            last = pixel
            continue
        width = last - start + 1
        if width >= min_width and (max_width is None or width <= max_width):
            segments.append((start, last))
        start = last = pixel
    width = last - start + 1
    if width >= min_width and (max_width is None or width <= max_width):
        segments.append((start, last))
    return segments


def center_out_segments(row, center_x, min_width=MIN_SEGMENT_WIDTH, max_width=None):
    """从画面中心向两侧取第一段黑色，外部黑块不会进入车道配对。"""
    segments = row_segments(row, min_width, max_width)
    left = [item for item in segments if item[1] < center_x]
    right = [item for item in segments if item[0] > center_x]
    return (max(left, key=lambda item: item[1]) if left else None,
            min(right, key=lambda item: item[0]) if right else None)


def clip_points_for_display(points, width):
    """仅把调试点裁到画面边缘，不改变控制使用的真实补线坐标。"""
    return [(int(clamp(round(x), 0, width - 1)), int(y)) for x, y in points]


def control_target_x(center_x, bias_pixels):
    return float(center_x) + float(bias_pixels)


def pd_gains(deviation):
    if abs(float(deviation)) >= LARGE_ERROR_THRESHOLD_PIXELS:
        return LARGE_ERROR_KP, LARGE_ERROR_KD, "large"
    return KP, KD, "small"


def normalize_turn_cmd(value):
    command = str(value).strip().lower()
    return command if command in ("left", "straight", "right") else "straight"


def maneuver_follow_side(turn_cmd):
    command = normalize_turn_cmd(turn_cmd)
    return command if command in ("left", "right") else None


def strong_lane_override_enabled(state, turn_cmd, maneuver_phase):
    return (
        state == "MANEUVER"
        and maneuver_follow_side(turn_cmd) is not None
        and maneuver_phase == "EXIT_STRAIGHT"
    )


def fixed_turn_command(turn_cmd, speed, angular):
    command = normalize_turn_cmd(turn_cmd)
    if command == "left":
        return float(speed), float(angular)
    if command == "right":
        return float(speed), -float(angular)
    return None


def turn_phase_next(phase, elapsed, entry_time, turn_time):
    if phase == "ENTRY" and elapsed >= entry_time:
        return "TURN"
    if phase == "TURN" and elapsed >= turn_time:
        return "EXIT_STRAIGHT"
    return None


def follow_entry_hits(candidate, current_hits):
    return current_hits + 1 if candidate else max(0, current_hits - 1)


def entry_acceptance_enabled(now, accept_after):
    return float(now) >= float(accept_after)


def yolo_route_context(task_index, state):
    if state in ("FINAL_EXIT", "DONE"):
        return {"kind": "off"}
    index = int(task_index)
    if index in YOLO_STREET_ROUTE_AREAS:
        return {"kind": "street", "areas": YOLO_STREET_ROUTE_AREAS[index]}
    if index in YOLO_BUILDING_ROUTE_AREAS:
        return {"kind": "building", "area": YOLO_BUILDING_ROUTE_AREAS[index]}
    return {"kind": "off"}


def yolo_model_profile(task_index):
    """返回当前路线应使用的任务识别模型。"""
    return "building" if int(task_index) >= YOLO_BUILDING_SWITCH_INDEX \
        else "street"


def same_tracked_bar(first, second, shape):
    height, width = shape[:2]
    return (
        abs(first["center"][1] - second["center"][1])
        <= height * BAR_TRACK_MAX_Y_RATIO
        and abs(first["center"][0] - second["center"][0])
        <= width * BAR_TRACK_MAX_X_RATIO
        and angle_delta(first["angle"], second["angle"])
        <= BAR_TRACK_MAX_ANGLE
    )


def resolve_yolo_model_path(model_path):
    path = os.path.expanduser(str(model_path))
    if os.path.isfile(path):
        if not path.lower().endswith(".onnx"):
            raise IOError("YOLO model must be an .onnx file: %s" % path)
        return path
    if not os.path.isdir(path):
        raise IOError("YOLO model path does not exist: %s" % path)
    onnx_files = [
        os.path.join(path, name)
        for name in os.listdir(path)
        if name.lower().endswith(".onnx")
    ]
    if not onnx_files:
        raise IOError("No .onnx YOLO model found in: %s" % path)
    by_name = {os.path.basename(item).lower(): item for item in onnx_files}
    for name in YOLO_MODEL_PREFERRED_FILES:
        preferred = by_name.get(name.lower())
        if preferred is not None:
            return preferred
    return sorted(onnx_files)[0]


class YoloDetection(object):
    def __init__(self, class_id, class_name, confidence, box,
                 frame_shape, center_band_ratio):
        self.class_id = int(class_id)
        self.class_name = str(class_name)
        self.confidence = float(confidence)
        self.box = tuple(float(value) for value in box)
        self.frame_shape = frame_shape
        self.center_band_ratio = float(center_band_ratio)
        height, width = frame_shape[:2]
        self.center_x = (self.box[0] + self.box[2]) * 0.5
        self.center_y = (self.box[1] + self.box[3]) * 0.5
        ratio = clamp(self.center_band_ratio, 0.0, 1.0)
        left = float(width) * (1.0 - ratio) * 0.5
        right = float(width) - left
        self.in_center = left <= self.center_x <= right
        self.target = self.class_name in YOLO_TARGET_CLASS_NAMES


class YoloTaskEvent(object):
    def __init__(self, kind, area, class_name, display_name, detection):
        self.kind = str(kind)
        self.area = str(area)
        self.class_name = str(class_name)
        self.display_name = str(display_name)
        self.detection = detection


class YoloTaskLedger(object):
    def __init__(self):
        self.street_results = dict(
            (area, None) for area in ("C区", "P区", "A区", "S区")
        )
        self.street_seen_classes = set()
        self.building_results = dict(
            (area, None) for area in ("楼宇A", "楼宇B", "楼宇C", "楼宇D")
        )
        self.building_seen_classes = set()
        self.people_stable_key = None
        self.people_stable_hits = 0
        self.pending_event = None
        self.save_index = 0

    def _target_candidates(self, detections, confidence):
        return [
            item for item in detections
            if item.target and item.in_center
            and item.confidence >= float(confidence)
        ]

    def _next_street_area(self, areas):
        for area in areas:
            if self.street_results.get(area) is None:
                return area
        return None

    def _reset_people_stability(self):
        self.people_stable_key = None
        self.people_stable_hits = 0

    def _stable_people_candidate(self, area, candidates, stable_frames):
        grouped = dict((name, []) for name in YOLO_PEOPLE_CLASS_NAMES)
        for item in candidates:
            if item.class_name in grouped:
                grouped[item.class_name].append(item)
        largest = max([len(items) for items in grouped.values()] or [0])
        winners = [
            name for name, items in grouped.items()
            if largest > 0 and len(items) == largest
        ]
        if len(winners) != 1:
            self._reset_people_stability()
            return None
        class_name = winners[0]
        key = (str(area), class_name)
        if key == self.people_stable_key:
            self.people_stable_hits += 1
        else:
            self.people_stable_key = key
            self.people_stable_hits = 1
        if self.people_stable_hits < max(1, int(stable_frames)):
            return None
        return max(grouped[class_name], key=lambda item: item.confidence)

    def select_event(self, context, detections, confidence,
                     building_confidence=None, people_stable_frames=1,
                     trash_confidence=None):
        kind = context.get("kind")
        if kind == "street":
            trash_threshold = confidence if trash_confidence is None \
                else float(trash_confidence)
            candidates = self._target_candidates(
                detections, min(float(confidence), trash_threshold)
            )
            area = self._next_street_area(context.get("areas", ()))
            if area is None:
                return None
            street = []
            for item in candidates:
                if (item.class_name not in YOLO_STREET_MESSAGES
                        or item.class_name in self.street_seen_classes):
                    continue
                target_kind, _ = YOLO_STREET_MESSAGES[item.class_name]
                threshold = trash_threshold if target_kind == "trash" \
                    else float(confidence)
                if item.confidence >= threshold:
                    street.append(item)
            if not street:
                self._reset_people_stability()
                return None
            people = [
                item for item in street
                if item.class_name in YOLO_PEOPLE_CLASS_NAMES
            ]
            if people:
                selected = self._stable_people_candidate(
                    area, people, people_stable_frames
                )
                if selected is None:
                    return None
            else:
                self._reset_people_stability()
                selected = max(street, key=lambda item: item.confidence)
            _, display_name = YOLO_STREET_MESSAGES[selected.class_name]
            return YoloTaskEvent(
                "street", area, selected.class_name, display_name, selected
            )
        if kind == "building":
            self._reset_people_stability()
            threshold = confidence if building_confidence is None \
                else building_confidence
            candidates = [
                item for item in detections
                if item.target
                and detection_center_in_x_roi(
                    item, YOLO_BUILDING_CENTER_ROI_X_RATIO)
                and item.confidence >= float(threshold)
            ]
            area = context.get("area")
            if self.building_results.get(area) is not None:
                return None
            buildings = [
                item for item in candidates
                if item.class_name in YOLO_BUILDING_MESSAGE_BY_CLASS
                and item.class_name not in self.building_seen_classes
            ]
            if not buildings:
                return None
            selected = max(buildings, key=lambda item: item.confidence)
            return YoloTaskEvent(
                "building", area, selected.class_name,
                YOLO_BUILDING_MESSAGE_BY_CLASS[selected.class_name],
                selected,
            )
        return None

    def accept(self, event):
        self.pending_event = event
        if event is None:
            return
        if event.kind == "street":
            self.street_results[event.area] = event
            self.street_seen_classes.add(event.class_name)
            self._reset_people_stability()
        elif event.kind == "building":
            self.building_results[event.area] = event
            self.building_seen_classes.add(event.class_name)


class YoloObstacleDetector(object):
    def __init__(self, model_path, confidence=YOLO_CONFIDENCE,
                 center_band_ratio=YOLO_CENTER_BAND_RATIO,
                 image_size=YOLO_IMAGE_SIZE,
                 nms_threshold=YOLO_NMS_THRESHOLD,
                 class_names=YOLO_CLASS_NAMES):
        self.model_path = resolve_yolo_model_path(model_path)
        self.confidence = float(confidence)
        self.center_band_ratio = float(center_band_ratio)
        self.image_size = int(image_size)
        self.nms_threshold = float(nms_threshold)
        self.model = None
        self.names = self._normalize_names(class_names)
        self.backend_name = "opencv-dnn-onnx"

    def load(self):
        if self.model is not None:
            return
        try:
            self.model = cv2.dnn.readNetFromONNX(self.model_path)
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except Exception as exc:
            raise RuntimeError("Could not load ONNX YOLO model: %s" % exc)

    def close(self):
        """释放 OpenCV DNN 模型内存，供路线中途切换模型。"""
        self.model = None

    def _normalize_names(self, names):
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        if isinstance(names, (list, tuple)):
            return {index: str(value) for index, value in enumerate(names)}
        return {}

    def _letterbox(self, frame):
        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("YOLO frame has invalid shape")
        size = int(self.image_size)
        scale = min(float(size) / float(width), float(size) / float(height))
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(frame, (resized_width, resized_height),
                             interpolation=cv2.INTER_LINEAR)
        padded = np.full((size, size, 3), 114, dtype=np.uint8)
        pad_left = (size - resized_width) // 2
        pad_top = (size - resized_height) // 2
        padded[pad_top:pad_top + resized_height,
               pad_left:pad_left + resized_width] = resized
        return padded, scale, pad_left, pad_top

    def _predictions_from_output(self, output):
        if isinstance(output, (list, tuple)):
            if not output:
                return np.empty((0, 5 + len(self.names)), dtype=np.float32)
            output = output[0]
        data = np.asarray(output)
        if data.ndim == 3 and data.shape[0] == 1:
            data = data[0]
        if data.ndim != 2:
            return np.empty((0, 5 + len(self.names)), dtype=np.float32)
        field_counts = (4 + len(self.names), 5 + len(self.names))
        if data.shape[0] in field_counts:
            data = data.T
        elif data.shape[1] not in field_counts:
            return np.empty((0, 5 + len(self.names)), dtype=np.float32)
        return np.asarray(data, dtype=np.float32)

    def _nms_indices(self, boxes, confidences):
        kept = []
        class_ids = sorted(set(item[0] for item in boxes))
        for class_id in class_ids:
            local = [index for index, item in enumerate(boxes)
                     if item[0] == class_id]
            local_boxes = [boxes[index][1] for index in local]
            local_scores = [confidences[index] for index in local]
            indices = cv2.dnn.NMSBoxes(
                local_boxes, local_scores, self.confidence, self.nms_threshold
            )
            for item in np.asarray(indices).reshape(-1):
                kept.append(local[int(item)])
        return sorted(kept, key=lambda index: confidences[index], reverse=True)

    def _decode(self, output, frame_shape, scale, pad_left, pad_top):
        predictions = self._predictions_from_output(output)
        if len(predictions) == 0:
            return []
        coords = predictions[:, :4].copy()
        if np.isfinite(coords).any() and np.nanmax(np.abs(coords)) <= 2.0:
            coords *= float(self.image_size)
        if predictions.shape[1] == 5 + len(self.names):
            objectness = predictions[:, 4]
            scores = predictions[:, 5:]
        else:
            objectness = None
            scores = predictions[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(scores.shape[0]), class_ids]
        if objectness is not None:
            confidences = confidences * objectness
        height, width = frame_shape[:2]
        boxes = []
        selected_confidences = []
        for index, confidence in enumerate(confidences):
            confidence = float(confidence)
            if confidence < self.confidence:
                continue
            center_x, center_y, box_width, box_height = [
                float(value) for value in coords[index]
            ]
            x1 = (center_x - box_width * 0.5 - pad_left) / scale
            y1 = (center_y - box_height * 0.5 - pad_top) / scale
            x2 = (center_x + box_width * 0.5 - pad_left) / scale
            y2 = (center_y + box_height * 0.5 - pad_top) / scale
            x1 = clamp(x1, 0.0, float(width - 1))
            y1 = clamp(y1, 0.0, float(height - 1))
            x2 = clamp(x2, 0.0, float(width - 1))
            y2 = clamp(y2, 0.0, float(height - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append((
                int(class_ids[index]),
                [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            ))
            selected_confidences.append(confidence)

        detections = []
        for index in self._nms_indices(boxes, selected_confidences):
            class_id, box = boxes[index]
            x, y, box_width, box_height = box
            class_name = self.names.get(class_id, str(class_id))
            detections.append(YoloDetection(
                class_id, class_name, selected_confidences[index],
                (x, y, x + box_width, y + box_height),
                frame_shape, self.center_band_ratio,
            ))
        return detections

    def detect(self, frame):
        self.load()
        padded, scale, pad_left, pad_top = self._letterbox(frame)
        blob = cv2.dnn.blobFromImage(
            padded, 1.0 / 255.0, (self.image_size, self.image_size),
            swapRB=True, crop=False,
        )
        self.model.setInput(blob)
        output = self.model.forward()
        return self._decode(output, frame.shape, scale, pad_left, pad_top)


class LaneObservation(object):
    def __init__(self, center_x, valid, dual_rows, left_points, right_points,
                 measured_width=None, follow_side=None, center_points=None,
                 virtual_left_points=None, virtual_right_points=None):
        self.center_x = float(center_x)
        self.valid = bool(valid)
        self.dual_rows = int(dual_rows)
        self.left_points = list(left_points)
        self.right_points = list(right_points)
        self.measured_width = measured_width
        self.follow_side = follow_side
        self.center_points = list(center_points or [])
        self.virtual_left_points = list(virtual_left_points or [])
        self.virtual_right_points = list(virtual_right_points or [])


class LaneDetector(object):
    def __init__(self, roi_top=ROI_TOP, roi_bottom=ROI_BOTTOM, scan_rows=SCAN_ROWS,
                 fill_width=0.0, left_fill_width=None,
                 right_fill_width=None,
                 center_near_weight=LANE_CENTER_NEAR_WEIGHT):
        self.roi_top = float(roi_top)
        self.roi_bottom = float(roi_bottom)
        self.scan_rows = int(scan_rows)
        self.fill_width = float(fill_width)
        self.left_fill_width = (self.fill_width if left_fill_width is None
                                else float(left_fill_width))
        self.right_fill_width = (self.fill_width if right_fill_width is None
                                 else float(right_fill_width))
        self.center_near_weight = max(1.0, float(center_near_weight))

    def _weighted_center_x(self, center_points, frame_height):
        top_y = float(frame_height) * self.roi_top
        bottom_y = float(frame_height) * self.roi_bottom
        span = max(1.0, bottom_y - top_y)
        weights = [
            1.0 + (self.center_near_weight - 1.0)
            * clamp((float(y) - top_y) / span, 0.0, 1.0)
            for _, y in center_points
        ]
        return float(np.average(
            [x for x, _ in center_points], weights=weights
        ))

    def points(self, binary, center_x=None):
        height, width = binary.shape[:2]
        center_x = width * 0.5 if center_x is None else float(center_x)
        rows = np.linspace(int(height * self.roi_bottom), int(height * self.roi_top),
                           self.scan_rows).astype(np.int32)
        left_points, right_points = [], []
        max_width = int(width * MAX_SEGMENT_WIDTH_RATIO)
        for y in rows:
            left, right = center_out_segments(binary[y], center_x, MIN_SEGMENT_WIDTH, max_width)
            if left is not None:
                left_points.append((int(left[1]), int(y)))
            if right is not None:
                right_points.append((int(right[0]), int(y)))
        return left_points, right_points

    def points_near_model(self, binary, model, band_ratio=0.10):
        """锁线后按预测位置取点，靠中心的圆角不能抢走外侧直线。"""
        height, width = binary.shape[:2]
        rows = np.linspace(int(height * self.roi_bottom), int(height * self.roi_top),
                           self.scan_rows).astype(np.int32)
        max_width = int(width * MAX_SEGMENT_WIDTH_RATIO)
        max_distance = width * float(band_ratio)
        points = []
        for y in rows:
            segments = row_segments(binary[y], MIN_SEGMENT_WIDTH, max_width)
            if not segments:
                continue
            expected = model.x_at(y)
            segment = min(segments, key=lambda item: abs((item[0] + item[1]) * 0.5 - expected))
            center = (segment[0] + segment[1]) * 0.5
            if abs(center - expected) <= max_distance:
                points.append((int(round(center)), int(y)))
        return points

    def observe(self, binary, lane_width, follow_side=None, center_hint=None,
                side_center_transform=None):
        height, width = binary.shape[:2]
        center_hint = width * 0.5 if center_hint is None else float(center_hint)
        left_points, right_points = self.points(binary, center_hint)
        left_by_y = {y: x for x, y in left_points}
        right_by_y = {y: x for x, y in right_points}
        expected = float(lane_width) if lane_width > 0 else width * DEFAULT_LANE_WIDTH_RATIO
        left_fill_width = (self.left_fill_width
                           if self.left_fill_width > 0 else expected)
        right_fill_width = (self.right_fill_width
                            if self.right_fill_width > 0 else expected)
        all_rows = sorted(set(left_by_y).union(right_by_y), reverse=True)
        center_points = []
        virtual_left, virtual_right = [], []
        widths = []
        last_center = center_hint
        used_sides = []
        for y in all_rows:
            left_x = left_by_y.get(y)
            right_x = right_by_y.get(y)
            gap = None if left_x is None or right_x is None else right_x - left_x
            dual_ok = (
                gap is not None
                and width * MIN_LANE_WIDTH_RATIO <= gap <= width * MAX_LANE_WIDTH_RATIO
            )
            if follow_side is None and dual_ok:
                center = (left_x + right_x) * 0.5
                widths.append(float(gap))
                used_sides.append("dual")
            else:
                candidates = []
                if left_x is not None and follow_side in (None, "left"):
                    offset = left_fill_width * 0.5
                    if follow_side == "left" and side_center_transform is not None:
                        offset = (side_center_transform[0] * y
                                  + side_center_transform[1])
                    candidates.append((left_x + offset, "left", offset))
                if right_x is not None and follow_side in (None, "right"):
                    offset = -right_fill_width * 0.5
                    if follow_side == "right" and side_center_transform is not None:
                        offset = (side_center_transform[0] * y
                                  + side_center_transform[1])
                    candidates.append((right_x + offset, "right", offset))
                if not candidates:
                    continue
                center, side, offset = min(
                    candidates, key=lambda item: abs(item[0] - last_center)
                )
                if side == "left":
                    virtual_right.append((left_x + offset * 2.0, y))
                else:
                    virtual_left.append((right_x + offset * 2.0, y))
                used_sides.append(side)
            center_points.append((center, y))
            last_center = center

        if not center_points:
            return LaneObservation(center_hint, False, 0, left_points, right_points,
                                   None, follow_side)
        measured = float(np.median(widths)) if widths else None
        if follow_side in ("left", "right"):
            active_side = follow_side
        elif widths:
            active_side = None
        else:
            active_side = max(set(used_sides), key=used_sides.count)
        return LaneObservation(
            self._weighted_center_x(center_points, height), True, len(widths),
            left_points, right_points, measured, active_side, center_points,
            virtual_left_points=virtual_left,
            virtual_right_points=virtual_right,
        )


class LineModel(object):
    def __init__(self, slope, intercept, inlier_count, error):
        self.slope = float(slope)
        self.intercept = float(intercept)
        self.inlier_count = int(inlier_count)
        self.error = float(error)

    def x_at(self, y):
        return self.slope * float(y) + self.intercept

    def shifted(self, slope_delta, intercept_delta):
        return LineModel(self.slope + float(slope_delta),
                         self.intercept + float(intercept_delta),
                         self.inlier_count, self.error)


def fit_line_ransac(points, residual=12.0, prior=None):
    """确定性点对 RANSAC；直线模型会将入口圆角当成离群点。"""
    points = [(float(x), float(y)) for x, y in points]
    if len(points) < 2:
        return None
    best = None
    for i in range(len(points) - 1):
        for j in range(i + 1, len(points)):
            x1, y1 = points[i]
            x2, y2 = points[j]
            if abs(y2 - y1) < 1.0:
                continue
            slope = (x2 - x1) / (y2 - y1)
            intercept = x1 - slope * y1
            errors = np.array([abs(x - (slope * y + intercept)) for x, y in points])
            mask = errors <= residual
            count = int(np.count_nonzero(mask))
            if count < 2:
                continue
            mean_error = float(np.mean(errors[mask]))
            continuity = 0.0
            if prior is not None:
                continuity = abs((slope * 360.0 + intercept) - prior.x_at(360.0)) * 0.03
            score = (-count, mean_error + continuity)
            if best is None or score < best[0]:
                best = (score, mask)
    if best is None:
        return None
    inliers = np.asarray(points, dtype=np.float64)[best[1]]
    if len(inliers) < 2:
        return None
    slope, intercept = np.polyfit(inliers[:, 1], inliers[:, 0], 1)
    error = float(np.mean(np.abs(inliers[:, 0] - (slope * inliers[:, 1] + intercept))))
    return LineModel(slope, intercept, len(inliers), error)


class DualLineBridge(object):
    def __init__(self, lane_width, fill_width=0.0,
                 hold_frames=MODEL_HOLD_FRAMES):
        self.lane_width = float(lane_width)
        self.fill_width = float(fill_width)
        self.hold_frames = int(hold_frames)
        self.left_model = None
        self.right_model = None
        self.left_lost_frames = 0
        self.right_lost_frames = 0
        self.last_center = None
        self.center_model = None
        self.left_to_center = None
        self.right_to_center = None
        self.selected_side = None

    def reset(self, lane_width=None):
        if lane_width is not None:
            self.lane_width = float(lane_width)
        self.left_model = None
        self.right_model = None
        self.left_lost_frames = 0
        self.right_lost_frames = 0
        self.last_center = None
        self.center_model = None
        self.left_to_center = None
        self.right_to_center = None
        self.selected_side = None

    def _learn_center_geometry(self):
        left = self.left_model
        right = self.right_model
        if (left is None or right is None or self.left_lost_frames != 0
                or self.right_lost_frames != 0):
            return
        center_slope = (left.slope + right.slope) * 0.5
        center_intercept = (left.intercept + right.intercept) * 0.5
        inliers = min(left.inlier_count, right.inlier_count)
        error = max(left.error, right.error)
        self.center_model = LineModel(center_slope, center_intercept, inliers, error)
        self.left_to_center = (center_slope - left.slope,
                               center_intercept - left.intercept)
        self.right_to_center = (center_slope - right.slope,
                                center_intercept - right.intercept)

    def _center_from_side(self, model, transform, direction, fill_width):
        if model is None:
            return None
        if transform is not None:
            return model.shifted(transform[0], transform[1])
        shift = fill_width * 0.5 * direction
        return model.shifted(0.0, shift)

    def _model_geometry_valid(self, candidate, side, target_y,
                              frame_width, validation_top_y):
        if candidate is None or abs(candidate.slope) > MODEL_MAX_ABS_SLOPE:
            return False
        center_x = float(frame_width) * 0.5
        margin = float(frame_width) * MODEL_CENTER_CROSS_MARGIN_RATIO
        sample_ys = (float(validation_top_y), float(target_y))
        if side == "left":
            return all(candidate.x_at(y) <= center_x + margin for y in sample_ys)
        return all(candidate.x_at(y) >= center_x - margin for y in sample_ys)

    def _update_model(self, points, model, lost_frames, target_y, side,
                      frame_width, validation_top_y):
        candidate = fit_line_ransac(points, residual=RANSAC_RESIDUAL_PIXELS,
                                    prior=model)
        enough = candidate is not None \
            and candidate.inlier_count >= RANSAC_MIN_INLIERS \
            and self._model_geometry_valid(
                candidate, side, target_y, frame_width, validation_top_y
            )
        continuous = enough
        if continuous and model is not None:
            max_shift = float(frame_width) * MODEL_MAX_SHIFT_RATIO
            continuous = (
                abs(candidate.x_at(target_y) - model.x_at(target_y)) <= max_shift
                and abs(candidate.slope - model.slope) <= MODEL_MAX_SLOPE_DELTA
            )
        if continuous:
            return candidate, 0
        lost_frames += 1
        if lost_frames > self.hold_frames:
            model = None
        return model, lost_frames

    def update(self, left_points, right_points, target_y, center_hint=None,
               frame_width=PROCESS_WIDTH, validation_top_y=None):
        if validation_top_y is None:
            all_points = list(left_points) + list(right_points)
            validation_top_y = min([y for _, y in all_points] or [target_y])
        self.left_model, self.left_lost_frames = self._update_model(
            left_points, self.left_model, self.left_lost_frames, target_y,
            "left", frame_width, validation_top_y
        )
        self.right_model, self.right_lost_frames = self._update_model(
            right_points, self.right_model, self.right_lost_frames, target_y,
            "right", frame_width, validation_top_y
        )
        self._learn_center_geometry()

        fill_width = self.fill_width if self.fill_width > 0 else self.lane_width
        fresh_candidates = []
        held_candidates = []
        if self.left_model is not None:
            model = self._center_from_side(
                self.left_model, self.left_to_center, 1.0, fill_width
            )
            item = (model.x_at(target_y), "left", model)
            (fresh_candidates if self.left_lost_frames == 0 else held_candidates).append(item)
        if self.right_model is not None:
            model = self._center_from_side(
                self.right_model, self.right_to_center, -1.0, fill_width
            )
            item = (model.x_at(target_y), "right", model)
            (fresh_candidates if self.right_lost_frames == 0 else held_candidates).append(item)
        candidates = fresh_candidates if fresh_candidates else held_candidates
        if not candidates:
            self.last_center = None
            self.selected_side = None
            return None, None, None

        if len(candidates) == 1:
            center = candidates[0][0]
            center_model = candidates[0][2]
            self.selected_side = candidates[0][1]
        else:
            self.selected_side = None
            left_center, right_center = candidates[0][0], candidates[1][0]
            consistent = abs(left_center - right_center) \
                <= fill_width * MODEL_CENTER_CONSISTENCY_RATIO
            if consistent:
                center = (left_center + right_center) * 0.5
                left_center_model, right_center_model = candidates[0][2], candidates[1][2]
                center_model = LineModel(
                    (left_center_model.slope + right_center_model.slope) * 0.5,
                    (left_center_model.intercept + right_center_model.intercept) * 0.5,
                    min(left_center_model.inlier_count, right_center_model.inlier_count),
                    max(left_center_model.error, right_center_model.error),
                )
            else:
                reference = self.last_center if self.last_center is not None else center_hint
                if reference is None:
                    center = (left_center + right_center) * 0.5
                    center_model = self.center_model
                else:
                    chosen = min(candidates, key=lambda item: abs(item[0] - reference))
                    center, center_model = chosen[0], chosen[2]
                    self.selected_side = chosen[1]
        self.last_center = float(center)
        if center_model is not None:
            self.center_model = center_model
        return self.last_center, self.left_model, self.right_model


class CrosswalkResult(object):
    def __init__(self):
        self.candidate = False
        self.confidence = 0.0
        self.stop_polygon = None
        self.stop_angle = None
        self.stop_bottom = 0
        self.tracking_polygon = None
        self.tracking_angle = None
        self.tracking_bottom = 0
        self.stripe_polygons = []
        self.loose_polygons = []


class CrosswalkDetector(object):
    def __init__(self):
        self.last_bar = None
        self.bar_only_hits = 0
        self.bar_lost_hits = 0
        self.bar_locked = False

    def lock_current_bar(self):
        self.bar_locked = self.last_bar is not None
        return self.bar_locked

    def unlock_bar(self):
        self.bar_locked = False

    def _drivable_mask(self, shape, lane_points):
        """只保留边线内侧；单边线时以画面中心判断哪一侧可通行。"""
        height, width = shape[:2]
        mask = np.full((height, width), 255, dtype=np.uint8)
        if not lane_points:
            return mask
        tracks = [lane_points] if isinstance(lane_points[0], tuple) else lane_points
        center_x = width * 0.5
        left_tracks, right_tracks = [], []
        for track in tracks:
            if not self._reliable_lane_track(track, shape):
                continue
            median_x = float(np.median([point[0] for point in track]))
            (left_tracks if median_x < center_x else right_tracks).append(track)

        margin = max(4, int(width * LANE_OUTSIDE_MARGIN_RATIO))
        rows = np.arange(height, dtype=np.float64)

        def boundary(track):
            points = np.asarray(track, dtype=np.float64)
            slope, intercept = np.polyfit(points[:, 1], points[:, 0], 1)
            return np.clip(slope * rows + intercept, 0, width - 1)

        if left_tracks:
            left = boundary(max(left_tracks, key=len))
            for y, x in enumerate(left):
                mask[y, :max(0, int(round(x)) - margin)] = 0
        if right_tracks:
            right = boundary(max(right_tracks, key=len))
            for y, x in enumerate(right):
                mask[y, min(width, int(round(x)) + margin + 1):] = 0
        return mask

    def _reliable_lane_track(self, track, shape):
        if len(track) < LANE_TRACK_MIN_POINTS:
            return False
        height, width = shape[:2]
        points = np.asarray(track, dtype=np.float64)
        y_span = float(np.max(points[:, 1]) - np.min(points[:, 1]))
        slope, intercept = np.polyfit(points[:, 1], points[:, 0], 1)
        error = float(np.mean(np.abs(points[:, 0] - (slope * points[:, 1] + intercept))))
        return y_span >= height * LANE_TRACK_MIN_Y_SPAN_RATIO \
            and error <= width * LANE_TRACK_MAX_ERROR_RATIO

    def _lane_models(self, lane_points, shape):
        if not lane_points:
            return []
        tracks = [lane_points] if isinstance(lane_points[0], tuple) else lane_points
        models = []
        for track in tracks:
            if not self._reliable_lane_track(track, shape):
                continue
            points = np.asarray(track, dtype=np.float64)
            slope, intercept = np.polyfit(points[:, 1], points[:, 0], 1)
            models.append(LineModel(slope, intercept, len(points), 0.0))
        return models

    def _bar_matches_lane(self, bar, lane_models, width,
                          allow_strong_override=False):
        if self._detached_strong_bar_geometry(bar, width):
            return False
        if allow_strong_override and self._strong_bar_geometry(bar, width):
            return False
        if not lane_models:
            return False
        bar_angle = bar["angle"]
        center_x, center_y = bar["center"]
        for model in lane_models:
            lane_angle = normalize_angle(np.degrees(np.arctan2(1.0, model.slope)))
            distance = abs(center_x - model.x_at(center_y))
            if (angle_delta(bar_angle, lane_angle) <= BAR_LANE_PARALLEL_ANGLE
                    and distance <= width * BAR_LANE_DISTANCE_RATIO):
                return True
        return False

    def _stripe_candidates(self, binary):
        height, width = binary.shape[:2]
        candidates = []
        for contour in find_contours(binary):
            area = cv2.contourArea(contour)
            if area < STRIPE_MIN_AREA:
                continue
            (cx, cy), (rw, rh), _ = cv2.minAreaRect(contour)
            long_side = max(rw, rh)
            short_side = max(1.0, min(rw, rh))
            fill = area / float(rw * rh + 1.0)
            ratio = long_side / short_side
            if not (STRIPE_RATIO_MIN <= ratio <= STRIPE_RATIO_MAX
                    and width * STRIPE_SHORT_MIN_RATIO <= short_side
                    <= width * STRIPE_SHORT_MAX_RATIO
                    and height * STRIPE_LONG_MIN_RATIO <= long_side
                    <= height * STRIPE_LONG_MAX_RATIO
                    and fill > STRIPE_MIN_FILL):
                continue
            polygon = np.rint(cv2.boxPoints(((cx, cy), (rw, rh),
                                              cv2.minAreaRect(contour)[2]))).astype(np.int32)
            candidates.append({"center": (float(cx), float(cy)),
                               "long": float(long_side), "short": float(short_side),
                               "bottom": int(np.max(polygon[:, 1])),
                               "angle": polygon_long_angle(polygon),
                               "polygon": polygon.tolist()})
        return candidates

    def _stripe_group(self, candidates, shape):
        height, width = shape[:2]
        best = []
        ordered = sorted(
            [item for item in candidates
             if width * STRIPE_CENTER_X_MIN_RATIO <= item["center"][0]
             <= width * STRIPE_CENTER_X_MAX_RATIO],
            key=lambda item: item["center"][0],
        )

        def neighbors(first, second):
            return (
                0 < second["center"][0] - first["center"][0]
                <= width * STRIPE_GROUP_MAX_GAP_RATIO
                and abs(second["center"][1] - first["center"][1])
                <= height * STRIPE_GROUP_Y_RATIO
                and angle_delta(second["angle"], first["angle"])
                <= STRIPE_GROUP_ANGLE
                and STRIPE_GROUP_SIZE_MIN <= second["long"] / first["long"]
                <= STRIPE_GROUP_SIZE_MAX
                and STRIPE_GROUP_SIZE_MIN <= second["short"] / first["short"]
                <= STRIPE_GROUP_SIZE_MAX
            )

        for start in range(len(ordered)):
            group = [ordered[start]]
            for item in ordered[start + 1:]:
                if neighbors(group[-1], item):
                    group.append(item)
                elif item["center"][0] - group[-1]["center"][0] \
                        > width * STRIPE_GROUP_MAX_GAP_RATIO:
                    break
            if len(group) > len(best):
                best = group
        best.sort(key=lambda item: item["center"][0])
        if (len(best) >= STRIPE_STRONG_COUNT
                and best[-1]["center"][0] - best[0]["center"][0]
                >= width * STRIPE_GROUP_MIN_SPAN):
            return best
        return best if 1 <= len(best) <= 2 else []

    def _bar_polygon(self, binary, line):
        """沿 Hough 线法线寻找实际白色带，校正横条中心和厚度。"""
        height, width = binary.shape[:2]
        x1, y1, x2, y2 = [float(value) for value in line]
        length = float(np.hypot(x2 - x1, y2 - y1))
        angle = normalize_angle(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if length < 1.0:
            return np.rint(cv2.boxPoints(((x1, y1), (1.0, 1.0), angle))).astype(np.int32)

        sample_count = max(20, int(length * 0.8))
        along = np.linspace(0.08, 0.92, sample_count)
        base_x = x1 + (x2 - x1) * along
        base_y = y1 + (y2 - y1) * along
        normal_x = -(y2 - y1) / length
        normal_y = (x2 - x1) / length
        search = max(12, int(width * BAR_THICKNESS_SEARCH_RATIO))
        offsets = np.arange(-search, search + 1, dtype=np.int32)
        occupied = []
        for offset in offsets:
            xs = np.rint(base_x + normal_x * offset).astype(np.int32)
            ys = np.rint(base_y + normal_y * offset).astype(np.int32)
            inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            ratio = 0.0 if not np.any(inside) else float(
                np.mean(binary[ys[inside], xs[inside]] > 0)
            )
            occupied.append(ratio >= BAR_THICKNESS_MIN_OCCUPANCY)

        runs = []
        start = None
        for index, value in enumerate(occupied + [False]):
            if value and start is None:
                start = index
            elif not value and start is not None:
                runs.append((start, index - 1))
                start = None
        if runs:
            first, last = max(runs, key=lambda run: run[1] - run[0])
            center_offset = (float(offsets[first]) + float(offsets[last])) * 0.5
            thickness = max(3.0, float(offsets[last] - offsets[first] + 1))
        else:
            center_offset = 0.0
            thickness = max(8.0, width * BAR_DEFAULT_THICKNESS_RATIO)

        center_x = (x1 + x2) * 0.5 + normal_x * center_offset
        center_y = (y1 + y2) * 0.5 + normal_y * center_offset
        return np.rint(cv2.boxPoints(((center_x, center_y),
                                      (length, thickness), angle))).astype(np.int32)

    def _merge_hough_lines(self, lines, width):
        """合并方向和位置接近的共线片段，避免横条只框住其中一半。"""
        pending = []
        for line in lines:
            x1, y1, x2, y2 = [float(value) for value in line]
            if x2 < x1:
                x1, y1, x2, y2 = x2, y2, x1, y1
            length = float(np.hypot(x2 - x1, y2 - y1))
            angle = normalize_angle(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            pending.append({"line": (x1, y1, x2, y2), "length": length,
                            "angle": angle})
        pending.sort(key=lambda item: item["length"], reverse=True)
        groups = []
        angle_limit = BAR_MERGE_ANGLE
        distance_limit = max(10.0, width * BAR_MERGE_DISTANCE_RATIO)
        gap_limit = width * BAR_MERGE_GAP_RATIO
        for item in pending:
            x1, y1, x2, y2 = item["line"]
            midpoint = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5])
            target = None
            for group in groups:
                direction = group["direction"]
                normal = np.array([-direction[1], direction[0]])
                distance = abs(float(np.dot(midpoint - group["origin"], normal)))
                projections = [float(np.dot(np.array(point) - group["origin"], direction))
                               for point in ((x1, y1), (x2, y2))]
                gap = max(0.0, group["min_t"] - max(projections),
                          min(projections) - group["max_t"])
                if (angle_delta(item["angle"], group["angle"]) <= angle_limit
                        and distance <= distance_limit and gap <= gap_limit):
                    target = group
                    break
            if target is None:
                radians = np.radians(item["angle"])
                direction = np.array([np.cos(radians), np.sin(radians)])
                origin = midpoint
                projections = [float(np.dot(np.array(point) - origin, direction))
                               for point in ((x1, y1), (x2, y2))]
                groups.append({"points": [(x1, y1), (x2, y2)], "origin": origin,
                               "direction": direction, "angle": item["angle"],
                               "min_t": min(projections), "max_t": max(projections)})
                continue
            target["points"].extend([(x1, y1), (x2, y2)])
            points = np.asarray(target["points"], dtype=np.float32)
            vx, vy, ox, oy = [float(value) for value in cv2.fitLine(
                points, cv2.DIST_L2, 0, 0.01, 0.01
            ).reshape(-1)]
            if vx < 0:
                vx, vy = -vx, -vy
            target["origin"] = np.array([ox, oy])
            target["direction"] = np.array([vx, vy])
            target["angle"] = normalize_angle(np.degrees(np.arctan2(vy, vx)))
            projections = np.dot(points - target["origin"], target["direction"])
            target["min_t"], target["max_t"] = float(np.min(projections)), float(np.max(projections))

        merged = []
        for group in groups:
            start = group["origin"] + group["direction"] * group["min_t"]
            end = group["origin"] + group["direction"] * group["max_t"]
            merged.append(tuple(np.rint(np.concatenate((start, end))).astype(np.int32)))
        return merged

    def _hough_bars(self, binary, stripes):
        height, width = binary.shape[:2]
        lines = cv2.HoughLinesP(binary, 1, np.pi / 180.0,
                                threshold=max(28, int(width * BAR_HOUGH_THRESHOLD_RATIO)),
                                minLineLength=max(55, int(width * BAR_HOUGH_MIN_LENGTH_RATIO)),
                                maxLineGap=max(14, int(width * BAR_HOUGH_MAX_GAP_RATIO)))
        bars = []
        if lines is None:
            return bars
        valid_lines = []
        for x1, y1, x2, y2 in lines[:, 0, :]:
            length = float(np.hypot(x2 - x1, y2 - y1))
            angle = normalize_angle(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if (length < width * BAR_HOUGH_MIN_LENGTH_RATIO
                    or abs(angle) > BAR_MAX_ABS_ANGLE):
                continue
            valid_lines.append((x1, y1, x2, y2))
        for x1, y1, x2, y2 in self._merge_hough_lines(valid_lines, width):
            length = float(np.hypot(x2 - x1, y2 - y1))
            angle = normalize_angle(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            polygon = self._bar_polygon(binary, (x1, y1, x2, y2))
            center_x = float(np.mean(polygon[:, 0]))
            center_y = float(np.mean(polygon[:, 1]))
            item = {"center": (center_x, center_y), "length": length, "angle": angle,
                    "bottom": int(np.max(polygon[:, 1])), "polygon": polygon.tolist()}
            matched = []
            min_x, max_x = sorted((x1, x2))
            for stripe in stripes:
                sx = stripe["center"][0]
                if not (min_x - width * BAR_STRIPE_X_MARGIN_RATIO
                        <= sx <= max_x + width * BAR_STRIPE_X_MARGIN_RATIO):
                    continue
                radians = np.radians(angle)
                stop_y = center_y + np.tan(radians) * (sx - center_x)
                stripe_polygon = np.asarray(
                    stripe["polygon"], dtype=np.float32
                )
                stripe_top = float(np.min(stripe_polygon[:, 1]))
                near_bottom = (
                    stripe["bottom"] - height * BAR_STRIPE_Y_ABOVE_RATIO
                    <= stop_y
                    <= stripe["bottom"] + height * BAR_STRIPE_Y_BELOW_RATIO
                )
                near_top = (
                    stripe_top - height * BAR_STRIPE_TOP_ABOVE_RATIO
                    <= stop_y
                    <= stripe_top + height * BAR_STRIPE_TOP_BELOW_RATIO
                )
                if ((near_bottom or near_top)
                        and angle_delta(stripe["angle"], angle)
                        >= BAR_STRIPE_MIN_ANGLE):
                    matched.append(stripe)
            item["matched"] = matched
            bars.append(item)
        return bars

    def _same_bar(self, first, second, shape):
        return same_tracked_bar(first, second, shape)

    def _select_bar(self, bars, shape):
        if not bars:
            return None
        if self.last_bar is not None:
            continuous = [bar for bar in bars if self._same_bar(bar, self.last_bar, shape)]
            if continuous:
                return max(continuous, key=lambda bar: (len(bar["matched"]), bar["length"]))
            if self.bar_locked:
                return None
        return max(bars, key=lambda bar: (len(bar["matched"]), bar["length"],
                                          bar["bottom"]))

    def _bar_only_valid(self, bar, shape):
        height, width = shape[:2]
        polygon = np.asarray(bar["polygon"], dtype=np.float32)
        (_, _), (side_a, side_b), _ = cv2.minAreaRect(polygon)
        thickness = min(side_a, side_b)
        return (
            bar["length"] >= width * BAR_ONLY_MIN_LENGTH_RATIO
            and width * BAR_ONLY_MIN_THICKNESS_RATIO <= thickness
            <= width * BAR_ONLY_MAX_THICKNESS_RATIO
            and bar["bottom"] >= height * BAR_TRACK_MIN_BOTTOM_RATIO
            and abs(bar["angle"]) <= BAR_ONLY_MAX_ABS_ANGLE
            and self._crosses_vehicle_axis(bar, width)
        )

    def _smooth_bar(self, current, previous):
        keep = BAR_TRACK_SMOOTH
        center_x = keep * previous["center"][0] + (1.0 - keep) * current["center"][0]
        center_y = keep * previous["center"][1] + (1.0 - keep) * current["center"][1]
        angle = keep * previous["angle"] + (1.0 - keep) * current["angle"]
        polygon = np.rint(cv2.boxPoints(((center_x, center_y),
                                          (current["length"], self._bar_thickness(current)),
                                          angle))).astype(np.int32)
        smoothed = dict(current)
        smoothed.update({"center": (center_x, center_y), "angle": angle,
                         "bottom": int(np.max(polygon[:, 1])),
                         "polygon": polygon.tolist()})
        return smoothed

    def _bar_thickness(self, bar):
        polygon = np.asarray(bar["polygon"], dtype=np.float32)
        (_, _), (side_a, side_b), _ = cv2.minAreaRect(polygon)
        return max(3.0, min(side_a, side_b))

    def _crosses_vehicle_axis(self, bar, width):
        polygon = np.asarray(bar["polygon"], dtype=np.float32)
        margin = width * BAR_STRONG_AXIS_MARGIN_RATIO
        return (float(np.min(polygon[:, 0])) <= width * 0.5 + margin
                and float(np.max(polygon[:, 0])) >= width * 0.5 - margin)

    def _strong_bar_geometry(self, bar, width):
        matched = bar.get("matched", [])
        if len(matched) < BAR_STRONG_MIN_MATCHED:
            return False
        stripe_xs = [stripe["center"][0] for stripe in matched]
        return (
            max(stripe_xs) - min(stripe_xs)
            >= width * STRIPE_GROUP_MIN_SPAN
            and self._crosses_vehicle_axis(bar, width)
        )

    def _detached_strong_bar_geometry(self, bar, width):
        """只信任位于多根斑马条同一侧、且不贴着条纹端点的强横条。"""
        if not self._strong_bar_geometry(bar, width):
            return False
        center_x, center_y = bar["center"]
        slope = np.tan(np.radians(bar["angle"]))
        sides = {"above": [], "below": []}
        for stripe in bar.get("matched", []):
            polygon = stripe.get("polygon")
            short_side = stripe.get("short")
            if not polygon or short_side is None:
                continue
            if angle_delta(stripe["angle"], bar["angle"]) \
                    < BAR_LANE_OVERRIDE_MIN_ANGLE:
                continue
            stripe_points = np.asarray(polygon, dtype=np.float32)
            stripe_top = float(np.min(stripe_points[:, 1]))
            stripe_bottom = float(np.max(stripe_points[:, 1]))
            stripe_x = float(stripe["center"][0])
            bar_y = center_y + slope * (stripe_x - center_x)
            min_gap = max(3.0, float(short_side) * BAR_LANE_OVERRIDE_GAP_RATIO)
            if bar_y <= stripe_top - min_gap:
                sides["above"].append(stripe)
            elif bar_y >= stripe_bottom + min_gap:
                sides["below"].append(stripe)

        for stripes in sides.values():
            if len(stripes) < BAR_STRONG_MIN_MATCHED:
                continue
            stripe_xs = [stripe["center"][0] for stripe in stripes]
            if max(stripe_xs) - min(stripe_xs) \
                    >= width * STRIPE_GROUP_MIN_SPAN:
                return True
        return False

    def detect(self, binary, lane_points=None,
               allow_strong_lane_override=False):
        result = CrosswalkResult()
        drivable_mask = self._drivable_mask(binary.shape, lane_points)
        detection_binary = cv2.bitwise_and(binary, drivable_mask)
        candidates = self._stripe_candidates(detection_binary)
        stripes = self._stripe_group(candidates, detection_binary.shape)
        bars = self._hough_bars(detection_binary, stripes)
        height, width = binary.shape[:2]

        full_candidates = self._stripe_candidates(binary)
        full_stripes = self._stripe_group(full_candidates, binary.shape)
        lane_models = self._lane_models(lane_points, binary.shape)
        bars = [bar for bar in bars
                if not self._bar_matches_lane(
                    bar, lane_models, width, allow_strong_lane_override
                )]
        if len(full_stripes) >= STRIPE_STRONG_COUNT:
            full_bars = self._hough_bars(binary, full_stripes)
            detached = [bar for bar in full_bars
                        if self._detached_strong_bar_geometry(bar, width)]
            fallback = detached or [bar for bar in full_bars
                                    if self._strong_bar_geometry(bar, width)]
            fallback = [bar for bar in fallback
                        if not self._bar_matches_lane(
                            bar, lane_models, width,
                            allow_strong_lane_override,
                        )]
            if fallback:
                candidates = full_candidates
                stripes = full_stripes
                bars = fallback

        result.loose_polygons = [item["polygon"] for item in candidates]
        result.stripe_polygons = [item["polygon"] for item in stripes]
        front_margin = width * BAR_FRONT_MARGIN_RATIO
        tracking = [bar for bar in bars
                    if bar["bottom"] >= height * BAR_TRACK_MIN_BOTTOM_RATIO
                    and bar["center"][0] + bar["length"] * 0.5 >= width * 0.5 - front_margin
                    and bar["center"][0] - bar["length"] * 0.5 <= width * 0.5 + front_margin]
        selected = self._select_bar(tracking, binary.shape)
        if selected is not None:
            same = self.last_bar is not None and self._same_bar(selected, self.last_bar,
                                                                binary.shape)
            if same:
                selected = self._smooth_bar(selected, self.last_bar)
            self.bar_lost_hits = 0
            self.last_bar = selected
            result.tracking_polygon = selected["polygon"]
            result.tracking_angle = selected["angle"]
            result.tracking_bottom = selected["bottom"]
        else:
            same = False
            self.bar_only_hits = 0
            self.bar_lost_hits += 1
            if self.last_bar is not None and self.bar_lost_hits <= BAR_TRACK_HOLD_FRAMES:
                result.tracking_polygon = self.last_bar["polygon"]
                result.tracking_angle = self.last_bar["angle"]
                result.tracking_bottom = self.last_bar["bottom"]
            else:
                self.last_bar = None

        enough_stripes = len(stripes) >= STRIPE_STRONG_COUNT
        strong = (selected is not None and enough_stripes
                  and self._strong_bar_geometry(selected, width))
        if selected is not None and not strong and self._bar_only_valid(selected, binary.shape):
            self.bar_only_hits = self.bar_only_hits + 1 if same else 1
        elif not strong:
            self.bar_only_hits = 0
        bar_only_confirmed = self.bar_only_hits >= BAR_ONLY_STABLE_FRAMES
        locked_confirmed = self.bar_locked and selected is not None
        locked_held = (self.bar_locked and selected is None
                       and result.tracking_polygon is not None)
        if selected is not None and (strong or bar_only_confirmed or locked_confirmed):
            result.candidate = True
            result.confidence = (min(1.0, 0.55 + len(stripes) * 0.08)
                                 if strong else (0.75 if locked_confirmed else 0.68))
            result.stop_polygon = selected["polygon"]
            result.stop_angle = selected["angle"]
            result.stop_bottom = selected["bottom"]
        elif locked_held:
            result.candidate = True
            result.confidence = 0.62
            result.stop_polygon = result.tracking_polygon
            result.stop_angle = result.tracking_angle
            result.stop_bottom = result.tracking_bottom
        else:
            temporal = min(0.55, self.bar_only_hits * 0.10)
            result.confidence = max(min(0.6, len(stripes) * 0.12), temporal)
        return result


def mask_crosswalk(binary, result, include_loose=False):
    polygons = list(result.stripe_polygons)
    if result.stop_polygon is not None:
        polygons.append(result.stop_polygon)
    if include_loose:
        polygons.extend(result.loose_polygons)
    if not polygons:
        return binary
    mask = np.zeros_like(binary)
    for polygon in polygons:
        cv2.fillConvexPoly(mask, np.asarray(polygon, dtype=np.int32), 255)
    pad = max(5, int(binary.shape[1] * 0.018))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1))
    mask = cv2.dilate(mask, kernel)
    cleaned = binary.copy()
    cleaned[mask > 0] = 0
    return cleaned


def maneuver_exit(entry_cleared, exit_hits, exit_visible, exit_bottom,
                  frame_height,
                  exit_required=EXIT_BAR_FRAMES):
    if not entry_cleared:
        return 0, False
    exit_hits = exit_hits + 1 if exit_visible else max(0, exit_hits - 1)
    near = exit_hits >= exit_required \
        and exit_bottom >= float(frame_height) * STOP_NEAR_RATIO
    return exit_hits, near


def approach_next_state(visible, bottom, frame_height, lost_hits):
    if lost_hits > LOST_LIMIT:
        return "FOLLOW"
    if visible and bottom >= float(frame_height) * STOP_NEAR_RATIO:
        return "ALIGN"
    return None


def alignment_next_state(angle, visible, align_hits, lost_hits, elapsed,
                         stripe_count):
    if align_hits >= ALIGN_STABLE_FRAMES:
        return "MANEUVER"
    if lost_hits > LOST_LIMIT:
        return "MANEUVER"
    if elapsed <= ALIGN_TIMEOUT:
        return None
    safe_timeout = (
        visible and angle is not None
        and abs(float(angle)) <= ALIGN_ENTRY_MAX_ANGLE
        and stripe_count >= ALIGN_ENTRY_MIN_STRIPES
    )
    if safe_timeout:
        return "MANEUVER"
    if visible and angle is not None and stripe_count >= ALIGN_ENTRY_MIN_STRIPES:
        return None
    return None


def exit_alignment_next_state(align_hits, lost_hits, elapsed):
    if align_hits >= ALIGN_STABLE_FRAMES:
        return "FOLLOW"
    if lost_hits >= EXIT_ALIGN_LOST_FRAMES:
        return "FOLLOW"
    return None


def maneuver_timeout_exits_to_follow(elapsed):
    return float(elapsed) >= MANEUVER_MAX_TIME


def wait_recovery_state(angle, visible, recover_hits, stripe_count):
    if visible and angle is not None and stripe_count >= ALIGN_ENTRY_MIN_STRIPES \
            and abs(float(angle)) > ALIGN_ENTRY_MAX_ANGLE:
        return "ALIGN"
    safe = (
        visible and angle is not None
        and abs(float(angle)) <= ALIGN_ENTRY_MAX_ANGLE
        and stripe_count >= ALIGN_ENTRY_MIN_STRIPES
    )
    return "MANEUVER" if safe and recover_hits >= WAIT_RECOVER_FRAMES else None


class BinaryVision(object):
    def __init__(self, black_v_max=BLACK_V_MAX):
        self.black_v_max = int(black_v_max)

    def apply(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color = cv2.inRange(hsv, np.array([0, 0, 0], np.uint8),
                            np.array([180, 255, self.black_v_max], np.uint8))
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV,
                                         ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C)
        binary = cv2.bitwise_and(color, adaptive)
        kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
        return cv2.morphologyEx(cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel),
                               cv2.MORPH_CLOSE, kernel)
