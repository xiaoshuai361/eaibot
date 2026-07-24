#!/usr/bin/env python3
"""Live YOLO preview for tagless blocks, with FPS drawn on the image."""

import argparse
import math
import os
import sys
from time import perf_counter

import cv2
import numpy as np


DEFAULT_MODEL = (
    "/home/eaibot/handeye-calib/src/model/"
    "Block_yolov8n_640_best.pt"
)
DEFAULT_OUTPUT = ""
IMAGE_SUFFIXES = (".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Preview tagless-block YOLO detections with processing FPS"
    )
    parser.add_argument(
        "--source",
        default="0",
        help="OpenCV camera index such as 0, or an image/video path",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Optional path for saving the last preview frame; empty means no save",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=15,
        help="Print grasp-point detections every N frames; 0 disables printing",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open a GUI window; useful for remote/headless checks",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames; 0 means until q/Esc/end-of-stream",
    )
    return parser.parse_args(argv)


def parse_source(raw_source):
    if not isinstance(raw_source, str) or not raw_source.strip():
        raise ValueError("--source must be a camera index or non-empty path")
    source = raw_source.strip()
    if source.isdigit():
        return int(source)
    if source.startswith("-") and source[1:].isdigit():
        raise ValueError("camera index must be non-negative")
    return source


def _finite_nonnegative(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("%s must be finite and non-negative" % label)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError("%s must be finite and non-negative" % label)
    return number


def annotate_fps(image_bgr, display_fps, inference_ms):
    image = np.asarray(image_bgr)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("preview image must be a BGR uint8 array")
    display_fps = _finite_nonnegative(display_fps, "display FPS")
    inference_ms = _finite_nonnegative(inference_ms, "inference milliseconds")

    output = image.copy()
    text = "FPS %.1f | inference %.1f ms" % (display_fps, inference_ms)
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
    )
    bottom = output.shape[0] - 6
    top = bottom - text_height - baseline - 12
    cv2.rectangle(
        output,
        (6, top),
        (18 + text_width, bottom),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        output,
        text,
        (12, bottom - baseline - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def validate_args(args):
    if not math.isfinite(args.confidence) or not 0.0 < args.confidence <= 1.0:
        raise ValueError("--confidence must be finite and in (0, 1]")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be positive")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be non-negative")
    if args.print_every < 0:
        raise ValueError("--print-every must be non-negative")
    return args


def _is_still_image(source):
    return isinstance(source, str) and source.lower().endswith(IMAGE_SUFFIXES)


def load_model(model_path):
    if not os.path.isfile(model_path):
        raise RuntimeError(
            "Model file does not exist or is not a regular file: %s" % model_path
        )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Could not import ultralytics: %s" % exc)
    try:
        return YOLO(model_path, task="detect")
    except Exception as exc:
        raise RuntimeError("Could not load YOLO model: %s" % exc)


def _single_box_scalar(value):
    try:
        values = value.tolist()
    except AttributeError:
        values = value
    if isinstance(values, (list, tuple)):
        if len(values) != 1:
            raise ValueError("YOLO scalar field must contain one value")
        values = values[0]
    number = float(values)
    if not math.isfinite(number):
        raise ValueError("YOLO scalar field must be finite")
    return number


def _single_box_xyxy(value):
    try:
        values = value.tolist()
    except AttributeError:
        values = value
    if (
        isinstance(values, (list, tuple))
        and len(values) == 1
        and isinstance(values[0], (list, tuple))
    ):
        values = values[0]
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError("YOLO box must contain four coordinates")
    box = [float(item) for item in values]
    if not all(math.isfinite(item) for item in box):
        raise ValueError("YOLO box coordinates must be finite")
    return box


def _class_name(names, class_id):
    if isinstance(names, dict):
        return names.get(class_id, str(class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return names[class_id]
    return str(class_id)


def locate_grasp_points(result):
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    points = []
    for box in boxes:
        try:
            xyxy = _single_box_xyxy(getattr(box, "xyxy", None))
            class_id = int(_single_box_scalar(getattr(box, "cls", None)))
            confidence = _single_box_scalar(getattr(box, "conf", None))
            center = ((xyxy[0] + xyxy[2]) * 0.5, (xyxy[1] + xyxy[3]) * 0.5)
            points.append(
                {
                    "ok": True,
                    "box": xyxy,
                    "class_id": class_id,
                    "class_name": _class_name(getattr(result, "names", {}), class_id),
                    "confidence": confidence,
                    "center": center,
                }
            )
        except ValueError as exc:
            points.append(
                {
                    "ok": False,
                    "error": str(exc),
                }
            )
    return points


def annotate_grasp_points(image_bgr, grasp_points):
    output = image_bgr.copy()
    for point in grasp_points:
        if not point.get("ok"):
            continue
        box = np.rint(point["box"]).astype(np.int32)
        center_x = int(math.floor(point["center"][0] + 0.5))
        center_y = int(math.floor(point["center"][1] + 0.5))
        cv2.rectangle(
            output,
            (box[0], box[1]),
            (box[2], box[3]),
            (0, 255, 0),
            2,
        )
        cv2.drawMarker(
            output,
            (center_x, center_y),
            (255, 0, 0),
            cv2.MARKER_CROSS,
            18,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            "grasp (%d,%d)" % (center_x, center_y),
            (center_x + 8, max(18, center_y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return output


def format_detection_summary(frame_count, grasp_points):
    detected = [point for point in grasp_points if point.get("ok")]
    parts = ["frame=%d detected=%d" % (frame_count, len(detected))]
    for point in detected:
        parts.append(
            "%s(conf=%.3f,u=%.1f,v=%.1f)"
            % (
                point["class_name"],
                point["confidence"],
                point["center"][0],
                point["center"][1],
            )
        )
    return " ".join(parts)


def print_grasp_points(frame_count, grasp_points):
    print(format_detection_summary(frame_count, grasp_points))


def run_preview(args):
    args = validate_args(args)
    source = parse_source(args.source)
    model = load_model(args.model)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(
            "Could not open source %r. If Astra ROS is using the camera, stop "
            "that node before opening the same /dev/video device directly." % source
        )

    still_image = _is_still_image(source)
    smoothed_fps = None
    last_annotated = None
    frame_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                if frame_count == 0:
                    raise RuntimeError("Source opened but did not return a frame")
                break

            started = perf_counter()
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.confidence,
                device=args.device,
                verbose=False,
            )
            elapsed = max(perf_counter() - started, 1e-9)
            if len(results) != 1:
                raise RuntimeError("YOLO preview expected exactly one frame result")
            result = results[0]
            instant_fps = 1.0 / elapsed
            smoothed_fps = (
                instant_fps
                if smoothed_fps is None
                else 0.85 * smoothed_fps + 0.15 * instant_fps
            )
            speed = getattr(result, "speed", {}) or {}
            inference_ms = speed.get("inference", elapsed * 1000.0)
            frame_count += 1
            grasp_points = locate_grasp_points(result)
            last_annotated = annotate_fps(
                annotate_grasp_points(result.plot(labels=True, conf=True), grasp_points),
                smoothed_fps,
                inference_ms,
            )
            if args.print_every and (
                still_image or frame_count == 1 or frame_count % args.print_every == 0
            ):
                print_grasp_points(frame_count, grasp_points)

            if not args.no_display:
                cv2.imshow("Tagless Block YOLO Preview - q/Esc to quit", last_annotated)
                key = cv2.waitKey(0 if still_image else 1) & 0xFF
                if key in (27, ord("q")) or still_image:
                    break
            elif still_image:
                break

            if args.max_frames and frame_count >= args.max_frames:
                break
    finally:
        capture.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    if last_annotated is None:
        raise RuntimeError("No annotated preview frame was produced")
    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        if not os.path.isdir(output_dir):
            raise RuntimeError("Output directory does not exist: %s" % output_dir)
        if not cv2.imwrite(args.output, last_annotated):
            raise RuntimeError("Could not save preview image: %s" % args.output)
        print("Saved last preview frame: %s" % args.output)
    return 0


def main(argv=None):
    return run_preview(parse_args(argv))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.stderr.write("Preview interrupted\n")
        sys.exit(1)
    except Exception as exc:
        sys.stderr.write("Error: %s\n" % exc)
        sys.exit(1)
