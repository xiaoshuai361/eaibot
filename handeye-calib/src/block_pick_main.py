#!/usr/bin/env python3
"""Python 3 YOLO entry point for tagless block grasp requests."""

import argparse
import math
import os
import subprocess
import sys

from block_detector_protocol import read_message, write_message


TARGET_CLASSES = {
    "power": {"class_id": 0, "class_name": "Emergency power supply device"},
    "fire": {"class_id": 1, "class_name": "Fire extinguishing device"},
    "gas": {"class_id": 2, "class_name": "Gas purification device"},
    "support": {"class_id": 3, "class_name": "Structural support device"},
}
TARGET_CLASS_IDS = {
    target: metadata["class_id"] for target, metadata in TARGET_CLASSES.items()
}

DEFAULT_MODEL = "/home/eaibot/models/Block_yolov8n_640/Block_yolov8n_640_best.pt"
DEFAULT_ARM_SCRIPT = "/home/eaibot/handeye-calib/src/mirobot_pick_test.py"
NORMAL_CHILD_TIMEOUT = 30.0
STOP_CHILD_TIMEOUT = 3.0


class DetectionError(RuntimeError):
    pass


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Detect and grasp one tagless supply block")
    parser.add_argument("--target", required=True, choices=sorted(TARGET_CLASSES))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--python2", default="python2")
    parser.add_argument("--arm-script", default=DEFAULT_ARM_SCRIPT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-at-pre-grasp", action="store_true")
    parser.add_argument("--tool-offset", type=float)
    parser.add_argument("--tool-axis", choices=("x", "-x", "y", "-y", "z", "-z"))
    parser.add_argument("--max-tool-camera-angle-deg", type=float, default=20.0)
    parser.add_argument("--approach-gap", type=float, default=0.03)
    parser.add_argument("--velocity-scale", type=float, default=0.05)
    parser.add_argument("--acceleration-scale", type=float, default=0.05)
    parser.add_argument("--debug-image")
    if argv is not None:
        # argparse treats values such as ``-inf`` and ``-x`` as new options.
        # Join these option/value pairs so all documented signed values reach
        # the validator and tool-axis choices correctly.
        signed_value_options = {
            "--confidence", "--tool-offset", "--tool-axis",
            "--max-tool-camera-angle-deg", "--approach-gap",
            "--velocity-scale", "--acceleration-scale",
        }
        normalized = []
        index = 0
        argv = list(argv)
        while index < len(argv):
            token = argv[index]
            if (
                token in signed_value_options
                and index + 1 < len(argv)
                and argv[index + 1].startswith("-")
            ):
                normalized.append(token + "=" + argv[index + 1])
                index += 2
                continue
            normalized.append(token)
            index += 1
        argv = normalized
    return parser.parse_args(argv)


def _finite(value, option):
    if not math.isfinite(value):
        raise ValueError("%s must be finite" % option)


def validate_runtime_args(args):
    numeric = (
        (args.confidence, "--confidence"),
        (args.max_tool_camera_angle_deg, "--max-tool-camera-angle-deg"),
        (args.approach_gap, "--approach-gap"),
        (args.velocity_scale, "--velocity-scale"),
        (args.acceleration_scale, "--acceleration-scale"),
    )
    if args.tool_offset is not None:
        numeric += ((args.tool_offset, "--tool-offset"),)
    for value, option in numeric:
        _finite(value, option)

    if not 0.0 < args.confidence <= 1.0:
        raise ValueError("--confidence must be in (0, 1]")
    for value, option in (
        (args.velocity_scale, "--velocity-scale"),
        (args.acceleration_scale, "--acceleration-scale"),
    ):
        if not 0.0 < value <= 1.0:
            raise ValueError("%s must be in (0, 1]" % option)
    if not 0.0 < args.approach_gap <= 0.15:
        raise ValueError("--approach-gap must be in (0, 0.15]")
    if args.tool_offset is not None and not 0.0 <= args.tool_offset <= 0.30:
        raise ValueError("--tool-offset must be in [0, 0.30]")
    if not 0.0 < args.max_tool_camera_angle_deg < 90.0:
        raise ValueError("--max-tool-camera-angle-deg must be in (0, 90)")

    if (args.tool_offset is None) != (args.tool_axis is None):
        raise ValueError("--tool-offset and --tool-axis must be provided together")
    if (not args.dry_run or args.stop_at_pre_grasp) and args.tool_offset is None:
        raise ValueError("tool offset and axis are required for arm motion")
    return args


def validate_model_names(names):
    expected = [TARGET_CLASSES[target]["class_name"] for target in TARGET_CLASSES]
    if isinstance(names, dict):
        try:
            actual = [names[index] for index in range(len(names))]
        except (KeyError, TypeError):
            raise DetectionError("Model class metadata must use consecutive integer IDs")
    elif isinstance(names, (list, tuple)):
        actual = list(names)
    else:
        raise DetectionError("Model class metadata must be a dict or list")
    if actual != expected:
        raise DetectionError("Model class metadata does not exactly match TARGET_CLASSES")


def load_model(model_path):
    if not os.path.isfile(model_path):
        raise DetectionError("Model file does not exist or is not a regular file: %s" % model_path)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise DetectionError("Could not import ultralytics: %s" % exc)
    try:
        model = YOLO(model_path)
    except Exception as exc:
        raise DetectionError("Could not load YOLO model: %s" % exc)
    validate_model_names(model.names)
    return model


def _single_scalar(value, field):
    try:
        values = value.tolist()
    except AttributeError:
        values = value
    if isinstance(values, (list, tuple)):
        if len(values) != 1:
            raise DetectionError("Detection %s must contain one value" % field)
        values = values[0]
    try:
        number = float(values)
    except (TypeError, ValueError, OverflowError):
        raise DetectionError("Detection %s is not numeric" % field)
    if not math.isfinite(number):
        raise DetectionError("Detection %s must be finite" % field)
    return number


def _box_coordinates(value):
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
        raise DetectionError("Detection box must have shape [1, 4] or [4]")
    try:
        coordinates = [float(value) for value in values]
    except (TypeError, ValueError, OverflowError):
        raise DetectionError("Detection box coordinates must be numeric")
    if not all(math.isfinite(value) for value in coordinates):
        raise DetectionError("Detection box coordinates must be finite")
    if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
        raise DetectionError("Detection box must have positive width and height")
    return coordinates


def infer_detections(model, image_path, inference_confidence):
    if not math.isfinite(inference_confidence) or not 0.0 < inference_confidence <= 1.0:
        raise DetectionError("Inference confidence must be finite and in (0, 1]")
    try:
        results = model.predict(
            source=image_path,
            imgsz=640,
            conf=inference_confidence,
            verbose=False,
        )
    except Exception as exc:
        raise DetectionError("YOLO inference failed: %s" % exc)
    try:
        result_count = len(results)
    except TypeError:
        raise DetectionError("YOLO inference did not return a result sequence")
    if result_count != 1:
        raise DetectionError("YOLO inference must return exactly one result")

    detections = []
    boxes = getattr(results[0], "boxes", None)
    if boxes is None:
        raise DetectionError("YOLO result has no boxes")
    try:
        iterator = iter(boxes)
    except TypeError:
        raise DetectionError("YOLO boxes are not iterable")
    for box in iterator:
        class_number = _single_scalar(getattr(box, "cls", None), "class_id")
        if not class_number.is_integer():
            raise DetectionError("Detection class_id must be an integer")
        class_id = int(class_number)
        if class_id not in set(TARGET_CLASS_IDS.values()):
            raise DetectionError("Detection class_id is outside model metadata")
        confidence = _single_scalar(getattr(box, "conf", None), "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise DetectionError("Detection confidence must be in [0, 1]")
        detections.append({
            "class_id": class_id,
            "confidence": confidence,
            "box": _box_coordinates(getattr(box, "xyxy", None)),
        })
    return detections


def select_unique_detection(detections, class_id, confidence_threshold):
    matches = [
        detection for detection in detections
        if detection["class_id"] == class_id
        and detection["confidence"] >= confidence_threshold
    ]
    if not matches:
        raise DetectionError("No detection matched the requested class and confidence")
    if len(matches) > 1:
        raise DetectionError("Multiple detections matched the requested class")
    return matches[0]


def _request_id(payload):
    request_id = payload.get("id") if isinstance(payload, dict) else None
    if isinstance(request_id, bool) or not isinstance(request_id, int) or request_id < 0:
        raise DetectionError("Request id must be a non-negative integer")
    return request_id


def _handle_request(model, payload, confidence_threshold):
    request_id = _request_id(payload)
    target = payload.get("target")
    if target not in TARGET_CLASSES:
        raise DetectionError("Unknown target: %r" % target)
    image_path = payload.get("image_path")
    if not isinstance(image_path, str) or not os.path.isfile(image_path):
        raise DetectionError("Image path does not exist or is not a regular file")
    metadata = TARGET_CLASSES[target]
    detections = infer_detections(model, image_path, confidence_threshold)
    selected = select_unique_detection(
        detections, metadata["class_id"], confidence_threshold
    )
    return {
        "id": request_id,
        "ok": True,
        "target": target,
        "class_id": metadata["class_id"],
        "class_name": metadata["class_name"],
        "confidence": selected["confidence"],
        "box": selected["box"],
    }


def serve_requests(model, request_stream, response_stream, confidence_threshold):
    while True:
        try:
            payload = read_message(request_stream)
        except EOFError:
            return
        try:
            response = _handle_request(model, payload, confidence_threshold)
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit, BrokenPipeError)):
                raise
            request_id = payload.get("id") if isinstance(payload, dict) else None
            response = {"id": request_id, "ok": False, "error": str(exc)}
        # Protocol/write failures are process-level failures and must stop service.
        write_message(response_stream, response)


def build_child_command(args, request_fd, response_fd):
    command = [
        args.python2,
        args.arm_script,
        "--mode", "block_grasp",
        "--target", args.target,
        "--detector-request-fd", str(request_fd),
        "--detector-response-fd", str(response_fd),
        "--approach-gap", str(args.approach_gap),
        "--max-tool-camera-angle-deg", str(args.max_tool_camera_angle_deg),
        "--velocity-scale", str(args.velocity_scale),
        "--acceleration-scale", str(args.acceleration_scale),
    ]
    if args.tool_offset is not None:
        command += ["--tool-offset", str(args.tool_offset), "--tool-axis", args.tool_axis]
    if args.debug_image:
        command += ["--debug-image", args.debug_image]
    if args.dry_run:
        command.append("--dry-run")
    if args.stop_at_pre_grasp:
        command.append("--stop-at-pre-grasp")
    return command


def close_fd_safely(fd):
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass


def _close_stream_safely(stream):
    if stream is None:
        return
    try:
        stream.close()
    except Exception:
        pass


def stop_child(child):
    if child is None or child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=STOP_CHILD_TIMEOUT)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=STOP_CHILD_TIMEOUT)


def main(argv=None):
    args = validate_runtime_args(parse_args(argv))
    model = load_model(args.model)

    request_read = request_write = response_read = response_write = None
    request_stream = response_stream = None
    child = None
    try:
        request_read, request_write = os.pipe()
        response_read, response_write = os.pipe()
        command = build_child_command(args, request_write, response_read)
        child = subprocess.Popen(
            command,
            pass_fds=(request_write, response_read),
            close_fds=True,
        )

        close_fd_safely(request_write)
        request_write = None
        close_fd_safely(response_read)
        response_read = None

        request_stream = os.fdopen(request_read, "r")
        request_read = None
        response_stream = os.fdopen(response_write, "w")
        response_write = None
        try:
            serve_requests(model, request_stream, response_stream, args.confidence)
        except BaseException:
            stop_child(child)
            raise

        try:
            return_code = child.wait(timeout=NORMAL_CHILD_TIMEOUT)
        except subprocess.TimeoutExpired:
            stop_child(child)
            raise RuntimeError("Arm child timed out after detector EOF")
        if return_code != 0:
            raise RuntimeError("Arm child exited with status %d" % return_code)
        return 0
    finally:
        _close_stream_safely(request_stream)
        _close_stream_safely(response_stream)
        close_fd_safely(request_read)
        close_fd_safely(request_write)
        close_fd_safely(response_read)
        close_fd_safely(response_write)
        if child is not None and child.poll() is None:
            stop_child(child)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.stderr.write("Interrupted\n")
        sys.exit(1)
    except Exception as exc:
        sys.stderr.write("Error: %s\n" % exc)
        sys.exit(1)
