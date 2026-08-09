#!/usr/bin/env python3
"""Python 3 ONNX detector parent for monocular tagless-block grasping."""

from __future__ import print_function

import argparse
import json
import math
import os
import signal
import shutil
import subprocess
import sys

from block_mono_vision import (
    DEFAULT_CONFIG,
    DEFAULT_CONFIG_PATH as PACKAGED_CONFIG_PATH,
    DEFAULT_TARGET_CLASSES,
    OnnxYoloDetector,
    is_detection_usable,
    load_config,
    normalize_config,
    parse_target_sequence,
    resolve_target_alias,
    select_target_detection,
    target_aliases,
)


DEFAULT_CONFIG_PATH = "/home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml"
DEFAULT_ARM_SCRIPT = "/home/eaibot/handeye-calib/src/mirobot_pick_test.py"
DEFAULT_BLOCK_PRESET_FILE = (
    "/home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json"
)
DEFAULT_PYTHON2 = "/usr/bin/python2"
NORMAL_CHILD_TIMEOUT = 180.0
STOP_CHILD_TIMEOUT = 3.0


class DetectorError(RuntimeError):
    pass


def ensure_config_file(path):
    if os.path.isfile(path):
        return False
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    shutil.copyfile(PACKAGED_CONFIG_PATH, path)
    return True


def _normalize_signed_args(argv):
    signed_options = {
        "--known-z-mm",
        "--arm-timeout",
    }
    normalized = []
    index = 0
    while index < len(argv):
        token = argv[index]
        next_value = argv[index + 1] if index + 1 < len(argv) else None
        if next_value is not None and token in signed_options and next_value.startswith("-"):
            normalized.append(token + "=" + next_value)
            index += 2
            continue
        normalized.append(token)
        index += 1
    return normalized


def parse_args(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Run ONNX YOLO monocular localization/grasp for tagless blocks"
    )
    parser.add_argument(
        "--target",
        choices=sorted(target_aliases(DEFAULT_CONFIG)),
        help="Target to localize/grasp. Omit in --dry-run to print every detected block.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model", default=DEFAULT_CONFIG["model_path"])
    parser.add_argument("--python2", default=DEFAULT_PYTHON2)
    parser.add_argument("--arm-script", default=DEFAULT_ARM_SCRIPT)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--live-preview", action="store_true")
    action.add_argument("--calib-record", action="store_true")
    action.add_argument("--teach-block-grasp", action="store_true")
    action.add_argument("--teach-block-place", action="store_true")
    action.add_argument("--teach-block-idle", action="store_true")
    action.add_argument("--teach-block-carry", action="store_true")
    action.add_argument("--preview-taught-block", action="store_true")
    action.add_argument("--stop-at-taught-pre-grasp", action="store_true")
    action.add_argument("--run-taught-block", action="store_true")
    action.add_argument("--run-chassis-sequence", action="store_true")
    parser.add_argument(
        "--sequence",
        default="1,2,3,4",
        help="Comma-separated target IDs or names for chassis/place teaching sequence.",
    )
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--fail-on-skip", action="store_true")
    parser.add_argument("--wait-key-between-targets", action="store_true")
    parser.add_argument("--align-only", action="store_true")
    parser.add_argument("--skip-startup-home", action="store_true")
    parser.add_argument("--preset-file", default=DEFAULT_BLOCK_PRESET_FILE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reset-pickup-model", action="store_true")
    parser.add_argument("--known-z-mm", type=float)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--preview-hz", type=float, default=1.0)
    parser.add_argument("--pregrasp-distance-mm", type=float)
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Override config confidence_min for this run, such as 0.55",
    )
    parser.add_argument("--show-rgb", action="store_true")
    parser.add_argument("--arm-timeout", type=float, default=NORMAL_CHILD_TIMEOUT)
    return parser.parse_args(_normalize_signed_args(raw_argv))


def _finite(value, option):
    if not math.isfinite(float(value)):
        raise ValueError("%s must be finite" % option)


def selected_action(args):
    actions = [
        ("dry_run", bool(args.dry_run)),
        ("live_preview", bool(args.live_preview)),
        ("calib_record", bool(args.calib_record)),
        ("teach_block_grasp", bool(args.teach_block_grasp)),
        ("teach_block_place", bool(args.teach_block_place)),
        ("teach_block_idle", bool(args.teach_block_idle)),
        ("teach_block_carry", bool(args.teach_block_carry)),
        ("preview_taught_block", bool(args.preview_taught_block)),
        ("stop_at_taught_pre_grasp", bool(args.stop_at_taught_pre_grasp)),
        ("run_taught_block", bool(args.run_taught_block)),
        ("run_chassis_sequence", bool(args.run_chassis_sequence)),
    ]
    enabled = [name for name, is_enabled in actions if is_enabled]
    if len(enabled) != 1:
        raise ValueError(
            "choose exactly one action: --dry-run, --live-preview, --calib-record, "
            "a teach action, --preview-taught-block, "
            "--stop-at-taught-pre-grasp, --run-taught-block, "
            "or --run-chassis-sequence"
        )
    return enabled[0]


def child_wait_timeout(args):
    if selected_action(args) in (
        "teach_block_grasp",
        "teach_block_place",
        "teach_block_idle",
        "teach_block_carry",
    ):
        return None
    return args.arm_timeout


def validate_runtime_args(args, config):
    action = selected_action(args)
    if args.frames is not None:
        if isinstance(args.frames, bool) or args.frames <= 0:
            raise ValueError("--frames must be a positive integer")
    if args.confidence is not None:
        _finite(args.confidence, "--confidence")
        if not 0.0 < args.confidence <= 1.0:
            raise ValueError("--confidence must be in (0, 1]")
    if args.arm_timeout <= 0.0:
        raise ValueError("--arm-timeout must be positive")
    _finite(args.arm_timeout, "--arm-timeout")
    _finite(args.preview_hz, "--preview-hz")
    if args.preview_hz <= 0.0:
        raise ValueError("--preview-hz must be positive")
    if args.pregrasp_distance_mm is not None:
        _finite(args.pregrasp_distance_mm, "--pregrasp-distance-mm")
        if args.pregrasp_distance_mm <= 0.0:
            raise ValueError("--pregrasp-distance-mm must be positive")
    if args.known_z_mm is not None:
        _finite(args.known_z_mm, "--known-z-mm")
        if args.known_z_mm <= 0.0:
            raise ValueError("--known-z-mm must be positive")
    if action == "calib_record" and args.known_z_mm is None:
        raise ValueError("--calib-record requires --known-z-mm")
    if args.reset_pickup_model and action != "teach_block_grasp":
        raise ValueError("--reset-pickup-model requires --teach-block-grasp")
    if args.align_only and action != "run_chassis_sequence":
        raise ValueError("--align-only requires --run-chassis-sequence")
    if args.wait_key_between_targets and action != "run_chassis_sequence":
        raise ValueError(
            "--wait-key-between-targets requires --run-chassis-sequence")
    if args.max_targets is not None:
        if action != "run_chassis_sequence":
            raise ValueError("--max-targets requires --run-chassis-sequence")
        target_count = len(parse_target_sequence(args.sequence, config))
        if not 1 <= args.max_targets <= target_count:
            raise ValueError(
                "--max-targets must be between 1 and the sequence length")
    if args.fail_on_skip and action != "run_chassis_sequence":
        raise ValueError("--fail-on-skip requires --run-chassis-sequence")
    if args.skip_startup_home and action != "run_chassis_sequence":
        raise ValueError("--skip-startup-home requires --run-chassis-sequence")
    targetless_actions = (
        "dry_run", "live_preview", "teach_block_place", "teach_block_idle",
        "teach_block_carry", "run_chassis_sequence")
    if action not in targetless_actions and args.target is None:
        raise ValueError("--target is required except for all-target --dry-run")

    method = str((config or {}).get("distance_method", "theory")).lower()
    if action in (
        "teach_block_grasp",
        "preview_taught_block",
        "stop_at_taught_pre_grasp",
        "run_taught_block",
        "run_chassis_sequence",
    ) and method == "theory":
        raise ValueError(
            "distance_method=theory is allowed for dry-run/calibration only; "
            "motion requires calibrated or fixed_plane distance"
        )
    return args


def read_message(stream):
    line = stream.readline()
    if line == "":
        raise EOFError("detector request stream reached EOF")
    try:
        payload = json.loads(line)
    except (TypeError, ValueError) as exc:
        raise DetectorError("Malformed detector request: %s" % exc)
    if not isinstance(payload, dict):
        raise DetectorError("Detector request must be a JSON object")
    return payload


def write_message(stream, payload):
    try:
        message = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise DetectorError("Detector response is not JSON serializable: %s" % exc)
    stream.write(message + "\n")
    stream.flush()


def _request_id(payload):
    request_id = payload.get("id")
    if isinstance(request_id, bool) or not isinstance(request_id, int) or request_id < 0:
        raise DetectorError("Request id must be a non-negative integer")
    return request_id


def _handle_request(detector, config, payload):
    request_id = _request_id(payload)
    target = payload.get("target")
    if target is not None and target not in DEFAULT_TARGET_CLASSES:
        raise DetectorError("Unknown target: %r" % target)
    image_path = payload.get("image_path")
    if not isinstance(image_path, str) or not os.path.isfile(image_path):
        raise DetectorError("Image path does not exist or is not a regular file")

    detections = detector.detect_path(image_path)
    if target is None:
        return _all_detections_response(request_id, detections, config)
    selected = select_target_detection(detections, target, config)
    usable, reason = is_detection_usable(selected, config)
    if not usable:
        raise DetectorError(reason)

    return {
        "id": request_id,
        "ok": True,
        "target": target,
        "class_id": int(selected["class_id"]),
        "class_name": selected.get("class_name", str(selected["class_id"])),
        "confidence": float(selected["confidence"]),
        "box": selected["box"],
    }


def _target_name_by_class_id(config):
    lookup = {}
    for target, metadata in normalize_config(config).get("target_classes", {}).items():
        lookup[int(metadata["class_id"])] = (target, metadata["class_name"])
    return lookup


def _all_detections_response(request_id, detections, config):
    lookup = _target_name_by_class_id(config)
    best_by_target = {}
    for detection in detections:
        try:
            class_id = int(detection.get("class_id"))
        except (TypeError, ValueError):
            continue
        if class_id not in lookup:
            continue
        target, class_name = lookup[class_id]
        item = dict(detection)
        item["target"] = target
        item["class_id"] = class_id
        item["class_name"] = class_name
        usable, _reason = is_detection_usable(item, config)
        if not usable:
            continue
        current = best_by_target.get(target)
        if current is None or float(item["confidence"]) > float(current["confidence"]):
            best_by_target[target] = item
    detections_out = [
        {
            "target": target,
            "class_id": int(item["class_id"]),
            "class_name": item.get("class_name", str(item["class_id"])),
            "confidence": float(item["confidence"]),
            "box": item["box"],
        }
        for target, item in sorted(
            best_by_target.items(),
            key=lambda entry: int(entry[1]["class_id"]),
        )
    ]
    if not detections_out:
        raise DetectorError("No usable YOLO detections")
    return {
        "id": request_id,
        "ok": True,
        "target": "all",
        "detections": detections_out,
    }


def serve_requests(detector, config, request_stream, response_stream):
    config = normalize_config(config)
    while True:
        try:
            payload = read_message(request_stream)
        except EOFError:
            return
        try:
            response = _handle_request(detector, config, payload)
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit, BrokenPipeError)):
                raise
            request_id = payload.get("id") if isinstance(payload, dict) else None
            response = {"id": request_id, "ok": False, "error": str(exc)}
        write_message(response_stream, response)


def build_child_command(args, request_fd, response_fd):
    command = [
        args.python2,
        args.arm_script,
        "--mode",
        "block_mono",
        "--detector-request-fd",
        str(request_fd),
        "--detector-response-fd",
        str(response_fd),
        "--supervisor-pid",
        str(os.getpid()),
        "--config",
        args.config,
    ]
    if args.target is not None:
        command += ["--block-target", args.target]
    if args.dry_run:
        command.append("--dry-run")
    if args.live_preview:
        command.append("--live-preview")
    if args.calib_record:
        command.append("--calib-record")
    if args.teach_block_grasp:
        command.append("--teach-block-grasp")
    if args.teach_block_place:
        command.append("--teach-block-place")
    if args.teach_block_idle:
        command.append("--teach-block-idle")
    if args.teach_block_carry:
        command.append("--teach-block-carry")
    if args.preview_taught_block:
        command.append("--preview-taught-block")
    if args.stop_at_taught_pre_grasp:
        command.append("--stop-at-taught-pre-grasp")
    if args.run_taught_block:
        command.append("--run-taught-block")
    if args.run_chassis_sequence:
        command.append("--run-chassis-sequence")
    if args.run_chassis_sequence or (args.teach_block_place and args.target is None):
        command += ["--sequence", args.sequence]
    if args.max_targets is not None:
        command += ["--max-targets", str(args.max_targets)]
    if args.fail_on_skip:
        command.append("--fail-on-skip")
    if args.wait_key_between_targets:
        command.append("--wait-key-between-targets")
    if args.align_only:
        command.append("--align-only")
    if args.skip_startup_home:
        command.append("--skip-startup-home")
    if args.preset_file:
        command += ["--preset-file", args.preset_file]
    if args.overwrite:
        command.append("--overwrite")
    if args.reset_pickup_model:
        command.append("--reset-pickup-model")
    if args.known_z_mm is not None:
        command += ["--known-z-mm", str(args.known_z_mm)]
    if args.frames is not None:
        command += ["--frames", str(args.frames)]
    command += ["--preview-hz", str(args.preview_hz)]
    if args.pregrasp_distance_mm is not None:
        command += ["--pregrasp-distance-mm", str(args.pregrasp_distance_mm)]
    if args.confidence is not None:
        command += ["--confidence", str(args.confidence)]
    if args.show_rgb:
        command.append("--show-rgb")
    return command


def build_child_env():
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("CONDA") or key in ("PYTHONHOME",):
            env.pop(key, None)
    if env.get("PYTHONPATH"):
        parts = [part for part in env["PYTHONPATH"].split(os.pathsep) if "conda" not in part]
        env["PYTHONPATH"] = os.pathsep.join(parts)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return env


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
    if child is None:
        return None
    try:
        if child.poll() is not None:
            return None
        child.terminate()
    except Exception as exc:
        return exc
    try:
        child.wait(timeout=STOP_CHILD_TIMEOUT)
    except subprocess.TimeoutExpired:
        try:
            child.kill()
            child.wait(timeout=STOP_CHILD_TIMEOUT)
        except Exception as exc:
            return exc
    except Exception as exc:
        return exc
    return None


def request_shutdown(signum, _frame):
    raise KeyboardInterrupt("received signal %d" % signum)


def install_shutdown_handlers():
    signal.signal(signal.SIGTERM, request_shutdown)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, request_shutdown)


def run_parent(args):
    created_config = ensure_config_file(args.config)
    if created_config:
        sys.stderr.write("Created default config: %s\n" % args.config)
    config = load_config(args.config)
    if args.target is not None:
        args.target = resolve_target_alias(args.target, config)
    if args.run_chassis_sequence or (args.teach_block_place and args.target is None):
        args.sequence = ",".join(parse_target_sequence(args.sequence, config))
    if args.model:
        config["model_path"] = args.model
    if args.confidence is not None:
        config["confidence_min"] = args.confidence
    args = validate_runtime_args(args, config)
    action = selected_action(args)
    detector = None
    if action not in ("teach_block_place", "teach_block_idle", "teach_block_carry"):
        detector = OnnxYoloDetector(config["model_path"], config)

    request_read = request_write = response_read = response_write = None
    request_stream = response_stream = None
    child = None
    try:
        request_read, request_write = os.pipe()
        response_read, response_write = os.pipe()
        child = subprocess.Popen(
            build_child_command(args, request_write, response_read),
            pass_fds=(request_write, response_read),
            close_fds=True,
            env=build_child_env(),
            start_new_session=True,
        )
        close_fd_safely(request_write)
        request_write = None
        close_fd_safely(response_read)
        response_read = None

        request_stream = os.fdopen(request_read, "r")
        request_read = None
        response_stream = os.fdopen(response_write, "w")
        response_write = None
        serve_requests(detector, config, request_stream, response_stream)

        return_code = child.wait(timeout=child_wait_timeout(args))
        if return_code != 0:
            raise RuntimeError("Arm child exited with status %d" % return_code)
        return 0
    except subprocess.TimeoutExpired:
        sys.stderr.write(
            "CRITICAL: arm child exceeded timeout; hardware state may be unknown.\n"
        )
        return 1
    finally:
        _close_stream_safely(request_stream)
        _close_stream_safely(response_stream)
        close_fd_safely(request_read)
        close_fd_safely(request_write)
        close_fd_safely(response_read)
        close_fd_safely(response_write)
        if child is not None and child.poll() is None:
            cleanup_error = stop_child(child)
            if cleanup_error is not None:
                sys.stderr.write("Failed to stop child: %s\n" % cleanup_error)


def main(argv=None):
    return run_parent(parse_args(argv))


if __name__ == "__main__":
    install_shutdown_handlers()
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.stderr.write("Interrupted\n")
        sys.exit(1)
    except Exception as exc:
        sys.stderr.write("Error: %s\n" % exc)
        sys.exit(1)
