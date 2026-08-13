#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Python 2 ROS/MoveIt helper for monocular tagless-block suction."""

from __future__ import absolute_import, division, print_function

import argparse
import copy
import fcntl
import json
import math
import os
import signal
import sys
import tempfile
import time

if sys.version_info[0] != 2:
    sys.stderr.write(
        "mirobot_pick_test.py 必须使用 Python 2 运行，因为 ROS Melodic 的 "
        "tf/moveit 模块是 Python 2 版本。\n"
        "无 Tag 新方案请从 Python3 入口启动：\n"
        "python3 /home/eaibot/handeye-calib/src/block_pick_main.py --target fire --dry-run\n"
    )
    sys.exit(1)

import moveit_commander
import rospy
import tf
from geometry_msgs.msg import PoseStamped, Twist
from std_srvs.srv import Trigger

from tag_chassis_align_pick_sequence import (
    compute_drive_command,
    roi_ratio_to_pixels,
)

from block_mono_vision import (
    DEFAULT_CONFIG,
    LocalizationError,
    detection_to_observation,
    deproject_pixel_to_camera_mm,
    draw_debug_detections,
    draw_debug_image,
    estimate_distance_from_box_mm,
    is_detection_usable,
    load_config,
    observation_in_roi,
    parse_target_sequence,
    scale_box_width_for_distance,
    stable_median_observation,
)


try:
    STRING_TYPES = (basestring,)
except NameError:
    STRING_TYPES = (str,)


WRIST_FORWARD_JOINT5 = -1.5709534265016345
BLOCK_PRESET_VERSION = 2
MOTION_SETTLE_SECONDS = DEFAULT_CONFIG["motion_settle_seconds"]
DEFAULT_BLOCK_PRESET_FILE = (
    "/home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json"
)
DEFAULT_DELIVERY_FILE = (
    "/home/eaibot/handeye-calib/config/untagged_delivery_presets.json"
)
MOTION_LOCK_PATH = "/tmp/mirobot_arm_motion.lock"
CONTACT_PROBE_ENABLE_SERVICE = "/mirobot_contact_probe_enable"
CONTACT_STATE_SERVICE = "/mirobot_contact_state"
CONTACT_PROBE_MISS_EXIT_CODE = 4


class TerminationRequested(RuntimeError):
    pass


class ContactProbeMiss(RuntimeError):
    def __init__(self, target):
        self.target = target
        RuntimeError.__init__(
            self, "CONTACT_PROBE_MISS: no contact for target %s." % target)


def raise_termination_requested(signum, _frame):
    raise TerminationRequested("Received termination signal %d." % signum)


def enable_parent_death_signal(expected_parent_pid, libc=None):
    """Ask Linux to terminate this arm helper if its Python3 parent dies."""
    if expected_parent_pid is None:
        return
    expected_parent_pid = int(expected_parent_pid)
    if os.getppid() != expected_parent_pid:
        raise TerminationRequested(
            "Arm supervisor exited before the child initialized.")
    if libc is None:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM) != 0:  # PR_SET_PDEATHSIG
        raise RuntimeError("Could not enable the arm parent-death signal.")
    # Close the race where the parent exits immediately before prctl().
    if os.getppid() != expected_parent_pid:
        raise TerminationRequested(
            "Arm supervisor exited while the child initialized.")


def action_uses_moveit(args):
    if args.mode in ("home", "current_pose", "wrist_forward"):
        return True
    if args.mode != "block_mono":
        return False
    return get_action(args) in (
        "teach_block_pick_place",
        "teach_block_pregrasp",
        "teach_block_place",
        "teach_block_idle",
        "teach_block_carry",
        "teach_building_contact_release",
        "stop_at_taught_pre_grasp",
        "run_taught_block",
        "run_chassis_sequence",
    )


def motion_action_label(args):
    if args.mode == "block_mono":
        return get_action(args)
    return args.mode


def acquire_motion_lock(args, path=MOTION_LOCK_PATH):
    if not action_uses_moveit(args):
        return None
    handle = open(path, "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        handle.seek(0)
        owner = handle.read().strip() or "unknown owner"
        handle.close()
        raise RuntimeError(
            "Another mechanical-arm command is still active (%s). "
            "Stop the old block_pick_main/mirobot_pick_test process first."
            % owner)
    os.ftruncate(handle.fileno(), 0)
    handle.write("pid=%d action=%s\n" % (os.getpid(), motion_action_label(args)))
    handle.flush()
    return handle


def _normalize_signed_args(argv):
    signed_options = set(["--known-z-mm"])
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


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Mirobot helper for home/pump and monocular tagless block grasp"
    )
    parser.add_argument(
        "--mode",
        choices=["home", "pump", "current_pose", "wrist_forward", "block_mono"],
        default="current_pose",
    )
    parser.add_argument("--config", default="/home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml")
    parser.add_argument("--group", default="manipulator")
    parser.add_argument("--base-frame")
    parser.add_argument("--velocity-scale", type=float)
    parser.add_argument("--acceleration-scale", type=float)
    parser.add_argument("--planning-time", type=float)
    parser.add_argument("--tf-timeout", type=float)
    parser.add_argument("--pump-seconds", type=float, default=2.0)
    parser.add_argument("--wrist-forward-joint5", type=float, default=WRIST_FORWARD_JOINT5)

    parser.add_argument("--block-target", choices=sorted(DEFAULT_CONFIG["target_classes"]))
    parser.add_argument("--detector-request-fd", type=int)
    parser.add_argument("--detector-response-fd", type=int)
    parser.add_argument("--supervisor-pid", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--live-preview", action="store_true")
    parser.add_argument("--calib-record", action="store_true")
    parser.add_argument("--teach-block-pick-place", action="store_true")
    parser.add_argument("--teach-block-pregrasp", action="store_true")
    parser.add_argument("--teach-block-place", action="store_true")
    parser.add_argument("--teach-block-idle", action="store_true")
    parser.add_argument("--teach-block-carry", action="store_true")
    parser.add_argument("--teach-building-contact-release", action="store_true")
    parser.add_argument("--preview-taught-block", action="store_true")
    parser.add_argument("--stop-at-taught-pre-grasp", action="store_true")
    parser.add_argument("--run-taught-block", action="store_true")
    parser.add_argument("--run-chassis-sequence", action="store_true")
    parser.add_argument("--sequence", default="1,2,3,4")
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--fail-on-skip", action="store_true")
    parser.add_argument("--result-file")
    # Legacy no-op: continuous no-Tag pickup never waits for keyboard input.
    parser.add_argument(
        "--wait-key-between-targets", action="store_true",
        help=argparse.SUPPRESS)
    parser.add_argument("--align-only", action="store_true")
    parser.add_argument("--skip-startup-home", action="store_true")
    parser.add_argument("--preset-file", default=DEFAULT_BLOCK_PRESET_FILE)
    parser.add_argument("--delivery-file", default=DEFAULT_DELIVERY_FILE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--place-approach-gap", type=float, default=0.02)
    parser.add_argument("--known-z-mm", type=float)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--preview-hz", type=float, default=1.0)
    parser.add_argument("--approach-gap-mm", type=float)
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--show-rgb", action="store_true")
    parser.add_argument("--search-before-chassis", action="store_true")
    parser.add_argument("--search-ready-file")
    parser.add_argument("--search-trigger-file")
    parser.add_argument("--search-release-file")
    parser.add_argument("--search-roi-ratio", default="0.60,0.05,0.98,0.95")
    parser.add_argument("--search-stable-frames", type=int, default=3)
    parser.add_argument("--search-poll-hz", type=float, default=3.0)
    ros_argv = rospy.myargv(argv)[1:]
    return parser.parse_args(_normalize_signed_args(ros_argv))


def finite_scalar(value, option):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise RuntimeError("%s must be finite." % option)
    if math.isnan(number) or math.isinf(number):
        raise RuntimeError("%s must be finite." % option)
    return number


def finite_vector3(values, option):
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise RuntimeError("%s must contain exactly three numbers." % option)
    return tuple(finite_scalar(value, "%s[%d]" % (option, index))
                 for index, value in enumerate(values))


def normalize_vector(values, name):
    x, y, z = finite_vector3(values, name)
    length = math.sqrt(x * x + y * y + z * z)
    if length <= 0.0:
        raise RuntimeError("%s must be non-zero." % name)
    return (x / length, y / length, z / length)


def format_triplet(values):
    return "({:.2f},{:.2f},{:.2f})".format(values[0], values[1], values[2])


def safe_log_text(value):
    try:
        text_type = unicode
    except NameError:
        text_type = str
    if isinstance(value, text_type):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    try:
        return text_type(value)
    except (TypeError, ValueError, UnicodeError):
        return text_type(repr(value))


def ascii_log_text(value):
    encoded = safe_log_text(value).encode("ascii", "backslashreplace")
    if isinstance(encoded, str):
        return encoded
    return encoded.decode("ascii")


def print_utf8(value):
    text = safe_log_text(value)
    if sys.version_info[0] < 3:
        sys.stdout.write(text.encode("utf-8", "replace") + "\n")
    else:
        print(text)


def format_localization_summary(localization):
    return (
        u"目标={target} 置信度={confidence:.3f} "
        u"检测框=({x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}) "
        u"框中心=({u:.2f},{v:.2f}) 框宽px={w:.2f} 框高px={h:.2f} "
        u"距离方法={distance_method} 单目距离Z_mm={z_mm:.2f} "
        u"相机坐标mm={camera_xyz} 机械臂坐标mm={base_xyz}"
    ).format(
        target=safe_log_text(localization["target"]),
        confidence=localization["confidence"],
        x1=localization["box"][0],
        y1=localization["box"][1],
        x2=localization["box"][2],
        y2=localization["box"][3],
        u=localization["u"],
        v=localization["v"],
        w=localization["w"],
        h=localization["h"],
        distance_method=localization["distance_method"],
        z_mm=localization["z_mm"],
        camera_xyz=format_triplet(localization["camera_xyz_mm"]),
        base_xyz=format_triplet(localization["base_xyz_mm"]),
    )


def normalize_quaternion(values):
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise RuntimeError("quaternion must contain x,y,z,w.")
    components = [finite_scalar(value, "quaternion") for value in values]
    length = math.sqrt(sum(value * value for value in components))
    if length <= 0.0:
        raise RuntimeError("quaternion must be non-zero.")
    return [value / length for value in components]


def pose_to_transform(pose_stamped):
    pose = pose_stamped.pose
    return {
        "position": [
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
        ],
        "orientation_xyzw": quaternion_msg_to_tuple(pose.orientation),
    }


def transform_to_pose(frame_id, transform):
    position = finite_vector3(transform.get("position"), "position")
    orientation = normalize_quaternion(transform.get("orientation_xyzw"))
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.header.stamp = rospy.Time.now() if "rospy" in globals() else None
    pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = position
    pose.pose.orientation.x, pose.pose.orientation.y = orientation[:2]
    pose.pose.orientation.z, pose.pose.orientation.w = orientation[2:]
    return pose


def block_anchor_pose_from_localization(localization, config):
    base_frame = localization.get("base_frame") or config["base_frame"]
    xyz_mm = finite_vector3(localization.get("base_xyz_mm"), "base_xyz_mm")
    orientation = config.get("block_anchor_orientation_xyzw")
    if orientation is None:
        orientation = [0.0, 0.0, 0.0, 1.0]
    qx, qy, qz, qw = normalize_quaternion(orientation)
    pose = PoseStamped()
    pose.header.frame_id = base_frame
    pose.header.stamp = rospy.Time.now() if "rospy" in globals() else None
    pose.pose.position.x = xyz_mm[0] * 0.001
    pose.pose.position.y = xyz_mm[1] * 0.001
    pose.pose.position.z = xyz_mm[2] * 0.001
    pose.pose.orientation.x = qx
    pose.pose.orientation.y = qy
    pose.pose.orientation.z = qz
    pose.pose.orientation.w = qw
    return pose


def create_block_pickup_model(taught_pre_grasp_pose, camera_forward_base):
    forward = normalize_vector(camera_forward_base, "camera_forward_base")
    return {
        "orientation_xyzw_base": list(normalize_quaternion(
            quaternion_msg_to_tuple(taught_pre_grasp_pose.pose.orientation))),
        # This axis points from the object toward the camera/safe side.
        "approach_axis_xyz_base": [-forward[0], -forward[1], -forward[2]],
    }


def compute_block_pregrasp_offset(anchor_pose, taught_pre_grasp_pose):
    return [
        float(taught_pre_grasp_pose.pose.position.x - anchor_pose.pose.position.x),
        float(taught_pre_grasp_pose.pose.position.y - anchor_pose.pose.position.y),
        float(taught_pre_grasp_pose.pose.position.z - anchor_pose.pose.position.z),
    ]


def require_block_pickup_model(entry, target):
    model = entry.get("pickup_model")
    if not isinstance(model, dict):
        raise RuntimeError(
            "Target %s has no pickup_model. Re-teach its tagless pick/place."
            % target)
    normalize_quaternion(model.get("orientation_xyzw_base"))
    normalize_vector(model.get("approach_axis_xyz_base"),
                     "approach_axis_xyz_base")
    return model


def require_block_pregrasp_offset(entry, target):
    offset = entry.get("pregrasp_offset_xyz_base")
    if offset is None:
        raise RuntimeError(
            "Target %s has no pregrasp_offset_xyz_base. Re-teach its "
            "tagless pick/place." % target)
    return finite_vector3(offset, "pregrasp_offset_xyz_base")


def require_block_target_entry(preset, target):
    entry = preset.get("targets", {}).get(target)
    if not isinstance(entry, dict):
        raise RuntimeError(
            "Preset has no independently taught data for target %s." % target)
    pickup_model = require_block_pickup_model(entry, target)
    pregrasp_offset = require_block_pregrasp_offset(entry, target)
    if "place_ee_in_base" not in entry:
        raise RuntimeError(
            "Target %s has no independent no-Tag place pose. Re-teach its "
            "tagless pick/place." % target)
    return entry, pickup_model, pregrasp_offset


def require_joint_values(preset, field):
    values = preset.get(field)
    if not isinstance(values, (list, tuple)) or len(values) != 6:
        raise RuntimeError(
            "Preset must contain six %s values; copy them from the Tag preset."
            % field)
    return [finite_scalar(value, field) for value in values]


def compute_taught_block_pregrasp_pose(anchor_pose, pickup_model, offset,
                                       base_frame):
    offset = finite_vector3(offset, "pregrasp_offset_xyz_base")
    orientation = normalize_quaternion(
        pickup_model.get("orientation_xyzw_base"))
    pose = copy.deepcopy(anchor_pose)
    pose.header.frame_id = base_frame
    pose.header.stamp = rospy.Time.now() if "rospy" in globals() else None
    pose.pose.position.x += offset[0]
    pose.pose.position.y += offset[1]
    pose.pose.position.z += offset[2]
    pose.pose.orientation.x = orientation[0]
    pose.pose.orientation.y = orientation[1]
    pose.pose.orientation.z = orientation[2]
    pose.pose.orientation.w = orientation[3]
    return pose


def build_block_backoff_pose(reference_pose, pickup_model, distance_mm,
                             base_frame):
    axis = normalize_vector(
        pickup_model.get("approach_axis_xyz_base"),
        "approach_axis_xyz_base")
    distance_m = finite_scalar(distance_mm, "backoff distance mm") * 0.001
    if distance_m <= 0.0:
        raise RuntimeError("backoff distance must be positive.")
    pose = copy.deepcopy(reference_pose)
    pose.header.frame_id = base_frame
    pose.header.stamp = rospy.Time.now() if "rospy" in globals() else None
    pose.pose.position.x += axis[0] * distance_m
    pose.pose.position.y += axis[1] * distance_m
    pose.pose.position.z += axis[2] * distance_m
    return pose


def build_contact_probe_end_pose(probe_start_pose, pickup_model,
                                 travel_mm, base_frame):
    axis = normalize_vector(
        pickup_model.get("approach_axis_xyz_base"),
        "approach_axis_xyz_base")
    travel_m = finite_scalar(travel_mm, "contact_probe.max_travel_mm") * 0.001
    if travel_m <= 0.0:
        raise RuntimeError("contact_probe.max_travel_mm must be positive.")
    pose = copy.deepcopy(probe_start_pose)
    pose.header.frame_id = base_frame
    pose.header.stamp = rospy.Time.now() if "rospy" in globals() else None
    pose.pose.position.x -= axis[0] * travel_m
    pose.pose.position.y -= axis[1] * travel_m
    pose.pose.position.z -= axis[2] * travel_m
    return pose


def require_contact_probe_config(config):
    settings = config.get("contact_probe")
    if not isinstance(settings, dict):
        raise RuntimeError("Missing contact_probe settings in block config.")
    normalized = dict(settings)
    for field in ("max_travel_mm", "staging_step_mm", "step_mm",
                  "point_interval_seconds", "retreat_extra_mm"):
        value = finite_scalar(settings.get(field), "contact_probe.%s" % field)
        if value <= 0.0:
            raise RuntimeError("contact_probe.%s must be positive." % field)
        normalized[field] = value
    poll_seconds = finite_scalar(
        settings.get("poll_seconds", 0.02), "contact_probe.poll_seconds")
    if poll_seconds < 0.0:
        raise RuntimeError("contact_probe.poll_seconds must be non-negative.")
    normalized["poll_seconds"] = poll_seconds
    if normalized["step_mm"] > normalized["max_travel_mm"]:
        raise RuntimeError(
            "contact_probe.step_mm cannot exceed max_travel_mm.")
    return normalized


def build_pregrasp_from_grasp(grasp_pose, camera_forward_base, gap_mm, base_frame):
    forward = normalize_vector(camera_forward_base, "camera_forward_base")
    gap_m = finite_scalar(gap_mm, "pregrasp gap mm") * 0.001
    if gap_m <= 0.0:
        raise RuntimeError("pregrasp gap must be positive.")
    pregrasp = copy.deepcopy(grasp_pose)
    pregrasp.header.frame_id = base_frame
    pregrasp.header.stamp = rospy.Time.now() if "rospy" in globals() else None
    pregrasp.pose.position.x -= forward[0] * gap_m
    pregrasp.pose.position.y -= forward[1] * gap_m
    pregrasp.pose.position.z -= forward[2] * gap_m
    return pregrasp


def load_block_preset(path):
    if not os.path.isfile(path):
        raise RuntimeError("Preset file does not exist: %s" % path)
    try:
        with open(path, "r") as handle:
            preset = json.load(handle)
    except ValueError as exc:
        raise RuntimeError("Could not parse preset JSON: %s" % exc)
    except IOError as exc:
        raise RuntimeError("Could not read preset file: %s" % exc)
    if preset.get("version") != BLOCK_PRESET_VERSION:
        raise RuntimeError(
            "Unsupported block preset version; version %d is required. "
            "The old full-transform grasp cannot be used safely and must be re-taught."
            % BLOCK_PRESET_VERSION)
    if not isinstance(preset.get("targets"), dict):
        raise RuntimeError("Preset file must contain a targets object.")
    return preset


def load_block_place_pose(entry, target, base_frame):
    transform = entry.get("place_ee_in_base")
    if not isinstance(transform, dict):
        raise RuntimeError(
            "Target %s has no independent no-Tag place pose." % target)
    return transform_to_pose(base_frame, transform)


def save_block_preset(path, preset, overwrite=False):
    if os.path.exists(path) and not overwrite:
        raise RuntimeError(
            "Preset file already exists: %s. Use --overwrite to replace it." % path)
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    try:
        with open(path, "w") as handle:
            json.dump(preset, handle, indent=2, sort_keys=True)
    except IOError as exc:
        raise RuntimeError("Could not write preset file: %s" % exc)


def load_or_create_block_preset(path, config):
    if os.path.isfile(path):
        return load_block_preset(path)
    return {
        "version": BLOCK_PRESET_VERSION,
        "base_frame": config["base_frame"],
        "targets": {},
    }


def require_taught_target(args, action):
    if args.block_target is None:
        raise RuntimeError("--target/--block-target is required for %s." % action)
    return args.block_target


def pose_to_text(name, pose_stamped):
    pose = pose_stamped.pose
    return (
        "%s position=(%.4f,%.4f,%.4f) orientation=(%.4f,%.4f,%.4f,%.4f)"
        % (
            name,
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
    )


def prompt_enter(message):
    print_utf8(message)
    while True:
        try:
            text = raw_input("> ")
        except NameError:
            text = input("> ")
        normalized = text.strip().lower()
        if normalized in ("q", "quit", "exit"):
            raise RuntimeError("User aborted taught block workflow.")
        if not normalized:
            return
        print_utf8(
            u"这里只接受空 Enter 确认，输入 q 可退出；粘贴的命令不会触发机械臂。")


def build_teach_assist_pose(localization, orientation, config):
    distance_mm = finite_scalar(
        config["teach_assist_distance_mm"],
        "teach_assist_distance_mm")
    if distance_mm <= 0.0:
        raise RuntimeError("teach_assist_distance_mm must be positive.")
    surface_pose = pose_from_base_mm(
        localization["base_frame"], localization["base_xyz_mm"], orientation)
    return build_pregrasp_from_grasp(
        surface_pose,
        localization["camera_forward_base"],
        distance_mm,
        localization["base_frame"],
    )


def build_pre_place_pose(place_pose, place_gap_m, base_frame):
    place_gap_m = finite_scalar(place_gap_m, "place_approach_gap")
    if place_gap_m <= 0.0:
        raise RuntimeError("place_approach_gap must be positive.")
    pre_place = copy.deepcopy(place_pose)
    pre_place.header.frame_id = base_frame
    pre_place.header.stamp = rospy.Time.now() if "rospy" in globals() else None
    pre_place.pose.position.z += place_gap_m
    return pre_place


class DetectorClient(object):
    def __init__(self, request_stream, response_stream):
        self._request_stream = request_stream
        self._response_stream = response_stream
        self._next_request_id = 1

    def detect(self, image_path, target):
        request_id = self._next_request_id
        self._next_request_id += 1
        payload = {"id": request_id, "image_path": image_path}
        if target is not None:
            payload["target"] = target
        self._write(payload)
        response = self._read()
        if response.get("id") != request_id:
            raise RuntimeError("detector response id does not match request")
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "detector failed"))
        return response

    def _write(self, payload):
        message = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        self._request_stream.write(message + "\n")
        self._request_stream.flush()

    def _read(self):
        line = self._response_stream.readline()
        if line == "":
            raise RuntimeError("detector response stream reached EOF")
        try:
            payload = json.loads(line)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("bad detector JSON response: %s" % exc)
        if not isinstance(payload, dict):
            raise RuntimeError("detector response must be a JSON object")
        return payload


def open_detector_streams(args):
    if args.detector_request_fd is None or args.detector_response_fd is None:
        raise RuntimeError("detector file descriptors are required for block_mono.")
    request_stream = os.fdopen(args.detector_request_fd, "w", 1)
    response_stream = os.fdopen(args.detector_response_fd, "r", 1)
    return request_stream, response_stream


def get_action(args):
    actions = [
        ("dry_run", args.dry_run),
        ("live_preview", args.live_preview),
        ("calib_record", args.calib_record),
        ("teach_block_pick_place", args.teach_block_pick_place),
        ("teach_block_pregrasp", args.teach_block_pregrasp),
        ("teach_block_place", args.teach_block_place),
        ("teach_block_idle", args.teach_block_idle),
        ("teach_block_carry", args.teach_block_carry),
        ("teach_building_contact_release",
         args.teach_building_contact_release),
        ("preview_taught_block", args.preview_taught_block),
        ("stop_at_taught_pre_grasp", args.stop_at_taught_pre_grasp),
        ("run_taught_block", args.run_taught_block),
        ("run_chassis_sequence", args.run_chassis_sequence),
    ]
    enabled = [name for name, selected in actions if selected]
    if len(enabled) != 1:
        raise RuntimeError("exactly one block action is required.")
    return enabled[0]


def capture_rgb_once(config):
    try:
        from cv_bridge import CvBridge
        from sensor_msgs.msg import CameraInfo, Image
    except ImportError as exc:
        raise RuntimeError("RGB ROS dependencies are unavailable: %s" % exc)
    bridge = CvBridge()
    timeout = finite_scalar(config["rgb_timeout"], "rgb_timeout")
    camera_info = rospy.wait_for_message(
        config["camera_info_topic"], CameraInfo, timeout=timeout)
    image_msg = rospy.wait_for_message(config["rgb_topic"], Image, timeout=timeout)
    try:
        rgb = bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
    except Exception as exc:
        raise RuntimeError("cv_bridge RGB conversion failed: %s" % exc)
    if int(camera_info.width) != int(image_msg.width) or int(camera_info.height) != int(image_msg.height):
        raise RuntimeError(
            "CameraInfo size %dx%d does not match RGB image %dx%d."
            % (camera_info.width, camera_info.height, image_msg.width, image_msg.height))
    stamp = image_msg.header.stamp
    stamp_ns = int(stamp.to_nsec())
    if stamp_ns <= 0:
        raise RuntimeError("RGB image has an invalid zero timestamp.")
    age = max(0.0, (rospy.Time.now() - stamp).to_sec())
    max_age = finite_scalar(
        config["image_max_age_seconds"], "image_max_age_seconds")
    if age > max_age:
        raise RuntimeError(
            "RGB image is stale: age %.3fs exceeds %.3fs." % (age, max_age))
    return {
        "rgb": rgb,
        "camera_info": camera_info,
        "header": copy.deepcopy(image_msg.header),
        "stamp_ns": stamp_ns,
    }


def camera_info_intrinsics(camera_info):
    projection = list(getattr(camera_info, "P", []))
    if len(projection) == 12 and projection[0] > 0.0 and projection[5] > 0.0:
        source = "CameraInfo.P"
        values = (projection[0], projection[5], projection[2], projection[6])
    else:
        matrix = list(camera_info.K)
        if len(matrix) != 9:
            raise RuntimeError("CameraInfo.K must contain 9 values.")
        source = "CameraInfo.K"
        values = (matrix[0], matrix[4], matrix[2], matrix[5])
    fx, fy, cx, cy = [finite_scalar(value, source) for value in values]
    if fx <= 0.0 or fy <= 0.0:
        raise RuntimeError("CameraInfo focal lengths must be positive.")
    return fx, fy, cx, cy


def request_detection(detector, target, rgb_image):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to write detector image: %s" % exc)
    image_fd, image_path = tempfile.mkstemp(prefix="block_mono_rgb_", suffix=".png")
    os.close(image_fd)
    try:
        if not cv2.imwrite(image_path, rgb_image):
            raise RuntimeError("could not write temporary RGB detector image")
        return detector.detect(image_path, target)
    finally:
        try:
            os.unlink(image_path)
        except OSError:
            pass


def response_detections(response):
    if isinstance(response, dict) and isinstance(response.get("detections"), list):
        return response["detections"]
    return [response]


def new_live_preview_labels(detections, reported_targets):
    abbreviations = {
        "power": "POW",
        "fire": "FIR",
        "gas": "GAS",
        "support": "SUP",
    }
    labels = []
    for detection in detections:
        if not isinstance(detection, dict):
            continue
        target = detection.get("target", "")
        if not isinstance(target, STRING_TYPES):
            continue
        target = target.strip().lower()
        if not target or target in reported_targets:
            continue
        confidence = finite_scalar(
            detection.get("confidence", 0.0), "preview confidence")
        short_name = abbreviations.get(target, target[:3].upper())
        labels.append(u"%s %s%d" % (
            target, short_name, int(round(confidence * 100.0))))
        reported_targets.add(target)
    return labels


def show_rgb_debug(
        image, detection, observation, milliseconds=1, roi_ratio=None):
    try:
        import cv2
        if isinstance(detection, (list, tuple)):
            debug = draw_debug_detections(
                image, detection, observation, roi_ratio=roi_ratio)
        else:
            debug = draw_debug_image(
                image, detection, observation, roi_ratio=roi_ratio)
        cv2.imshow("Block mono RGB - q/Esc to close", debug)
        if int(milliseconds) == 0:
            rospy.loginfo("RGB debug window is open; press q or Esc to close it.")
            while True:
                key = cv2.waitKey(100) & 0xFF
                if key in (27, ord("q")):
                    return False
        else:
            key = cv2.waitKey(int(milliseconds)) & 0xFF
            if key in (27, ord("q")):
                return False
    except Exception as exc:
        rospy.logwarn("Could not show RGB debug image: %s", ascii_log_text(exc))
    return True


def run_live_preview(args, config, detector):
    preview_hz = finite_scalar(args.preview_hz, "preview_hz")
    if preview_hz <= 0.0:
        raise RuntimeError("preview_hz must be positive.")
    period = 1.0 / preview_hz
    rospy.loginfo("Live YOLO preview started at up to %.2f Hz.", preview_hz)
    reported_targets = set()
    while not rospy.is_shutdown():
        started = time.time()
        capture = None
        detections = []
        observations = []
        try:
            capture = capture_rgb_once(config)
            try:
                response = request_detection(
                    detector, args.block_target, capture["rgb"])
                detections = response_detections(response)
                observations = [
                    detection_to_observation(item) for item in detections]
                new_labels = new_live_preview_labels(
                    detections, reported_targets)
                if new_labels:
                    print_utf8(u"检测到：" + u"，".join(new_labels))
            except Exception as exc:
                error_text = safe_log_text(exc)
                if "No usable YOLO detections" not in error_text:
                    rospy.logwarn(
                        "Live preview frame failed: %s",
                        ascii_log_text(error_text))
            if not show_rgb_debug(
                    capture["rgb"], detections, observations, milliseconds=1,
                    roi_ratio=config.get(
                        "grasp_roi_ratio", DEFAULT_CONFIG["grasp_roi_ratio"])):
                return
        except Exception as exc:
            error_text = safe_log_text(exc)
            rospy.logwarn(
                "Live preview capture/display failed: %s",
                ascii_log_text(error_text))
        remaining = period - (time.time() - started)
        if remaining > 0.0:
            rospy.sleep(remaining)


def collect_all_observations(args, config, detector):
    action = get_action(args)
    frames_required = int(args.frames or config["frames_required"])
    if frames_required <= 0:
        raise RuntimeError("frames must be positive.")
    rules = {
        "confidence_min": config["confidence_min"],
        "box_width_min_px": config["box_width_min_px"],
        "box_aspect_ratio_min": config["box_aspect_ratio_min"],
        "box_aspect_ratio_max": config["box_aspect_ratio_max"],
    }
    observations_by_target = {}
    last_capture = None
    max_attempts = max(frames_required * 4, frames_required + 10)
    for attempt in range(max_attempts):
        capture = capture_rgb_once(config)
        last_capture = capture
        frame_detections = []
        frame_observations = []
        try:
            detector_response = request_detection(detector, None, capture["rgb"])
            detections = response_detections(detector_response)
        except Exception as exc:
            rospy.logwarn(
                "YOLO all-target detection failed: %s", ascii_log_text(exc))
            continue
        for detection in detections:
            try:
                target = detection.get("target")
                if not isinstance(target, STRING_TYPES) or not target:
                    rospy.logwarn("Rejected YOLO detection without target name.")
                    continue
                usable, reason = is_detection_usable(detection, rules)
                if not usable:
                    rospy.logwarn(
                        "Rejected YOLO detection %s: %s",
                        ascii_log_text(target), ascii_log_text(reason))
                    continue
                observation = detection_to_observation(detection)
                observation["target"] = target
                inside, reason = observation_in_roi(
                    observation, capture["rgb"].shape,
                    config.get(
                        "grasp_roi_ratio", DEFAULT_CONFIG["grasp_roi_ratio"]))
                if not inside:
                    rospy.logwarn(
                        "Rejected YOLO detection %s: %s",
                        ascii_log_text(target), ascii_log_text(reason))
                    continue
            except Exception as exc:
                rospy.logwarn(
                    "YOLO detection parse failed: %s", ascii_log_text(exc))
                continue
            observations_by_target.setdefault(target, []).append(observation)
            frame_detections.append(detection)
            frame_observations.append(observation)
        user_requested_stop = False
        if args.show_rgb and frame_detections:
            if not show_rgb_debug(
                    capture["rgb"], frame_detections, frame_observations, 1,
                    roi_ratio=config.get(
                        "grasp_roi_ratio", DEFAULT_CONFIG["grasp_roi_ratio"])):
                user_requested_stop = True
        if attempt + 1 >= frames_required:
            break
        if user_requested_stop:
            break
    if not observations_by_target:
        raise RuntimeError("No usable YOLO detections were collected.")
    return observations_by_target, last_capture


def collect_observations(args, config, detector):
    action = get_action(args)
    frames_required = int(args.frames or config["frames_required"])
    if frames_required <= 0:
        raise RuntimeError("frames must be positive.")
    rules = {
        "confidence_min": config["confidence_min"],
        "box_width_min_px": config["box_width_min_px"],
        "box_aspect_ratio_min": config["box_aspect_ratio_min"],
        "box_aspect_ratio_max": config["box_aspect_ratio_max"],
    }
    observations = []
    seen_stamps = set()
    last_capture = None
    timeout = finite_scalar(
        config["observation_timeout"], "observation_timeout")
    deadline = time.time() + timeout
    latest_filter_error = None
    max_attempts = max(frames_required * 8, frames_required + 20)
    if action == "calib_record":
        print("known_z_mm,target,conf,x1,y1,x2,y2,u,v,w,h")
    for _attempt in range(max_attempts):
        if time.time() >= deadline:
            break
        capture = capture_rgb_once(config)
        if capture["stamp_ns"] in seen_stamps:
            rospy.logwarn("Rejected duplicate RGB frame timestamp: %d", capture["stamp_ns"])
            continue
        seen_stamps.add(capture["stamp_ns"])
        last_capture = capture
        try:
            detection = request_detection(detector, args.block_target, capture["rgb"])
            usable, reason = is_detection_usable(detection, rules)
            if not usable:
                rospy.logwarn(
                    "Rejected YOLO detection: %s", ascii_log_text(reason))
                continue
            observation = detection_to_observation(detection)
            observation["stamp_ns"] = capture["stamp_ns"]
            margin = config.get("horizontal_box_margin_px")
            if margin is not None:
                frame_width = int(capture["rgb"].shape[1])
                left, _top, right, _bottom = observation["box"]
                margin = finite_scalar(
                    margin, "horizontal_box_margin_px")
                if left <= margin or right >= frame_width - 1 - margin:
                    rospy.logwarn(
                        "Rejected YOLO detection: horizontal box edge is cropped")
                    continue
            inside, reason = observation_in_roi(
                observation, capture["rgb"].shape,
                config.get(
                    "grasp_roi_ratio", DEFAULT_CONFIG["grasp_roi_ratio"]))
            if not inside:
                rospy.logwarn(
                    "Rejected YOLO detection: %s", ascii_log_text(reason))
                continue
        except Exception as exc:
            rospy.logwarn("YOLO detection failed: %s", ascii_log_text(exc))
            continue
        observations.append(observation)
        user_requested_stop = False
        if args.show_rgb:
            if not show_rgb_debug(
                    capture["rgb"], detection, observation, 1,
                    roi_ratio=config.get(
                        "grasp_roi_ratio", DEFAULT_CONFIG["grasp_roi_ratio"])):
                user_requested_stop = True
        if action == "calib_record":
            print("{:.2f},{},{:.6f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}".format(
                args.known_z_mm, args.block_target, observation["confidence"],
                observation["box"][0], observation["box"][1],
                observation["box"][2], observation["box"][3],
                observation["u"], observation["v"], observation["w"], observation["h"]))
        if len(observations) >= frames_required:
            try:
                stable_median_observation(
                    observations,
                    frames_required,
                    config["center_std_max_px"],
                    config["width_cv_max"],
                )
                return observations, last_capture
            except LocalizationError as exc:
                latest_filter_error = str(exc)
                observations = observations[-frames_required * 3:]
        if user_requested_stop:
            break
    raise RuntimeError(
        "Could not collect %d stable fresh YOLO observations within %.1fs; "
        "collected %d unique frames. Last filter error: %s" %
        (frames_required, timeout, len(observations), latest_filter_error or "none")
    )


def target_number(config, target):
    metadata = config["target_classes"].get(target) or {}
    try:
        return int(metadata["target_id"])
    except (KeyError, TypeError, ValueError, OverflowError):
        raise RuntimeError("Target %s has no valid target_id." % target)


def write_chassis_sequence_result(path, completed_ids):
    """原子记录无 Tag 连续抓取实际完成的物资 ID。"""
    if not path:
        return
    path = os.path.abspath(os.path.expanduser(path))
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    temporary = "%s.%d.tmp" % (path, os.getpid())
    try:
        with open(temporary, "w") as handle:
            json.dump({"completed_ids": list(completed_ids)}, handle,
                      sort_keys=True)
            handle.write("\n")
        os.rename(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except OSError:
            pass


def make_chassis_twist(linear_x):
    message = Twist()
    message.linear.x = float(linear_x)
    message.linear.y = message.linear.z = 0.0
    message.angular.x = message.angular.y = message.angular.z = 0.0
    return message


def stop_chassis(publisher):
    for _index in range(5):
        publisher.publish(make_chassis_twist(0.0))
        rospy.sleep(0.03)


def chassis_alignment_error_px(result):
    return max(
        float(result.left) - float(result.center_x),
        float(result.center_x) - float(result.right),
        0.0,
    )


class ChassisVelocityKeeper(object):
    """Refresh the latest fresh command while synchronous ONNX inference runs."""

    def __init__(self, publisher, control_hz, command_max_age_seconds):
        self._publisher = publisher
        self._max_age = float(command_max_age_seconds)
        self._speed = 0.0
        self._updated_at = 0.0
        self._closed = False
        self._timer = rospy.Timer(
            rospy.Duration(1.0 / float(control_hz)), self._refresh)

    def publish(self, message):
        self._speed = float(message.linear.x)
        self._updated_at = time.time()
        self._publisher.publish(message)

    def _refresh(self, _event):
        if self._closed:
            return
        speed = self._speed
        if speed and time.time() - self._updated_at > self._max_age:
            speed = 0.0
            self._speed = 0.0
        self._publisher.publish(make_chassis_twist(speed))

    def shutdown(self):
        self._closed = True
        self._timer.shutdown()
        stop_chassis(self._publisher)


def require_chassis_sequence_config(config):
    settings = config.get("chassis_sequence")
    if not isinstance(settings, dict):
        raise RuntimeError("Config must contain chassis_sequence settings.")
    order = str(settings.get("order", "")).strip().lower()
    if order not in ("left_to_right", "sequence"):
        raise RuntimeError(
            "chassis_sequence.order must be left_to_right or sequence.")
    positive_fields = (
        "drive_speed",
        "stable_frames",
        "max_align_seconds",
        "progress_reset_px",
        "control_hz",
        "command_max_age_seconds",
        "startup_home_wait_seconds",
    )
    for field in positive_fields:
        if finite_scalar(settings.get(field), "chassis_sequence.%s" % field) <= 0.0:
            raise RuntimeError("chassis_sequence.%s must be positive." % field)
    stable_frames = finite_scalar(
        settings.get("stable_frames"), "chassis_sequence.stable_frames")
    if int(stable_frames) != stable_frames:
        raise RuntimeError("chassis_sequence.stable_frames must be an integer.")
    nonnegative_fields = (
        "align_tolerance_px",
        "chassis_settle_seconds",
        "startup_home_settle_seconds",
    )
    for field in nonnegative_fields:
        if finite_scalar(settings.get(field), "chassis_sequence.%s" % field) < 0.0:
            raise RuntimeError(
                "chassis_sequence.%s must be non-negative." % field)
    if str(settings.get("target_right_motion")) not in ("forward", "backward"):
        raise RuntimeError(
            "chassis_sequence.target_right_motion must be forward or backward.")
    for field in ("cmd_vel_topic", "startup_home_service"):
        if not isinstance(settings.get(field), STRING_TYPES) or not settings[field].strip():
            raise RuntimeError("chassis_sequence.%s must not be empty." % field)
    return settings


def validate_chassis_sequence_preset(path, targets, config):
    preset = load_block_preset(path)
    require_joint_values(preset, "carry_joint_values")
    require_joint_values(preset, "idle_joint_values")
    for target in targets:
        entry, _, _ = require_block_target_entry(preset, target)
        load_block_place_pose(entry, target, config["base_frame"])
    return preset


def request_sequence_detections(args, config, detector, remaining_targets,
                                display_roi_ratio=None):
    capture = capture_rgb_once(config)
    response = request_detection(detector, None, capture["rgb"])
    rules = {
        "confidence_min": config["confidence_min"],
        "box_width_min_px": config["box_width_min_px"],
        "box_aspect_ratio_min": config["box_aspect_ratio_min"],
        "box_aspect_ratio_max": config["box_aspect_ratio_max"],
    }
    detections = []
    for detection in response_detections(response):
        target = detection.get("target") if isinstance(detection, dict) else None
        if target not in remaining_targets:
            continue
        usable, _reason = is_detection_usable(detection, rules)
        if usable:
            detections.append(detection)
    if args.show_rgb:
        observations = [detection_to_observation(item) for item in detections]
        show_rgb_debug(
            capture["rgb"], detections, observations, 1,
            roi_ratio=(display_roi_ratio or config["grasp_roi_ratio"]))
    return capture, detections


def parse_search_roi_ratio(value):
    try:
        values = [float(item.strip()) for item in str(value).split(",")]
    except (TypeError, ValueError):
        raise RuntimeError("--search-roi-ratio must contain four numbers")
    if (len(values) != 4 or not
            (0.0 <= values[0] < values[2] <= 1.0 and
             0.0 <= values[1] < values[3] <= 1.0)):
        raise RuntimeError(
            "--search-roi-ratio must satisfy 0<=x1<x2<=1 and "
            "0<=y1<y2<=1")
    return values


def write_search_signal(path, text):
    if not path:
        raise RuntimeError("search signal file path is required")
    temporary = path + ".tmp.%d" % os.getpid()
    with open(temporary, "w") as handle:
        handle.write(str(text) + "\n")
    os.rename(temporary, path)


def wait_for_search_trigger(args, config, detector, remaining_targets):
    search_roi = parse_search_roi_ratio(args.search_roi_ratio)
    required = int(args.search_stable_frames)
    poll_hz = float(args.search_poll_hz)
    if required <= 0 or poll_hz <= 0.0:
        raise RuntimeError("search stable frames and poll rate must be positive")
    stable = 0
    rate = rospy.Rate(poll_hz)
    write_search_signal(args.search_ready_file, "ready")
    rospy.loginfo(
        "A-point right-side search started: roi=%s stable=%d.",
        search_roi, required)
    while not rospy.is_shutdown():
        try:
            capture, detections = request_sequence_detections(
                args, config, detector, remaining_targets,
                display_roi_ratio=search_roi)
            height, width = capture["rgb"].shape[:2]
            x1, y1, x2, y2 = roi_ratio_to_pixels(
                search_roi, width, height)
            inside = []
            for detection in detections:
                observation = detection_to_observation(detection)
                if (x1 <= observation["u"] <= x2 and
                        y1 <= observation["v"] <= y2):
                    inside.append(detection)
            stable = stable + 1 if inside else 0
            if inside:
                rospy.loginfo(
                    "A-point right-side target confirmation: %d/%d.",
                    stable, required)
            if stable >= required:
                write_search_signal(args.search_trigger_file, "triggered")
                rospy.loginfo(
                    "A-point target confirmed; waiting for line controller "
                    "to stop and release chassis ownership.")
                break
        except Exception as exc:
            stable = 0
            if "No usable YOLO detections" not in safe_log_text(exc):
                rospy.logwarn_throttle(
                    2.0, "A-point search failed: %s",
                    ascii_log_text(exc))
        rate.sleep()
    while not rospy.is_shutdown():
        if os.path.isfile(args.search_release_file):
            rospy.loginfo(
                "A-point chassis ownership released; starting slow alignment.")
            return
        rospy.sleep(0.05)
    raise TerminationRequested(
        "ROS shut down while waiting for A-point chassis release.")


def select_next_sequence_target(args, config, detector, remaining_targets,
                                settings):
    if settings["order"] == "sequence":
        return remaining_targets[0]
    strict_deadline = None
    if getattr(args, "fail_on_skip", False):
        strict_deadline = time.time() + float(settings["max_align_seconds"])
    rate = rospy.Rate(float(settings["control_hz"]))
    while not rospy.is_shutdown():
        if strict_deadline is not None and time.time() >= strict_deadline:
            raise RuntimeError(
                "No remaining block target became visible within %.1fs: %s"
                % (float(settings["max_align_seconds"]), remaining_targets))
        try:
            _capture, detections = request_sequence_detections(
                args, config, detector, remaining_targets)
        except Exception as exc:
            if "No usable YOLO detections" not in safe_log_text(exc):
                rospy.logwarn_throttle(
                    2.0, "YOLO sequence selection failed: %s",
                    ascii_log_text(exc))
            rate.sleep()
            continue
        if detections:
            detections.sort(
                key=lambda item: detection_to_observation(item)["u"])
            target = detections[0]["target"]
            rospy.loginfo(
                "Visible remaining targets left-to-right: %s; selecting %d=%s.",
                [item["target"] for item in detections],
                target_number(config, target), ascii_log_text(target))
            return target
        rospy.logwarn_throttle(
            5.0, "Waiting for a remaining block target: %s",
            [ascii_log_text(item) for item in remaining_targets])
        rate.sleep()
    raise TerminationRequested(
        "ROS shut down while waiting for a remaining block target.")


def align_sequence_target(args, config, detector, target, publisher, settings):
    stable = 0
    last_stamp_ns = None
    last_result = None
    best_error_px = None
    confirmation_required = False
    deadline = time.time() + float(settings["max_align_seconds"])
    progress_reset_px = float(settings["progress_reset_px"])
    rate = rospy.Rate(float(settings["control_hz"]))
    target_right_forward = settings["target_right_motion"] == "forward"
    while not rospy.is_shutdown() and time.time() < deadline:
        try:
            capture, detections = request_sequence_detections(
                args, config, detector, [target])
        except Exception as exc:
            stable = 0
            publisher.publish(make_chassis_twist(0.0))
            if "No usable YOLO detections" not in safe_log_text(exc):
                rospy.logwarn_throttle(
                    2.0, "YOLO chassis alignment failed: %s",
                    ascii_log_text(exc))
            rate.sleep()
            continue
        if capture["stamp_ns"] == last_stamp_ns:
            publisher.publish(make_chassis_twist(0.0))
            rate.sleep()
            continue
        last_stamp_ns = capture["stamp_ns"]
        if not detections:
            stable = 0
            publisher.publish(make_chassis_twist(0.0))
            rospy.logwarn_throttle(
                2.0, "Waiting for target %d=%s before chassis alignment.",
                target_number(config, target), ascii_log_text(target))
            rate.sleep()
            continue
        detection = max(
            detections, key=lambda item: float(item.get("confidence", 0.0)))
        height, width = capture["rgb"].shape[:2]
        roi_pixels = roi_ratio_to_pixels(
            config["grasp_roi_ratio"], width, height)
        result = compute_drive_command(
            detection,
            roi_pixels,
            float(settings["drive_speed"]),
            float(settings["align_tolerance_px"]),
            target_right_forward,
        )
        last_result = result
        alignment_error_px = chassis_alignment_error_px(result)
        if (best_error_px is None or
                alignment_error_px <= best_error_px - progress_reset_px):
            best_error_px = alignment_error_px
            deadline = time.time() + float(settings["max_align_seconds"])
        publisher.publish(make_chassis_twist(result.linear_x))
        rospy.loginfo_throttle(
            1.0,
            "Chassis align %d=%s center=%.1f window=[%.1f, %.1f] "
            "cmd_vel=%.3f.",
            target_number(config, target), ascii_log_text(target),
            result.center_x, result.left, result.right, result.linear_x)
        if result.aligned:
            if not confirmation_required:
                stop_chassis(publisher)
                rospy.loginfo(
                    "Target %d=%s entered ROI; settling chassis for %.2fs.",
                    target_number(config, target), ascii_log_text(target),
                    float(settings["chassis_settle_seconds"]))
                rospy.sleep(float(settings["chassis_settle_seconds"]))
                stable = 0
                confirmation_required = True
                # Unlike the Tag relay, monocular ONNX inference is synchronous
                # and may take several seconds per fresh frame. Keep the same
                # stable-frame gate, but allow the normal alignment timeout.
                deadline = time.time() + float(settings["max_align_seconds"])
            else:
                stable += 1
                stop_chassis(publisher)
                rospy.loginfo(
                    "Target %d=%s stopped confirmation: %d/%d fresh frames.",
                    target_number(config, target), ascii_log_text(target),
                    stable, int(settings["stable_frames"]))
                if stable >= int(settings["stable_frames"]):
                    rospy.loginfo(
                        "Target %d=%s alignment confirmed with %d fresh frames.",
                        target_number(config, target), ascii_log_text(target), stable)
                    return
        else:
            stable = 0
            if confirmation_required:
                rospy.logwarn(
                    "Target %d=%s left ROI after settling; resuming slow alignment.",
                    target_number(config, target), ascii_log_text(target))
                deadline = time.time() + float(settings["max_align_seconds"])
                best_error_px = None
            confirmation_required = False
        rate.sleep()
    stop_chassis(publisher)
    detail = ""
    if last_result is not None:
        detail = (
            " No progress for %.1fs. Last center=%.1f, "
            "window=[%.1f, %.1f], cmd_vel=%.3f."
            % (float(settings["max_align_seconds"]), last_result.center_x,
               last_result.left, last_result.right, last_result.linear_x))
    raise RuntimeError(
        "Target %d=%s chassis alignment timed out.%s" % (
            target_number(config, target), target, detail))


def run_sequence_startup_home(config, target, settings, skip):
    if skip:
        return
    service_name = settings["startup_home_service"]
    rospy.loginfo(
        "Preparing target %d=%s; calling startup home service %s before localization.",
        target_number(config, target), ascii_log_text(target), service_name)
    rospy.wait_for_service(
        service_name, timeout=float(settings["startup_home_wait_seconds"]))
    response = rospy.ServiceProxy(service_name, Trigger)()
    if not response.success:
        raise RuntimeError(
            "Startup home failed before target %s: %s" % (
                target, safe_log_text(response.message)))
    rospy.sleep(float(settings["startup_home_settle_seconds"]))


def run_block_chassis_sequence(args, config, detector):
    settings = require_chassis_sequence_config(config)
    targets = parse_target_sequence(args.sequence, config)
    if not args.align_only:
        validate_chassis_sequence_preset(args.preset_file, targets, config)
    if args.search_before_chassis:
        wait_for_search_trigger(args, config, detector, targets)
    raw_publisher = rospy.Publisher(
        settings["cmd_vel_topic"], Twist, queue_size=1)
    publisher = ChassisVelocityKeeper(
        raw_publisher, settings["control_hz"],
        settings["command_max_age_seconds"])
    remaining_targets = list(targets)
    requested = getattr(args, "max_targets", None)
    total = len(remaining_targets) if requested is None else int(requested)
    if not 1 <= total <= len(remaining_targets):
        raise RuntimeError(
            "--max-targets must be between 1 and the sequence length.")
    completed = 0
    completed_ids = []
    result_file = getattr(args, "result_file", None)
    write_chassis_sequence_result(result_file, completed_ids)
    try:
        while (remaining_targets and completed < total
               and not rospy.is_shutdown()):
            target = select_next_sequence_target(
                args, config, detector, remaining_targets, settings)
            rospy.loginfo(
                "Starting slow chassis alignment for %d=%s.",
                target_number(config, target), ascii_log_text(target))
            align_sequence_target(
                args, config, detector, target, publisher, settings)
            stop_chassis(publisher)
            if not args.align_only:
                run_sequence_startup_home(
                    config, target, settings, args.skip_startup_home)
                args.block_target = target
                localization = compute_block_localization(
                    args, config, detector)
                try:
                    do_run_taught_block_mono(
                        args, config, localization, "run_taught_block")
                except ContactProbeMiss:
                    rospy.logwarn(
                        "Target %d=%s did not trigger the contact switch; "
                        "skipping it and continuing with the remaining targets.",
                        target_number(config, target), ascii_log_text(target))
                    remaining_targets.remove(target)
                    continue
            remaining_targets.remove(target)
            completed += 1
            completed_ids.append(target_number(config, target))
            write_chassis_sequence_result(result_file, completed_ids)
        if completed < total:
            raise RuntimeError(
                "Only %d/%d tagless targets completed." % (completed, total))
    finally:
        publisher.shutdown()


def pose_from_camera_xyz_mm(frame_id, camera_xyz_mm):
    point = PoseStamped()
    point.header.frame_id = frame_id
    point.header.stamp = rospy.Time(0)
    point.pose.position.x = camera_xyz_mm[0] * 0.001
    point.pose.position.y = camera_xyz_mm[1] * 0.001
    point.pose.position.z = camera_xyz_mm[2] * 0.001
    point.pose.orientation.w = 1.0
    return point


def transform_camera_point(listener, base_frame, camera_frame, camera_xyz_mm, tf_timeout):
    point = pose_from_camera_xyz_mm(camera_frame, camera_xyz_mm)
    listener.waitForTransform(base_frame, camera_frame, rospy.Time(0),
                              rospy.Duration(tf_timeout))
    transformed = listener.transformPose(base_frame, point)
    return (
        transformed.pose.position.x * 1000.0,
        transformed.pose.position.y * 1000.0,
        transformed.pose.position.z * 1000.0,
    )


def compute_block_localization(args, config, detector):
    if args.block_target is None:
        return compute_all_block_localizations(args, config, detector)

    observations, capture = collect_observations(args, config, detector)
    action = get_action(args)
    if action == "calib_record":
        return None

    listener = tf.TransformListener()
    rospy.sleep(0.2)
    localization = build_localization_from_observations(
        args.block_target, observations, capture, args, config, listener)
    summary = format_localization_summary(localization)
    rospy.loginfo("Localized target %s.", ascii_log_text(args.block_target))
    print_utf8(summary)
    if args.show_rgb and action == "dry_run":
        show_rgb_debug(
            capture["rgb"],
            localization_debug_detection(localization),
            localization_debug_observation(localization),
            0,
            roi_ratio=config.get(
                "grasp_roi_ratio", DEFAULT_CONFIG["grasp_roi_ratio"]),
        )
    return localization


def build_localization_from_observations(target, observations, capture, args, config, listener):
    stable = stable_median_observation(
        observations,
        int(args.frames or config["frames_required"]),
        config["center_std_max_px"],
        config["width_cv_max"],
    )
    fx, fy, cx, cy = camera_info_intrinsics(capture["camera_info"])
    distance_width_px = scale_box_width_for_distance(
        stable["w"], int(capture["rgb"].shape[1]),
        config.get("distance_model_frame_width"))
    z_mm = estimate_distance_from_box_mm(
        config["distance_method"],
        distance_width_px,
        stable["h"],
        fx,
        fy,
        config["target_size_mm"],
        config["target_height_mm"],
        target,
        config["distance_models"],
        config.get("fixed_z_mm"),
        config["max_axis_distance_disagreement_mm"],
    )
    distance_range = (config.get("distance_ranges_mm") or {}).get(target)
    if distance_range is not None:
        minimum, maximum = [
            finite_scalar(value, "%s distance range" % target)
            for value in distance_range]
        if not minimum <= z_mm <= maximum:
            raise RuntimeError(
                "%s estimated distance %.1fmm is outside %.1f~%.1fmm" %
                (target, z_mm, minimum, maximum))
    camera_xyz = deproject_pixel_to_camera_mm(
        stable["u"], stable["v"], z_mm, fx, fy, cx, cy)
    camera_frame = getattr(capture["header"], "frame_id", "") or config["camera_frame"]
    base_frame = config["base_frame"]
    tf_timeout = finite_scalar(config["tf_timeout"], "tf_timeout")
    surface_base = transform_camera_point(
        listener, base_frame, camera_frame, camera_xyz, tf_timeout)
    reference_camera = (camera_xyz[0], camera_xyz[1], camera_xyz[2] + 100.0)
    reference_base = transform_camera_point(
        listener, base_frame, camera_frame, reference_camera, tf_timeout)
    camera_forward_base = tuple(reference_base[i] - surface_base[i] for i in range(3))
    localization = {
        "target": target,
        "confidence": stable["confidence"],
        "box": [
            stable["u"] - stable["w"] * 0.5,
            stable["v"] - stable["h"] * 0.5,
            stable["u"] + stable["w"] * 0.5,
            stable["v"] + stable["h"] * 0.5,
        ],
        "u": stable["u"],
        "v": stable["v"],
        "w": stable["w"],
        "h": stable["h"],
        "center_std_px": stable["center_std_px"],
        "width_cv": stable["width_cv"],
        "distance_method": config["distance_method"],
        "z_mm": z_mm,
        "camera_xyz_mm": camera_xyz,
        "base_xyz_mm": surface_base,
        "camera_forward_base": camera_forward_base,
        "base_frame": base_frame,
    }
    return localization


def localization_debug_detection(localization):
    return {
        "target": localization["target"],
        "class_name": localization["target"],
        "confidence": localization["confidence"],
        "box": localization["box"],
    }


def localization_debug_observation(localization):
    return {
        "target": localization["target"],
        "confidence": localization["confidence"],
        "box": localization["box"],
        "u": localization["u"],
        "v": localization["v"],
        "w": localization["w"],
        "h": localization["h"],
    }


def compute_all_block_localizations(args, config, detector):
    action = get_action(args)
    if action != "dry_run":
        raise RuntimeError("All-target localization is only allowed in --dry-run.")
    observations_by_target, capture = collect_all_observations(args, config, detector)
    listener = tf.TransformListener()
    rospy.sleep(0.2)
    localizations = []
    frames_required = int(args.frames or config["frames_required"])
    for target in sorted(observations_by_target):
        observations = observations_by_target[target]
        if len(observations) < frames_required:
            rospy.logwarn(
                "Target %s has insufficient valid frames: %d/%d; skipping.",
                ascii_log_text(target), len(observations), frames_required)
            continue
        try:
            localization = build_localization_from_observations(
                target, observations, capture, args, config, listener)
        except Exception as exc:
            rospy.logwarn(
                "Target %s localization failed: %s",
                ascii_log_text(target), ascii_log_text(exc))
            continue
        localizations.append(localization)
        summary = format_localization_summary(localization)
        rospy.loginfo("Localized target %s.", ascii_log_text(target))
        print_utf8(summary)
    if not localizations:
        raise RuntimeError("No target had enough stable observations for localization.")
    count_text = u"YOLO稳定识别到{}个目标。".format(len(localizations))
    rospy.loginfo("Localized %d stable YOLO targets.", len(localizations))
    print_utf8(count_text)
    if args.show_rgb:
        show_rgb_debug(
            capture["rgb"],
            [localization_debug_detection(item) for item in localizations],
            [localization_debug_observation(item) for item in localizations],
            0,
            roi_ratio=config.get(
                "grasp_roi_ratio", DEFAULT_CONFIG["grasp_roi_ratio"]),
        )
    return localizations


def quaternion_msg_to_tuple(quaternion):
    return (quaternion.x, quaternion.y, quaternion.z, quaternion.w)


def pose_from_base_mm(base_frame, xyz_mm, orientation):
    pose = PoseStamped()
    pose.header.frame_id = base_frame
    pose.header.stamp = rospy.Time.now()
    pose.pose.position.x = xyz_mm[0] * 0.001
    pose.pose.position.y = xyz_mm[1] * 0.001
    pose.pose.position.z = xyz_mm[2] * 0.001
    pose.pose.orientation = copy.deepcopy(orientation)
    return pose


def validate_workspace(point_mm, config, label):
    z_min = finite_scalar(config["base_min_z_mm"], "base_min_z_mm")
    radius_max = finite_scalar(config["base_max_radius_mm"], "base_max_radius_mm")
    x, y, z = finite_vector3(point_mm, label)
    if z < z_min:
        raise RuntimeError("%s z %.2f below %.2f mm." % (label, z, z_min))
    if math.sqrt(x * x + y * y) > radius_max:
        raise RuntimeError("%s radius exceeds %.2f mm." % (label, radius_max))


def validate_pose_workspace(pose_stamped, config, label):
    position = pose_stamped.pose.position
    validate_workspace(
        (position.x * 1000.0, position.y * 1000.0, position.z * 1000.0),
        config,
        label,
    )


def build_move_group(config, group_name):
    arm = moveit_commander.MoveGroupCommander(group_name)
    arm.set_pose_reference_frame(config["base_frame"])
    arm.allow_replanning(True)
    arm.set_max_velocity_scaling_factor(float(config["velocity_scale"]))
    arm.set_max_acceleration_scaling_factor(float(config["acceleration_scale"]))
    arm.set_planning_time(float(config["planning_time"]))
    return arm


def execute_pose(arm, target_pose, label, position_only=False):
    rospy.loginfo("Executing %s", label)
    for attempt in range(2):
        arm.set_start_state_to_current_state()
        if position_only:
            position = target_pose.pose.position
            arm.set_position_target([
                position.x, position.y, position.z])
        else:
            arm.set_pose_target(target_pose)
        success = arm.go(wait=True)
        arm.stop()
        arm.clear_pose_targets()
        if success:
            if MOTION_SETTLE_SECONDS > 0.0:
                rospy.sleep(MOTION_SETTLE_SECONDS)
            return
        if attempt == 0:
            rospy.logwarn("MoveIt failed during %s; retrying once from current state.", label)
            rospy.sleep(0.4)
    raise RuntimeError("MoveIt failed during %s." % label)


def execute_joint_values(arm, joint_values, label):
    rospy.loginfo("Executing %s joint_values=%s", label, joint_values)
    for attempt in range(2):
        arm.set_start_state_to_current_state()
        arm.set_joint_value_target([float(value) for value in joint_values])
        success = arm.go(wait=True)
        arm.stop()
        arm.clear_pose_targets()
        if success:
            if MOTION_SETTLE_SECONDS > 0.0:
                rospy.sleep(MOTION_SETTLE_SECONDS)
            return
        if attempt == 0:
            rospy.logwarn("MoveIt failed during %s; retrying once from current state.", label)
            rospy.sleep(0.5)
    raise RuntimeError("MoveIt failed during %s." % label)


def execute_cartesian_pose(arm, target_pose, label,
                           retry_without_collisions=False,
                           fallback_to_pose=False, eef_step=0.005,
                           settle=True, stop_after=True,
                           min_point_interval=0.0, quiet=False):
    if not quiet:
        rospy.loginfo("Executing cartesian %s", label)
    for attempt in range(2):
        arm.set_start_state_to_current_state()
        plan, fraction = arm.compute_cartesian_path(
            [copy.deepcopy(target_pose.pose)], eef_step, 0.0, True)
        if fraction < 0.999 and retry_without_collisions:
            arm.set_start_state_to_current_state()
            plan, fraction = arm.compute_cartesian_path(
                [copy.deepcopy(target_pose.pose)], eef_step, 0.0, False)
        if fraction < 0.999:
            if fallback_to_pose:
                execute_pose(arm, target_pose, label + "_pose_fallback")
                return
            raise RuntimeError(
                "MoveIt cartesian path failed during %s (fraction=%.3f)."
                % (label, fraction))
        if not plan.joint_trajectory.points:
            raise RuntimeError("MoveIt returned an empty cartesian plan for %s." % label)
        if min_point_interval > 0.0:
            for point_index, point in enumerate(plan.joint_trajectory.points):
                minimum_seconds = point_index * min_point_interval
                if point.time_from_start.to_sec() < minimum_seconds:
                    point.time_from_start = rospy.Duration(minimum_seconds)
        success = arm.execute(plan, wait=True)
        if stop_after or not success:
            arm.stop()
        arm.clear_pose_targets()
        if success:
            if settle and MOTION_SETTLE_SECONDS > 0.0:
                rospy.sleep(MOTION_SETTLE_SECONDS)
            return
        if attempt == 0:
            rospy.logwarn(
                "MoveIt execute failed during %s; retrying once from current state.",
                label)
            rospy.sleep(0.5)
    raise RuntimeError("MoveIt execute failed during %s." % label)


def get_mirobot_pump_type():
    try:
        from mirobot_urdf_2.srv import mirobotPump
        return mirobotPump
    except ImportError:
        raise RuntimeError(
            "mirobot_urdf_2.srv is unavailable; source the robot workspace first.")


def get_pump_proxy():
    rospy.wait_for_service("switch_pump_status", timeout=5.0)
    return rospy.ServiceProxy("switch_pump_status", get_mirobot_pump_type())


def get_contact_service_types():
    try:
        from std_srvs.srv import SetBool, Trigger
        return SetBool, Trigger
    except ImportError:
        raise RuntimeError(
            "std_srvs contact service types are unavailable; source the ROS workspaces first.")


def get_contact_proxies():
    enable_type, state_type = get_contact_service_types()
    rospy.loginfo("Waiting for contact probe services: %s, %s",
                  CONTACT_PROBE_ENABLE_SERVICE, CONTACT_STATE_SERVICE)
    rospy.wait_for_service(CONTACT_PROBE_ENABLE_SERVICE, timeout=5.0)
    rospy.wait_for_service(CONTACT_STATE_SERVICE, timeout=5.0)
    return (
        rospy.ServiceProxy(CONTACT_PROBE_ENABLE_SERVICE, enable_type),
        rospy.ServiceProxy(CONTACT_STATE_SERVICE, state_type),
    )


def set_pump(pump_proxy, enabled):
    rospy.loginfo("Pump %s", "ON" if enabled else "OFF")
    response = pump_proxy(enabled)
    if not response.Sucess:
        raise RuntimeError("Pump service returned failure.")


def set_contact_probe_enabled(enable_proxy, enabled):
    response = enable_proxy(bool(enabled))
    if not response.success:
        raise RuntimeError(
            "Contact probe %s failed: %s" % (
                "enable" if enabled else "disable", response.message))


def contact_is_triggered(state_proxy):
    response = state_proxy()
    message = str(response.message or "")
    if not response.success and message.startswith("ERROR:"):
        raise RuntimeError("Contact state read failed: %s" % message)
    return bool(response.success)


def run_contact_approach(arm, taught_pre_grasp_pose, pickup_model, base_frame,
                         settings, enable_proxy, state_proxy):
    max_travel_mm = finite_scalar(
        settings["max_travel_mm"], "contact_probe.max_travel_mm")
    staging_step_mm = finite_scalar(
        settings["staging_step_mm"], "contact_probe.staging_step_mm")
    step_mm = finite_scalar(settings["step_mm"], "contact_probe.step_mm")
    interval = finite_scalar(
        settings["point_interval_seconds"],
        "contact_probe.point_interval_seconds")
    set_contact_probe_enabled(enable_proxy, True)
    try:
        if contact_is_triggered(state_proxy):
            return True
        rospy.loginfo(
            "Contact guard active before taught pre-grasp; staging uses "
            "%.0fmm steps, then probes up to %.0fmm past P in %.0fmm steps.",
            staging_step_mm, max_travel_mm, step_mm)
        execute_cartesian_pose(
            arm, taught_pre_grasp_pose, "block_guarded_to_pregrasp",
            eef_step=staging_step_mm * 0.001,
            settle=False, stop_after=False,
            min_point_interval=interval, quiet=True)
        rospy.sleep(float(settings.get("poll_seconds", 0.02)))
        if contact_is_triggered(state_proxy):
            rospy.loginfo(
                "Contact triggered before reaching the taught pre-grasp P.")
            return True
        probe_end_pose = build_contact_probe_end_pose(
            taught_pre_grasp_pose, pickup_model, max_travel_mm, base_frame)
        execute_cartesian_pose(
            arm, probe_end_pose, "block_contact_probe",
            eef_step=step_mm * 0.001, settle=False, stop_after=False,
            min_point_interval=interval, quiet=True)
        rospy.sleep(float(settings.get("poll_seconds", 0.02)))
        return contact_is_triggered(state_proxy)
    finally:
        set_contact_probe_enabled(enable_proxy, False)


def do_home(config, group_name):
    arm = build_move_group(config, group_name)
    arm.set_start_state_to_current_state()
    arm.set_named_target("home")
    success = arm.go(wait=True)
    arm.stop()
    arm.clear_pose_targets()
    if not success:
        raise RuntimeError("Failed to move home.")


def do_current_pose(config, group_name):
    arm = build_move_group(config, group_name)
    pose = arm.get_current_pose()
    position = pose.pose.position
    orientation = pose.pose.orientation
    text = (
        "current_pose position=({:.4f},{:.4f},{:.4f}) "
        "orientation=({:.4f},{:.4f},{:.4f},{:.4f})"
    ).format(position.x, position.y, position.z,
             orientation.x, orientation.y, orientation.z, orientation.w)
    rospy.loginfo(text)
    print(text)


def do_wrist_forward(config, group_name, joint5_target):
    arm = build_move_group(config, group_name)
    values = list(arm.get_current_joint_values())
    if len(values) < 5:
        raise RuntimeError("Expected at least 5 joints.")
    values[4] = joint5_target
    arm.set_start_state_to_current_state()
    arm.set_joint_value_target(values)
    success = arm.go(wait=True)
    arm.stop()
    arm.clear_pose_targets()
    if not success:
        raise RuntimeError("Failed to move wrist forward.")


def capture_block_pregrasp(config, localization, arm, target):
    current_orientation = copy.deepcopy(
        arm.get_current_pose().pose.orientation)
    assist_pose = build_teach_assist_pose(
        localization, current_orientation, config)
    assist_distance_mm = finite_scalar(
        config["teach_assist_distance_mm"],
        "teach_assist_distance_mm")
    rospy.loginfo(pose_to_text("block_teach_assist_front", assist_pose))
    prompt_enter(
        u"确认路径安全，按 Enter 自动移动到 %s 检测表面前约 %.0fmm。\n"
        u"本步保持当前末端姿态，不套用旧抓取姿态。" % (
            safe_log_text(target), assist_distance_mm))
    try:
        execute_pose(arm, assist_pose, "block_teach_assist_front")
    except RuntimeError as exc:
        rospy.logwarn(
            "Automatic %.0fmm teach-assist move failed: %s. "
            "Continue in RViz and move to the pre-grasp pose manually.",
            assist_distance_mm, safe_log_text(exc))
    prompt_enter(
        u"请在 RViz 中 Plan/Execute 微调到靠近、正对但未接触 %s 的预抓点 P，"
        u"确认到位后按 Enter 记录；不要用手掰机械臂。正式抓取会先到 P 后方安全点，"
        u"再直线伸到 P 并启动限位前探。" %
        safe_log_text(target))
    taught_pre_grasp_pose = arm.get_current_pose()
    anchor_pose = block_anchor_pose_from_localization(localization, config)
    pickup_model = create_block_pickup_model(
        taught_pre_grasp_pose, localization["camera_forward_base"])
    pregrasp_offset = compute_block_pregrasp_offset(
        anchor_pose, taught_pre_grasp_pose)
    validate_pose_workspace(
        taught_pre_grasp_pose, config, "block_taught_pre_grasp")
    return taught_pre_grasp_pose, {
        "pregrasp_offset_xyz_base": pregrasp_offset,
        "pickup_model": pickup_model,
    }


def move_to_saved_block_pregrasp(config, localization, arm, target, entry):
    pickup_model = require_block_pickup_model(entry, target)
    pregrasp_offset = require_block_pregrasp_offset(entry, target)
    anchor_pose = block_anchor_pose_from_localization(localization, config)
    taught_pre_grasp_pose = compute_taught_block_pregrasp_pose(
        anchor_pose, pickup_model, pregrasp_offset,
        localization["base_frame"])
    staging_pose = build_block_backoff_pose(
        taught_pre_grasp_pose, pickup_model, config["approach_gap_mm"],
        localization["base_frame"])
    validate_pose_workspace(staging_pose, config, "block_approach_staging")
    validate_pose_workspace(
        taught_pre_grasp_pose, config, "block_taught_pre_grasp")
    rospy.loginfo(
        "Moving to the saved %s pre-grasp before place-only teaching.",
        ascii_log_text(target))
    execute_pose(arm, staging_pose, "block_place_teach_staging")
    execute_cartesian_pose(
        arm, taught_pre_grasp_pose, "block_place_teach_pregrasp")
    return taught_pre_grasp_pose


def capture_block_place(config, arm, target):
    prompt_enter(
        u"机械臂当前停在 %s 的预抓点 P。\n"
        u"请从该姿态开始，在 RViz 中 Plan/Execute 到 %s 对应的无 Tag 载物仓"
        u"释放姿态；确认到位后按 Enter 记录完整 Link6 放置位姿。"
        u"不要用手掰机械臂。" % (
            safe_log_text(target), safe_log_text(target)))
    place_pose = arm.get_current_pose()
    validate_pose_workspace(place_pose, config, "block_taught_place")
    return place_pose


def remove_obsolete_shared_block_teaching(preset):
    preset.pop("pickup_model", None)
    preset.pop("shared_pregrasp_offset_xyz_base", None)
    preset.pop("shared_pregrasp_reference_target", None)
    preset.pop("shared_grasp_offset_xyz_base", None)
    preset.pop("shared_grasp_reference_target", None)


def record_block_teaching(args, config, localization, arm, preset, action):
    target = require_taught_target(args, action)
    targets = preset.setdefault("targets", {})
    old_entry = targets.get(target)
    entry = copy.deepcopy(old_entry) if isinstance(old_entry, dict) else {}
    if entry:
        print_utf8(
            u"该类别已有无 Tag 示教；本次要求的点全部采集成功后才替换，"
            u"中途失败会保留旧值。")

    taught_pre_grasp_pose = None
    if action in ("teach_block_pick_place", "teach_block_pregrasp"):
        taught_pre_grasp_pose, pregrasp_fields = capture_block_pregrasp(
            config, localization, arm, target)
        entry.update(pregrasp_fields)
    else:
        taught_pre_grasp_pose = move_to_saved_block_pregrasp(
            config, localization, arm, target, entry)

    place_pose = None
    if action in ("teach_block_pick_place", "teach_block_place"):
        place_pose = capture_block_place(config, arm, target)
        entry["place_ee_in_base"] = pose_to_transform(place_pose)
    targets[target] = entry
    remove_obsolete_shared_block_teaching(preset)
    rospy.loginfo(
        "Saved independent no-Tag teaching action=%s target=%s.",
        action, ascii_log_text(target))
    if taught_pre_grasp_pose is not None:
        rospy.loginfo(pose_to_text(
            "block_taught_pre_grasp", taught_pre_grasp_pose))
    if place_pose is not None:
        rospy.loginfo(pose_to_text("block_taught_place", place_pose))
    return entry


def do_teach_block_mono(args, config, localization, action):
    arm = build_move_group(config, args.group)
    preset = load_or_create_block_preset(args.preset_file, config)
    target = args.block_target
    if action in ("teach_block_pick_place", "teach_block_pregrasp",
                  "teach_block_place"):
        record_block_teaching(
            args, config, localization, arm, preset, action)
        preset["base_frame"] = config["base_frame"]
        save_block_preset(args.preset_file, preset, overwrite=True)
        rospy.loginfo(
            "Saved taught block preset for %s: %s",
            ascii_log_text(target), ascii_log_text(args.preset_file))
        return

def do_teach_block_joints(args, config, action):
    arm = build_move_group(config, args.group)
    preset = load_or_create_block_preset(args.preset_file, config)
    if action == "teach_block_idle":
        field = "idle_joint_values"
        prompt = u"请在 RViz 中移动到比赛等待/空闲姿态，Plan/Execute 后按 Enter 记录。"
    else:
        field = "carry_joint_values"
        prompt = u"请在 RViz 中移动到抓起物块后的安全搬运姿态，Plan/Execute 后按 Enter 记录。"
    if field in preset and not args.overwrite:
        raise RuntimeError("Preset already contains %s. Use --overwrite to update it." % field)
    prompt_enter(prompt)
    preset[field] = [float(value) for value in arm.get_current_joint_values()]
    save_block_preset(args.preset_file, preset, overwrite=True)
    rospy.loginfo("Saved %s=%s", field, preset[field])


def do_run_taught_block_mono(args, config, localization, action):
    target = require_taught_target(args, action)
    preset = load_block_preset(args.preset_file)
    entry, pickup_model, pregrasp_offset = require_block_target_entry(
        preset, target)
    probe_settings = require_contact_probe_config(config)
    anchor_pose = block_anchor_pose_from_localization(localization, config)
    taught_pre_grasp_pose = compute_taught_block_pregrasp_pose(
        anchor_pose, pickup_model, pregrasp_offset,
        localization["base_frame"])
    approach_staging_pose = build_block_backoff_pose(
        taught_pre_grasp_pose, pickup_model,
        config["approach_gap_mm"],
        localization["base_frame"],
    )
    place_pose = load_block_place_pose(
        entry, target, localization["base_frame"])
    pre_place_pose = build_pre_place_pose(
        place_pose, args.place_approach_gap, localization["base_frame"])
    probe_end_pose = build_contact_probe_end_pose(
        taught_pre_grasp_pose, pickup_model,
        probe_settings["max_travel_mm"], localization["base_frame"])
    retreat_pose = build_block_backoff_pose(
        taught_pre_grasp_pose, pickup_model,
        probe_settings["retreat_extra_mm"],
        localization["base_frame"])

    for label, pose in (
        ("block_approach_staging", approach_staging_pose),
        ("taught_block_pre_grasp", taught_pre_grasp_pose),
        ("taught_block_probe_end", probe_end_pose),
        ("taught_block_retreat", retreat_pose),
        ("taught_block_pre_place", pre_place_pose),
        ("taught_block_place", place_pose),
    ):
        validate_pose_workspace(pose, config, label)

    rospy.loginfo(pose_to_text(
        "block_approach_staging", approach_staging_pose))
    rospy.loginfo(pose_to_text(
        "taught_block_pre_grasp", taught_pre_grasp_pose))
    rospy.loginfo(pose_to_text("taught_block_probe_end", probe_end_pose))
    rospy.loginfo(pose_to_text("taught_block_retreat", retreat_pose))
    rospy.loginfo(pose_to_text("taught_block_pre_place", pre_place_pose))
    rospy.loginfo(pose_to_text("taught_block_place", place_pose))

    if action == "preview_taught_block":
        rospy.logwarn("Preview only: no arm motion or pump command executed.")
        return
    arm = build_move_group(config, args.group)
    if action == "stop_at_taught_pre_grasp":
        execute_pose(arm, approach_staging_pose, "block_approach_staging")
        execute_cartesian_pose(
            arm, taught_pre_grasp_pose, "taught_block_pre_grasp")
        rospy.logwarn("Stopped at taught pre-grasp; pump was not enabled.")
        return

    carry_joint_values = require_joint_values(preset, "carry_joint_values")
    pump_proxy = get_pump_proxy()
    contact_proxies = get_contact_proxies()
    holding_object = False
    try:
        set_pump(pump_proxy, False)
        execute_pose(arm, approach_staging_pose, "block_approach_staging")
        if not run_contact_approach(
                arm, taught_pre_grasp_pose, pickup_model,
                localization["base_frame"], probe_settings,
                contact_proxies[0], contact_proxies[1]):
            rospy.logwarn(
                "CONTACT_PROBE_MISS target=%s: no contact within %.0fmm; retreating.",
                ascii_log_text(target), probe_settings["max_travel_mm"])
            set_pump(pump_proxy, False)
            execute_cartesian_pose(
                arm, retreat_pose, "block_contact_probe_miss_retreat")
            if preset.get("idle_joint_values"):
                execute_joint_values(
                    arm, preset["idle_joint_values"], "block_idle")
            raise ContactProbeMiss(target)
        set_pump(pump_proxy, True)
        holding_object = True
        rospy.sleep(0.8)
        rospy.loginfo(
            "Contact secured; retreating straight %.0fmm past pre-grasp before carry planning.",
            probe_settings["retreat_extra_mm"])
        execute_cartesian_pose(arm, retreat_pose, "taught_block_retreat")
        execute_joint_values(arm, carry_joint_values, "block_carry")
        execute_pose(arm, pre_place_pose, "taught_block_pre_place")
        execute_cartesian_pose(
            arm, place_pose, "taught_block_place",
            retry_without_collisions=True, fallback_to_pose=True)
        set_pump(pump_proxy, False)
        holding_object = False
        rospy.sleep(0.5)
        execute_cartesian_pose(
            arm, pre_place_pose, "taught_block_place_retreat",
            retry_without_collisions=True, fallback_to_pose=True)
        if preset.get("idle_joint_values"):
            execute_joint_values(arm, preset["idle_joint_values"], "block_idle")
    except Exception:
        if holding_object:
            try:
                rospy.logwarn("Motion failed while pump was ON; turning pump OFF before aborting.")
                set_pump(pump_proxy, False)
            except Exception as pump_error:
                rospy.logerr(
                    "Failed to turn pump OFF after motion error: %s",
                    ascii_log_text(pump_error))
        raise


def teach_building_contact_release(args, config, detector):
    import mirobot_delivery as delivery_api

    target = args.block_target
    item_id = target_number(config, target)
    delivery_preset = delivery_api.load_delivery_preset(
        args.delivery_file, allow_missing=True)
    targets = delivery_preset["contact_delivery_targets_by_id"]
    key = str(item_id)
    if key in targets and not args.overwrite:
        raise RuntimeError(
            "ID%d already has a building contact point; use --overwrite."
            % item_id)

    block_preset = load_block_preset(args.preset_file)
    idle_joint_values = require_joint_values(
        block_preset, "idle_joint_values")
    arm = build_move_group(config, args.group)
    prompt_enter(
        u"示教 ID%d=%s：确认路径安全，按 Enter 先自动移动到"
        u"无 Tag idle，避免机械臂遮挡楼宇摄像头。" %
        (item_id, safe_log_text(target)))
    execute_joint_values(
        arm, idle_joint_values,
        "building_%d_teach_idle" % item_id)

    localization = compute_block_localization(args, config, detector)
    current_orientation = copy.deepcopy(
        arm.get_current_pose().pose.orientation)
    assist_pose = build_teach_assist_pose(
        localization, current_orientation, config)
    validate_pose_workspace(
        assist_pose, config,
        "building_%d_teach_assist_front" % item_id)
    assist_distance_mm = finite_scalar(
        config["teach_assist_distance_mm"], "teach_assist_distance_mm")
    rospy.loginfo(pose_to_text(
        "building_%d_teach_assist_front" % item_id, assist_pose))
    prompt_enter(
        u"已完成楼宇稳定检测、纯RGB估距和手眼TF定位。\n"
        u"确认路径安全，按 Enter 自动移动到检测楼面前约 %.0fmm。" %
        assist_distance_mm)
    try:
        execute_pose(
            arm, assist_pose,
            "building_%d_teach_assist_front" % item_id,
            position_only=True)
    except RuntimeError as exc:
        rospy.logwarn(
            "Building teach-assist move failed: %s. Continue with RViz.",
            ascii_log_text(exc))
    prompt_enter(
        u"请从当前较远安全点开始，在 RViz 中 Plan/Execute 微调到"
        u"靠近、正对但不接触楼面的预投递点 P；到位后按 Enter 保存。")
    rospy.sleep(MOTION_SETTLE_SECONDS)
    precontact_pose = arm.get_current_pose()
    validate_pose_workspace(
        precontact_pose, config,
        "building_%d_taught_precontact" % item_id)
    pickup_model = create_block_pickup_model(
        precontact_pose, localization["camera_forward_base"])
    targets[key] = {
        "precontact_joint_values": delivery_api.validate_joint_values(
            list(arm.get_current_joint_values()),
            "ID%d current joint values" % item_id),
        "approach_axis_xyz_base":
            pickup_model["approach_axis_xyz_base"],
    }
    delivery_api.save_delivery_preset(args.delivery_file, delivery_preset)
    rospy.loginfo(
        "ID%d building contact point saved: joints=%s safe_axis=%s",
        item_id, targets[key]["precontact_joint_values"],
        targets[key]["approach_axis_xyz_base"])


def do_block_mono(args, config):
    action = get_action(args)
    if action == "calib_record" and args.known_z_mm is None:
        raise RuntimeError("--calib-record requires --known-z-mm.")
    if action in ("teach_block_pick_place", "teach_block_pregrasp",
                  "teach_block_place",
                  "teach_building_contact_release",
                  "preview_taught_block", "stop_at_taught_pre_grasp",
                  "run_taught_block"):
        require_taught_target(args, action)

    request_stream, response_stream = open_detector_streams(args)
    try:
        if action == "teach_building_contact_release":
            detector = DetectorClient(request_stream, response_stream)
            teach_building_contact_release(args, config, detector)
            return
        if action == "live_preview":
            detector = DetectorClient(request_stream, response_stream)
            run_live_preview(args, config, detector)
            return
        if action == "run_chassis_sequence":
            detector = DetectorClient(request_stream, response_stream)
            run_block_chassis_sequence(args, config, detector)
            return
        if action in ("teach_block_idle", "teach_block_carry"):
            localization = None
        else:
            detector = DetectorClient(request_stream, response_stream)
            localization = compute_block_localization(args, config, detector)
    finally:
        try:
            request_stream.close()
        except Exception:
            pass
        try:
            response_stream.close()
        except Exception:
            pass

    if action == "calib_record":
        return
    if action == "dry_run":
        rospy.logwarn("Dry run: no arm motion or pump command executed.")
        return
    if action in ("teach_block_pick_place", "teach_block_pregrasp",
                  "teach_block_place"):
        do_teach_block_mono(args, config, localization, action)
        return
    if action in ("teach_block_idle", "teach_block_carry"):
        do_teach_block_joints(args, config, action)
        return
    if action in ("preview_taught_block", "stop_at_taught_pre_grasp",
                  "run_taught_block"):
        do_run_taught_block_mono(args, config, localization, action)
        return

    raise RuntimeError("Unsupported block action: %s" % action)


def main():
    global MOTION_SETTLE_SECONDS
    signal.signal(signal.SIGTERM, raise_termination_requested)
    args = parse_args(sys.argv)
    enable_parent_death_signal(args.supervisor_pid)
    config = load_config(args.config)
    if args.base_frame:
        config["base_frame"] = args.base_frame
    if args.velocity_scale is not None:
        config["velocity_scale"] = args.velocity_scale
    if args.acceleration_scale is not None:
        config["acceleration_scale"] = args.acceleration_scale
    if args.planning_time is not None:
        config["planning_time"] = args.planning_time
    if args.tf_timeout is not None:
        config["tf_timeout"] = args.tf_timeout
    if args.approach_gap_mm is not None:
        distance = finite_scalar(
            args.approach_gap_mm, "--approach-gap-mm")
        if distance <= 0.0:
            raise RuntimeError("--approach-gap-mm must be positive.")
        config["approach_gap_mm"] = distance
    if args.confidence is not None:
        confidence = finite_scalar(args.confidence, "--confidence")
        if not 0.0 < confidence <= 1.0:
            raise RuntimeError("--confidence must be in (0, 1].")
        config["confidence_min"] = confidence
    MOTION_SETTLE_SECONDS = finite_scalar(
        config["motion_settle_seconds"], "motion_settle_seconds")

    moveit_initialized = False
    motion_lock = None
    exit_code = 0
    try:
        rospy.init_node("mirobot_pick_test", anonymous=True)
        motion_lock = acquire_motion_lock(args)
        if action_uses_moveit(args):
            moveit_commander.roscpp_initialize(sys.argv)
            moveit_initialized = True
        if args.mode == "home":
            do_home(config, args.group)
        elif args.mode == "pump":
            pump_proxy = get_pump_proxy()
            set_pump(pump_proxy, True)
            rospy.sleep(args.pump_seconds)
            set_pump(pump_proxy, False)
        elif args.mode == "current_pose":
            do_current_pose(config, args.group)
        elif args.mode == "wrist_forward":
            do_wrist_forward(config, args.group, args.wrist_forward_joint5)
        elif args.mode == "block_mono":
            do_block_mono(args, config)
        else:
            raise RuntimeError("Unsupported mode: %s" % args.mode)
        rospy.loginfo("Test finished.")
    except ContactProbeMiss as exc:
        rospy.logwarn("%s", ascii_log_text(exc))
        exit_code = CONTACT_PROBE_MISS_EXIT_CODE
    except Exception as exc:
        rospy.logerr("%s", ascii_log_text(exc))
        raise
    finally:
        if moveit_initialized:
            moveit_commander.roscpp_shutdown()
        if motion_lock is not None:
            motion_lock.close()
    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
