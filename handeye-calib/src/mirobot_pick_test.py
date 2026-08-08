#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Python 2 ROS/MoveIt helper for monocular tagless-block suction."""

from __future__ import absolute_import, division, print_function

import argparse
import copy
import json
import math
import os
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
from geometry_msgs.msg import PoseStamped

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
    stable_median_observation,
)


try:
    STRING_TYPES = (basestring,)
except NameError:
    STRING_TYPES = (str,)


WRIST_FORWARD_JOINT5 = -1.5709534265016345
BLOCK_PRESET_VERSION = 2
MOTION_SETTLE_SECONDS = 0.25
DEFAULT_BLOCK_PRESET_FILE = (
    "/home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json"
)


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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--live-preview", action="store_true")
    parser.add_argument("--calib-record", action="store_true")
    parser.add_argument("--teach-block-grasp", action="store_true")
    parser.add_argument("--teach-block-place", action="store_true")
    parser.add_argument("--teach-block-idle", action="store_true")
    parser.add_argument("--teach-block-carry", action="store_true")
    parser.add_argument("--preview-taught-block", action="store_true")
    parser.add_argument("--stop-at-taught-pre-grasp", action="store_true")
    parser.add_argument("--run-taught-block", action="store_true")
    parser.add_argument("--preset-file", default=DEFAULT_BLOCK_PRESET_FILE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reset-pickup-model", action="store_true")
    parser.add_argument("--place-approach-gap", type=float, default=0.02)
    parser.add_argument("--known-z-mm", type=float)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--preview-hz", type=float, default=1.0)
    parser.add_argument("--pregrasp-distance-mm", type=float)
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--show-rgb", action="store_true")
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
    base_frame = localization.get("base_frame") or config.get("base_frame", "base")
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


def create_block_pickup_model(grasp_pose, camera_forward_base):
    forward = normalize_vector(camera_forward_base, "camera_forward_base")
    return {
        "orientation_xyzw_base": list(normalize_quaternion(
            quaternion_msg_to_tuple(grasp_pose.pose.orientation))),
        # This axis points from contact back toward the camera/safe side.
        "approach_axis_xyz_base": [-forward[0], -forward[1], -forward[2]],
    }


def compute_block_grasp_offset(anchor_pose, grasp_pose):
    return [
        float(grasp_pose.pose.position.x - anchor_pose.pose.position.x),
        float(grasp_pose.pose.position.y - anchor_pose.pose.position.y),
        float(grasp_pose.pose.position.z - anchor_pose.pose.position.z),
    ]


def require_block_pickup_model(preset):
    model = preset.get("pickup_model")
    if not isinstance(model, dict):
        raise RuntimeError(
            "Preset has no pickup_model. Re-teach one block grasp with the new workflow.")
    normalize_quaternion(model.get("orientation_xyzw_base"))
    normalize_vector(model.get("approach_axis_xyz_base"),
                     "approach_axis_xyz_base")
    return model


def compute_constrained_block_grasp_pose(anchor_pose, pickup_model, entry,
                                         base_frame):
    offset = finite_vector3(entry.get("grasp_offset_xyz_base"),
                            "grasp_offset_xyz_base")
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


def build_constrained_block_pregrasp(grasp_pose, pickup_model, gap_mm,
                                     base_frame):
    axis = normalize_vector(
        pickup_model.get("approach_axis_xyz_base"),
        "approach_axis_xyz_base")
    gap_m = finite_scalar(gap_mm, "pregrasp_distance_mm") * 0.001
    if gap_m <= 0.0:
        raise RuntimeError("pregrasp_distance_mm must be positive.")
    pose = copy.deepcopy(grasp_pose)
    pose.header.frame_id = base_frame
    pose.header.stamp = rospy.Time.now() if "rospy" in globals() else None
    pose.pose.position.x += axis[0] * gap_m
    pose.pose.position.y += axis[1] * gap_m
    pose.pose.position.z += axis[2] * gap_m
    return pose


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
        "base_frame": config.get("base_frame", "base"),
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
    try:
        text = raw_input("> ")
    except NameError:
        text = input("> ")
    if text.strip().lower() in ("q", "quit", "exit"):
        raise RuntimeError("User aborted taught block workflow.")


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
        ("teach_block_grasp", args.teach_block_grasp),
        ("teach_block_place", args.teach_block_place),
        ("teach_block_idle", args.teach_block_idle),
        ("teach_block_carry", args.teach_block_carry),
        ("preview_taught_block", args.preview_taught_block),
        ("stop_at_taught_pre_grasp", args.stop_at_taught_pre_grasp),
        ("run_taught_block", args.run_taught_block),
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
    timeout = finite_scalar(config.get("rgb_timeout", 5.0), "rgb_timeout")
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
        config.get("image_max_age_seconds", 1.0), "image_max_age_seconds")
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
        try:
            capture = capture_rgb_once(config)
            response = request_detection(
                detector, args.block_target, capture["rgb"])
            detections = response_detections(response)
            observations = [detection_to_observation(item) for item in detections]
            new_labels = new_live_preview_labels(detections, reported_targets)
            if new_labels:
                print_utf8(u"检测到：" + u"，".join(new_labels))
            if not show_rgb_debug(
                    capture["rgb"], detections, observations, milliseconds=1,
                    roi_ratio=config.get(
                        "grasp_roi_ratio", DEFAULT_CONFIG["grasp_roi_ratio"])):
                return
        except Exception as exc:
            error_text = safe_log_text(exc)
            if "No usable YOLO detections" not in error_text:
                rospy.logwarn(
                    "Live preview frame failed: %s", ascii_log_text(error_text))
        remaining = period - (time.time() - started)
        if remaining > 0.0:
            rospy.sleep(remaining)


def collect_all_observations(args, config, detector):
    action = get_action(args)
    frames_required = int(args.frames or config.get("frames_required", 10))
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
    frames_required = int(args.frames or config.get("frames_required", 10))
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
        config.get("observation_timeout", 12.0), "observation_timeout")
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
    z_mm = estimate_distance_from_box_mm(
        config.get("distance_method", "theory"),
        stable["w"],
        stable["h"],
        fx,
        fy,
        config["target_size_mm"],
        config.get("target_height_mm", config["target_size_mm"]),
        target,
        config.get("distance_models", {}),
        config.get("fixed_z_mm"),
        config.get("max_axis_distance_disagreement_mm", 20.0),
    )
    camera_xyz = deproject_pixel_to_camera_mm(
        stable["u"], stable["v"], z_mm, fx, fy, cx, cy)
    camera_frame = getattr(capture["header"], "frame_id", "") or config["camera_frame"]
    base_frame = config.get("base_frame", "base")
    tf_timeout = finite_scalar(config.get("tf_timeout", 5.0), "tf_timeout")
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
        "distance_method": config.get("distance_method", "theory"),
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
    z_min = finite_scalar(config.get("base_min_z_mm", 40.0), "base_min_z_mm")
    radius_max = finite_scalar(config.get("base_max_radius_mm", 500.0), "base_max_radius_mm")
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
    arm.set_pose_reference_frame(config.get("base_frame", "base"))
    arm.allow_replanning(True)
    arm.set_max_velocity_scaling_factor(float(config.get("velocity_scale", 0.05)))
    arm.set_max_acceleration_scaling_factor(float(config.get("acceleration_scale", 0.05)))
    arm.set_planning_time(float(config.get("planning_time", 5.0)))
    return arm


def execute_pose(arm, target_pose, label):
    rospy.loginfo("Executing %s", label)
    for attempt in range(2):
        arm.set_start_state_to_current_state()
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
                           fallback_to_pose=False):
    rospy.loginfo("Executing cartesian %s", label)
    for attempt in range(2):
        arm.set_start_state_to_current_state()
        plan, fraction = arm.compute_cartesian_path(
            [copy.deepcopy(target_pose.pose)], 0.005, 0.0, True)
        if fraction < 0.999 and retry_without_collisions:
            arm.set_start_state_to_current_state()
            plan, fraction = arm.compute_cartesian_path(
                [copy.deepcopy(target_pose.pose)], 0.005, 0.0, False)
        if fraction < 0.999:
            if fallback_to_pose:
                execute_pose(arm, target_pose, label + "_pose_fallback")
                return
            raise RuntimeError(
                "MoveIt cartesian path failed during %s (fraction=%.3f)."
                % (label, fraction))
        if not plan.joint_trajectory.points:
            raise RuntimeError("MoveIt returned an empty cartesian plan for %s." % label)
        success = arm.execute(plan, wait=True)
        arm.stop()
        arm.clear_pose_targets()
        if success:
            if MOTION_SETTLE_SECONDS > 0.0:
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


def set_pump(pump_proxy, enabled):
    rospy.loginfo("Pump %s", "ON" if enabled else "OFF")
    response = pump_proxy(enabled)
    if not response.Sucess:
        raise RuntimeError("Pump service returned failure.")


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


def _require_overwrite(entry, field, args, target):
    if field in entry and not args.overwrite:
        raise RuntimeError(
            "Preset target %s already contains %s. Use --overwrite to update it."
            % (target, field))


def record_block_grasp(args, config, localization, arm, preset):
    target = require_taught_target(args, "teach_block_grasp")
    targets = preset.setdefault("targets", {})
    entry = targets.setdefault(target, {})
    _require_overwrite(entry, "grasp_offset_xyz_base", args, target)

    pickup_model = preset.get("pickup_model")
    prompt_enter(
        u"示教模式不会自动移动机械臂，也不会自动改变 joint5。\n"
        u"请在 RViz 中 Plan/Execute 到能可靠吸住 %s 的接触姿态，"
        u"确认到位后按 Enter 记录；不要用手掰机械臂。" %
        safe_log_text(target))
    grasp_pose = arm.get_current_pose()
    anchor_pose = block_anchor_pose_from_localization(localization, config)
    if not isinstance(pickup_model, dict) or args.reset_pickup_model:
        preset["pickup_model"] = create_block_pickup_model(
            grasp_pose, localization["camera_forward_base"])
        rospy.loginfo("Locked the shared suction orientation and approach axis.")
    entry["grasp_offset_xyz_base"] = compute_block_grasp_offset(
        anchor_pose, grasp_pose)
    rospy.loginfo(pose_to_text("block_taught_grasp", grasp_pose))
    return entry


def record_block_place(args, config, arm, preset):
    target = require_taught_target(args, "teach_block_place")
    entry = preset.setdefault("targets", {}).setdefault(target, {})
    _require_overwrite(entry, "place_ee_in_base", args, target)
    prompt_enter(
        u"请在 RViz 中移动到 %s 对应载物仓的释放姿态。\n"
        u"Plan/Execute 到位后按 Enter 记录；不要用手掰机械臂。" %
        safe_log_text(target))
    place_pose = arm.get_current_pose()
    entry["place_ee_in_base"] = pose_to_transform(place_pose)
    rospy.loginfo(pose_to_text("block_taught_place", place_pose))
    return entry


def do_teach_block_mono(args, config, localization, action):
    arm = build_move_group(config, args.group)
    preset = load_or_create_block_preset(args.preset_file, config)
    target = args.block_target
    if action == "teach_block_grasp":
        record_block_grasp(args, config, localization, arm, preset)
    else:
        record_block_place(args, config, arm, preset)
    preset["base_frame"] = config.get("base_frame", "base")
    save_block_preset(args.preset_file, preset, overwrite=True)
    rospy.loginfo(
        "Saved taught block preset for %s: %s",
        ascii_log_text(target), ascii_log_text(args.preset_file))


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
    entry = preset.get("targets", {}).get(target)
    if not isinstance(entry, dict):
        raise RuntimeError("Preset file is missing target %s." % target)
    if "grasp_offset_xyz_base" not in entry or "place_ee_in_base" not in entry:
        raise RuntimeError(
            "Preset target %s must contain grasp_offset_xyz_base and place_ee_in_base."
            % target)
    pickup_model = require_block_pickup_model(preset)
    arm = build_move_group(config, args.group)
    anchor_pose = block_anchor_pose_from_localization(localization, config)
    grasp_pose = compute_constrained_block_grasp_pose(
        anchor_pose, pickup_model, entry, localization["base_frame"])
    pre_grasp_pose = build_constrained_block_pregrasp(
        grasp_pose, pickup_model,
        config.get("pregrasp_distance_mm", 80.0),
        localization["base_frame"],
    )
    place_pose = transform_to_pose(
        localization["base_frame"], entry["place_ee_in_base"])
    pre_place_pose = build_pre_place_pose(
        place_pose, args.place_approach_gap, localization["base_frame"])

    for label, pose in (
        ("taught_block_pre_grasp", pre_grasp_pose),
        ("taught_block_grasp", grasp_pose),
        ("taught_block_pre_place", pre_place_pose),
        ("taught_block_place", place_pose),
    ):
        validate_pose_workspace(pose, config, label)

    rospy.loginfo(pose_to_text("taught_block_pre_grasp", pre_grasp_pose))
    rospy.loginfo(pose_to_text("taught_block_grasp", grasp_pose))
    rospy.loginfo(pose_to_text("taught_block_pre_place", pre_place_pose))
    rospy.loginfo(pose_to_text("taught_block_place", place_pose))

    if action == "preview_taught_block":
        rospy.logwarn("Preview only: no arm motion or pump command executed.")
        return
    if action == "stop_at_taught_pre_grasp":
        execute_pose(arm, pre_grasp_pose, "taught_block_pre_grasp")
        rospy.logwarn("Stopped at taught pre-grasp; pump was not enabled.")
        return

    pump_proxy = get_pump_proxy()
    holding_object = False
    try:
        execute_pose(arm, pre_grasp_pose, "taught_block_pre_grasp")
        execute_cartesian_pose(arm, grasp_pose, "taught_block_grasp")
        set_pump(pump_proxy, True)
        holding_object = True
        rospy.sleep(0.8)
        execute_cartesian_pose(arm, pre_grasp_pose, "taught_block_retreat")
        if preset.get("carry_joint_values"):
            execute_joint_values(arm, preset["carry_joint_values"], "block_carry")
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


def do_block_mono(args, config):
    action = get_action(args)
    if action == "calib_record" and args.known_z_mm is None:
        raise RuntimeError("--calib-record requires --known-z-mm.")
    if action in ("teach_block_grasp", "teach_block_place",
                  "preview_taught_block", "stop_at_taught_pre_grasp",
                  "run_taught_block"):
        require_taught_target(args, action)

    request_stream, response_stream = open_detector_streams(args)
    try:
        if action == "live_preview":
            detector = DetectorClient(request_stream, response_stream)
            run_live_preview(args, config, detector)
            return
        if action in ("teach_block_place", "teach_block_idle", "teach_block_carry"):
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
    if action in ("teach_block_grasp", "teach_block_place"):
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
    args = parse_args(sys.argv)
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
    if args.pregrasp_distance_mm is not None:
        distance = finite_scalar(
            args.pregrasp_distance_mm, "--pregrasp-distance-mm")
        if distance <= 0.0:
            raise RuntimeError("--pregrasp-distance-mm must be positive.")
        config["pregrasp_distance_mm"] = distance
    if args.confidence is not None:
        confidence = finite_scalar(args.confidence, "--confidence")
        if not 0.0 < confidence <= 1.0:
            raise RuntimeError("--confidence must be in (0, 1].")
        config["confidence_min"] = confidence
    MOTION_SETTLE_SECONDS = finite_scalar(
        config.get("motion_settle_seconds", 0.25), "motion_settle_seconds")

    rospy.init_node("mirobot_pick_test", anonymous=True)
    moveit_commander.roscpp_initialize(sys.argv)
    try:
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
    except Exception as exc:
        rospy.logerr("%s", ascii_log_text(exc))
        raise
    finally:
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()
