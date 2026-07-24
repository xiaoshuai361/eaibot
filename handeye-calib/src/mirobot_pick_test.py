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
    estimate_distance_mm,
    is_detection_usable,
    load_config,
    stable_median_observation,
)


try:
    STRING_TYPES = (basestring,)
except NameError:
    STRING_TYPES = (str,)


WRIST_FORWARD_JOINT5 = -1.5709534265016345
BLOCK_PRESET_VERSION = 1
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
    parser.add_argument("--stop-at-pre-grasp", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--calib-record", action="store_true")
    parser.add_argument("--teach-block", action="store_true")
    parser.add_argument("--run-taught-block", action="store_true")
    parser.add_argument("--preset-file", default=DEFAULT_BLOCK_PRESET_FILE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--place-approach-gap", type=float, default=0.02)
    parser.add_argument("--known-z-mm", type=float)
    parser.add_argument("--frames", type=int)
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


def build_block_motion_points(surface_base_mm, camera_forward_base,
                              tool_offset_base_mm, target_offset_mm,
                              pregrasp_distance_mm, suction_compression_mm):
    surface = finite_vector3(surface_base_mm, "surface_base_mm")
    forward = normalize_vector(camera_forward_base, "camera_forward_base")
    tool_offset = finite_vector3(tool_offset_base_mm, "tool_offset_base_mm")
    target_offset = finite_vector3(target_offset_mm, "target_offset_mm")
    pregrasp_distance_mm = finite_scalar(pregrasp_distance_mm, "pregrasp_distance_mm")
    suction_compression_mm = finite_scalar(suction_compression_mm, "suction_compression_mm")
    if pregrasp_distance_mm <= 0.0:
        raise RuntimeError("pregrasp_distance_mm must be positive.")
    if suction_compression_mm < 0.0:
        raise RuntimeError("suction_compression_mm must be non-negative.")

    surface_tcp = tuple(surface[i] + target_offset[i] for i in range(3))
    pregrasp_tcp = tuple(surface_tcp[i] - forward[i] * pregrasp_distance_mm
                         for i in range(3))
    contact_tcp = tuple(surface_tcp[i] + forward[i] * suction_compression_mm
                        for i in range(3))
    pregrasp_link = tuple(pregrasp_tcp[i] - tool_offset[i] for i in range(3))
    contact_link = tuple(contact_tcp[i] - tool_offset[i] for i in range(3))
    return {
        "surface_tcp_mm": surface_tcp,
        "pregrasp_link_mm": pregrasp_link,
        "contact_link_mm": contact_link,
    }


def require_motion_config(config, action):
    method = str(config.get("distance_method", "theory")).lower()
    if action in ("stop_at_pre_grasp", "execute") and method == "theory":
        raise RuntimeError("distance_method=theory is not allowed for motion.")
    if action in ("stop_at_pre_grasp", "execute"):
        if config.get("tool_offset_mm") is None:
            raise RuntimeError("tool_offset_mm is required for motion.")
        finite_vector3(config.get("tool_offset_mm"), "tool_offset_mm")
    return config


def format_triplet(values):
    return "({:.2f},{:.2f},{:.2f})".format(values[0], values[1], values[2])


def format_localization_summary(localization):
    return (
        "目标={target} 置信度={confidence:.3f} "
        "检测框=({x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}) "
        "框中心=({u:.2f},{v:.2f}) 框宽px={w:.2f} 框高px={h:.2f} "
        "距离方法={distance_method} 单目距离Z_mm={z_mm:.2f} "
        "相机坐标mm={camera_xyz} 机械臂坐标mm={base_xyz}"
    ).format(
        target=localization["target"],
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


def quaternion_to_matrix(values):
    x, y, z, w = normalize_quaternion(values)
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]


def quaternion_from_matrix(matrix):
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2][1] - matrix[1][2]) / scale
        qy = (matrix[0][2] - matrix[2][0]) / scale
        qz = (matrix[1][0] - matrix[0][1]) / scale
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
        qw = (matrix[2][1] - matrix[1][2]) / scale
        qx = 0.25 * scale
        qy = (matrix[0][1] + matrix[1][0]) / scale
        qz = (matrix[0][2] + matrix[2][0]) / scale
    elif matrix[1][1] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
        qw = (matrix[0][2] - matrix[2][0]) / scale
        qx = (matrix[0][1] + matrix[1][0]) / scale
        qy = 0.25 * scale
        qz = (matrix[1][2] + matrix[2][1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
        qw = (matrix[1][0] - matrix[0][1]) / scale
        qx = (matrix[0][2] + matrix[2][0]) / scale
        qy = (matrix[1][2] + matrix[2][1]) / scale
        qz = 0.25 * scale
    return normalize_quaternion([qx, qy, qz, qw])


def pose_to_matrix(pose_stamped):
    rotation = quaternion_to_matrix(quaternion_msg_to_tuple(pose_stamped.pose.orientation))
    position = pose_stamped.pose.position
    return [
        [rotation[0][0], rotation[0][1], rotation[0][2], float(position.x)],
        [rotation[1][0], rotation[1][1], rotation[1][2], float(position.y)],
        [rotation[2][0], rotation[2][1], rotation[2][2], float(position.z)],
        [0.0, 0.0, 0.0, 1.0],
    ]


def transform_to_matrix(transform):
    rotation = quaternion_to_matrix(transform["orientation_xyzw"])
    position = transform["position"]
    return [
        [rotation[0][0], rotation[0][1], rotation[0][2], float(position[0])],
        [rotation[1][0], rotation[1][1], rotation[1][2], float(position[1])],
        [rotation[2][0], rotation[2][1], rotation[2][2], float(position[2])],
        [0.0, 0.0, 0.0, 1.0],
    ]


def matrix_multiply(first, second):
    result = [[0.0 for _column in range(4)] for _row in range(4)]
    for row in range(4):
        for column in range(4):
            result[row][column] = sum(
                first[row][index] * second[index][column]
                for index in range(4))
    return result


def inverse_rigid_matrix(matrix):
    result = [
        [matrix[0][0], matrix[1][0], matrix[2][0], 0.0],
        [matrix[0][1], matrix[1][1], matrix[2][1], 0.0],
        [matrix[0][2], matrix[1][2], matrix[2][2], 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    translation = [matrix[0][3], matrix[1][3], matrix[2][3]]
    for row in range(3):
        result[row][3] = -sum(result[row][index] * translation[index]
                              for index in range(3))
    return result


def matrix_to_pose(frame_id, matrix):
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.header.stamp = rospy.Time.now() if "rospy" in globals() else None
    pose.pose.position.x = matrix[0][3]
    pose.pose.position.y = matrix[1][3]
    pose.pose.position.z = matrix[2][3]
    qx, qy, qz, qw = quaternion_from_matrix(matrix)
    pose.pose.orientation.x = qx
    pose.pose.orientation.y = qy
    pose.pose.orientation.z = qz
    pose.pose.orientation.w = qw
    return pose


def matrix_to_transform(matrix):
    return {
        "position": [matrix[0][3], matrix[1][3], matrix[2][3]],
        "orientation_xyzw": quaternion_from_matrix(matrix),
    }


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
    return matrix_to_pose(frame_id, transform_to_matrix(transform))


def compute_grasp_ee_in_block(block_anchor_pose, ee_base_pose):
    block_in_base = pose_to_matrix(block_anchor_pose)
    ee_in_base = pose_to_matrix(ee_base_pose)
    ee_in_block = matrix_multiply(inverse_rigid_matrix(block_in_base), ee_in_base)
    return matrix_to_transform(ee_in_block)


def compute_taught_grasp_pose(block_anchor_pose, grasp_ee_in_block, base_frame):
    grasp_matrix = matrix_multiply(
        pose_to_matrix(block_anchor_pose),
        transform_to_matrix(grasp_ee_in_block))
    return matrix_to_pose(base_frame, grasp_matrix)


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
        raise RuntimeError("Unsupported preset version.")
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
    print(message)
    try:
        text = raw_input("> ")
    except NameError:
        text = input("> ")
    if text.strip().lower() in ("q", "quit", "exit"):
        raise RuntimeError("User aborted taught block workflow.")


def apply_fixed_orientation_if_configured(pose_stamped, config):
    if config.get("fixed_orientation_xyzw") is None:
        return pose_stamped
    qx, qy, qz, qw = normalize_quaternion(config["fixed_orientation_xyzw"])
    pose_stamped.pose.orientation.x = qx
    pose_stamped.pose.orientation.y = qy
    pose_stamped.pose.orientation.z = qz
    pose_stamped.pose.orientation.w = qw
    return pose_stamped


def build_teach_assist_pose(localization, orientation, config):
    surface_pose = pose_from_base_mm(
        localization["base_frame"], localization["base_xyz_mm"], orientation)
    return build_pregrasp_from_grasp(
        surface_pose,
        localization["camera_forward_base"],
        config.get("pregrasp_distance_mm", 80.0),
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
        ("stop_at_pre_grasp", args.stop_at_pre_grasp),
        ("execute", args.execute),
        ("calib_record", args.calib_record),
        ("teach_block", args.teach_block),
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
    return {"rgb": rgb, "camera_info": camera_info, "header": copy.deepcopy(image_msg.header)}


def camera_info_intrinsics(camera_info):
    k = list(camera_info.K)
    if len(k) != 9:
        raise RuntimeError("CameraInfo.K must contain 9 values.")
    fx, fy, cx, cy = [finite_scalar(value, "CameraInfo.K")
                     for value in (k[0], k[4], k[2], k[5])]
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


def show_rgb_debug(image, detection, observation, milliseconds=1):
    try:
        import cv2
        if isinstance(detection, (list, tuple)):
            debug = draw_debug_detections(image, detection, observation)
        else:
            debug = draw_debug_image(image, detection, observation)
        cv2.imshow("Block mono RGB - q/Esc to close", debug)
        if int(milliseconds) == 0:
            rospy.loginfo("RGB检测窗口已保持显示：按 q 或 Esc 退出窗口并结束 dry-run。")
            while True:
                key = cv2.waitKey(100) & 0xFF
                if key in (27, ord("q")):
                    return False
        else:
            key = cv2.waitKey(int(milliseconds)) & 0xFF
            if key in (27, ord("q")):
                return False
    except Exception as exc:
        rospy.logwarn("Could not show RGB debug image: %s", exc)
    return True


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
            rospy.logwarn("YOLO all-target detection failed: %s", exc)
            continue
        for detection in detections:
            try:
                target = detection.get("target")
                if not isinstance(target, STRING_TYPES) or not target:
                    rospy.logwarn("Rejected YOLO detection without target name.")
                    continue
                usable, reason = is_detection_usable(detection, rules)
                if not usable:
                    rospy.logwarn("Rejected YOLO detection %s: %s", target, reason)
                    continue
                observation = detection_to_observation(detection)
                observation["target"] = target
            except Exception as exc:
                rospy.logwarn("YOLO detection parse failed: %s", exc)
                continue
            observations_by_target.setdefault(target, []).append(observation)
            frame_detections.append(detection)
            frame_observations.append(observation)
        user_requested_stop = False
        if args.show_rgb and frame_detections:
            if not show_rgb_debug(capture["rgb"], frame_detections, frame_observations, 1):
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
    last_capture = None
    max_attempts = max(frames_required * 4, frames_required + 10)
    if action == "calib_record":
        print("known_z_mm,target,conf,x1,y1,x2,y2,u,v,w,h")
    for _attempt in range(max_attempts):
        capture = capture_rgb_once(config)
        last_capture = capture
        try:
            detection = request_detection(detector, args.block_target, capture["rgb"])
            usable, reason = is_detection_usable(detection, rules)
            if not usable:
                rospy.logwarn("Rejected YOLO detection: %s", reason)
                continue
            observation = detection_to_observation(detection)
        except Exception as exc:
            rospy.logwarn("YOLO detection failed: %s", exc)
            continue
        observations.append(observation)
        user_requested_stop = False
        if args.show_rgb:
            if not show_rgb_debug(capture["rgb"], detection, observation, 1):
                user_requested_stop = True
        if action == "calib_record":
            print("{:.2f},{},{:.6f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}".format(
                args.known_z_mm, args.block_target, observation["confidence"],
                observation["box"][0], observation["box"][1],
                observation["box"][2], observation["box"][3],
                observation["u"], observation["v"], observation["w"], observation["h"]))
        if len(observations) >= frames_required:
            return observations, last_capture
        if user_requested_stop:
            break
    raise RuntimeError(
        "Only collected %d valid YOLO observations, need %d." %
        (len(observations), frames_required)
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
    rospy.loginfo(summary)
    print(summary)
    if args.show_rgb and action == "dry_run":
        show_rgb_debug(
            capture["rgb"],
            localization_debug_detection(localization),
            localization_debug_observation(localization),
            0,
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
    z_mm = estimate_distance_mm(
        config.get("distance_method", "theory"),
        stable["w"],
        fx,
        config["target_size_mm"],
        target,
        config.get("distance_models", {}),
        config.get("fixed_z_mm"),
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
                "目标=%s 有效帧不足：%d/%d，跳过坐标计算。",
                target, len(observations), frames_required)
            continue
        try:
            localization = build_localization_from_observations(
                target, observations, capture, args, config, listener)
        except Exception as exc:
            rospy.logwarn("目标=%s 坐标计算失败：%s", target, exc)
            continue
        localizations.append(localization)
        summary = format_localization_summary(localization)
        rospy.loginfo(summary)
        print(summary)
    if not localizations:
        raise RuntimeError("No target had enough stable observations for localization.")
    count_text = "YOLO稳定识别到{}个目标。".format(len(localizations))
    rospy.loginfo(count_text)
    print(count_text)
    if args.show_rgb:
        show_rgb_debug(
            capture["rgb"],
            [localization_debug_detection(item) for item in localizations],
            [localization_debug_observation(item) for item in localizations],
            0,
        )
    return localizations


def quaternion_msg_to_tuple(quaternion):
    return (quaternion.x, quaternion.y, quaternion.z, quaternion.w)


def rotate_vector_by_quaternion(vector, quaternion_xyzw):
    vector = finite_vector3(vector, "vector")
    qx, qy, qz, qw = [finite_scalar(value, "quaternion")
                     for value in quaternion_xyzw]
    length = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if length <= 0.0:
        raise RuntimeError("quaternion must be non-zero.")
    qx, qy, qz, qw = qx / length, qy / length, qz / length, qw / length
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


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
    arm.set_start_state_to_current_state()
    arm.set_pose_target(target_pose)
    success = arm.go(wait=True)
    arm.stop()
    arm.clear_pose_targets()
    if not success:
        raise RuntimeError("MoveIt failed during %s." % label)


def execute_cartesian_pose(arm, target_pose, label):
    rospy.loginfo("Executing cartesian %s", label)
    arm.set_start_state_to_current_state()
    plan, fraction = arm.compute_cartesian_path(
        [copy.deepcopy(target_pose.pose)], 0.003, 0.0, True)
    if fraction < 0.999:
        raise RuntimeError("MoveIt cartesian path failed during %s." % label)
    success = arm.execute(plan, wait=True)
    arm.stop()
    arm.clear_pose_targets()
    if not success:
        raise RuntimeError("MoveIt execute failed during %s." % label)


def get_mirobot_pump_type():
    try:
        from mirobot_urdf_2.srv import mirobotPump
        return mirobotPump
    except ImportError:
        raise RuntimeError("未加载 mirobot_urdf_2.srv，请先 source 机械臂工作空间。")


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


def do_teach_block_mono(args, config, localization):
    target = require_taught_target(args, "teach_block")
    arm = build_move_group(config, args.group)
    current_pose = apply_fixed_orientation_if_configured(arm.get_current_pose(), config)
    assist_pose = build_teach_assist_pose(
        localization, current_pose.pose.orientation, config)
    rospy.loginfo(pose_to_text("block_teach_assist_front", assist_pose))
    prompt_enter(
        "步骤 1：准备移动到目标 %s 前方安全点。\n"
        "确认路径安全后按 Enter；输入 q 取消。" % target)
    execute_pose(arm, assist_pose, "block_teach_assist_front")

    prompt_enter(
        "步骤 2：请在 RViz 里微调吸盘到真正能吸住 %s 的接触姿态。\n"
        "Plan/Execute 到位后回到这里按 Enter 记录抓取姿态；输入 q 取消。" % target)
    grasp_pose = arm.get_current_pose()
    rospy.loginfo(pose_to_text("block_taught_grasp", grasp_pose))

    prompt_enter(
        "步骤 3：请在 RViz 里把吸盘移动到该物资的载物仓释放位置。\n"
        "Plan/Execute 到位后回到这里按 Enter 记录放置姿态；输入 q 取消。")
    place_pose = arm.get_current_pose()
    rospy.loginfo(pose_to_text("block_taught_place", place_pose))

    anchor_pose = block_anchor_pose_from_localization(localization, config)
    preset = load_or_create_block_preset(args.preset_file, config)
    targets = preset.setdefault("targets", {})
    if target in targets and not args.overwrite:
        raise RuntimeError(
            "Preset already contains target %s. Use --overwrite to replace it." % target)
    targets[target] = {
        "grasp_ee_in_block": compute_grasp_ee_in_block(anchor_pose, grasp_pose),
        "place_ee_in_base": pose_to_transform(place_pose),
    }
    preset["base_frame"] = config.get("base_frame", localization["base_frame"])
    save_block_preset(args.preset_file, preset, overwrite=True)
    rospy.loginfo("Saved taught block preset for %s: %s", target, args.preset_file)
    print("已保存无Tag示教：目标={} preset={}".format(target, args.preset_file))


def do_run_taught_block_mono(args, config, localization):
    target = require_taught_target(args, "run_taught_block")
    preset = load_block_preset(args.preset_file)
    entry = preset.get("targets", {}).get(target)
    if not isinstance(entry, dict):
        raise RuntimeError("Preset file is missing target %s." % target)
    arm = build_move_group(config, args.group)
    anchor_pose = block_anchor_pose_from_localization(localization, config)
    grasp_pose = compute_taught_grasp_pose(
        anchor_pose, entry["grasp_ee_in_block"], localization["base_frame"])
    pre_grasp_pose = build_pregrasp_from_grasp(
        grasp_pose,
        localization["camera_forward_base"],
        config.get("pregrasp_distance_mm", 80.0),
        localization["base_frame"],
    )
    place_pose = transform_to_pose(
        localization["base_frame"], entry["place_ee_in_base"])
    pre_place_pose = build_pre_place_pose(
        place_pose, args.place_approach_gap, localization["base_frame"])

    rospy.loginfo(pose_to_text("taught_block_pre_grasp", pre_grasp_pose))
    rospy.loginfo(pose_to_text("taught_block_grasp", grasp_pose))
    rospy.loginfo(pose_to_text("taught_block_pre_place", pre_place_pose))
    rospy.loginfo(pose_to_text("taught_block_place", place_pose))

    pump_proxy = get_pump_proxy()
    execute_pose(arm, pre_grasp_pose, "taught_block_pre_grasp")
    execute_cartesian_pose(arm, grasp_pose, "taught_block_grasp")
    set_pump(pump_proxy, True)
    rospy.sleep(0.8)
    execute_cartesian_pose(arm, pre_grasp_pose, "taught_block_retreat")
    execute_pose(arm, pre_place_pose, "taught_block_pre_place")
    execute_cartesian_pose(arm, place_pose, "taught_block_place")
    set_pump(pump_proxy, False)
    rospy.sleep(0.5)
    execute_cartesian_pose(arm, pre_place_pose, "taught_block_place_retreat")


def do_block_mono(args, config):
    action = get_action(args)
    if action == "calib_record" and args.known_z_mm is None:
        raise RuntimeError("--calib-record requires --known-z-mm.")
    if action in ("teach_block", "run_taught_block"):
        require_taught_target(args, action)
    if action in ("stop_at_pre_grasp", "execute"):
        require_motion_config(config, action)

    request_stream, response_stream = open_detector_streams(args)
    try:
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
    if action == "teach_block":
        do_teach_block_mono(args, config, localization)
        return
    if action == "run_taught_block":
        do_run_taught_block_mono(args, config, localization)
        return

    group_name = args.group
    arm = build_move_group(config, group_name)
    current_pose = arm.get_current_pose()
    if config.get("fixed_orientation_xyzw") is not None:
        q = config["fixed_orientation_xyzw"]
        current_pose.pose.orientation.x = q[0]
        current_pose.pose.orientation.y = q[1]
        current_pose.pose.orientation.z = q[2]
        current_pose.pose.orientation.w = q[3]
    orientation = current_pose.pose.orientation
    tool_offset_base = rotate_vector_by_quaternion(
        config["tool_offset_mm"], quaternion_msg_to_tuple(orientation))
    points = build_block_motion_points(
        localization["base_xyz_mm"],
        localization["camera_forward_base"],
        tool_offset_base,
        config.get("target_offset_mm", [0.0, 0.0, 0.0]),
        config.get("pregrasp_distance_mm", 50.0),
        config.get("suction_compression_mm", 3.0),
    )
    validate_workspace(points["pregrasp_link_mm"], config, "pregrasp")
    validate_workspace(points["contact_link_mm"], config, "contact")
    pregrasp = pose_from_base_mm(
        localization["base_frame"], points["pregrasp_link_mm"], orientation)
    contact = pose_from_base_mm(
        localization["base_frame"], points["contact_link_mm"], orientation)

    execute_pose(arm, pregrasp, "block_pre_grasp")
    if action == "stop_at_pre_grasp":
        rospy.logwarn("Stopped at pre-grasp. Pump was not enabled.")
        return

    pump_proxy = get_pump_proxy()
    set_pump(pump_proxy, False)
    rospy.sleep(0.3)
    execute_cartesian_pose(arm, contact, "block_contact")
    rospy.sleep(0.2)
    set_pump(pump_proxy, True)
    rospy.sleep(0.8)
    execute_cartesian_pose(arm, pregrasp, "block_retreat")


def main():
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
    if args.confidence is not None:
        confidence = finite_scalar(args.confidence, "--confidence")
        if not 0.0 < confidence <= 1.0:
            raise RuntimeError("--confidence must be in (0, 1].")
        config["confidence_min"] = confidence

    rospy.init_node("mirobot_pick_test", anonymous=False)
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
        rospy.logerr(str(exc))
        raise
    finally:
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()
