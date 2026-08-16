#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import copy
import errno
import json
import math
import os
import select
import sys

if sys.version_info[0] != 2:
    sys.stderr.write(
        'mirobot_pick_test_tag.py must run with Python 2 because this ROS '
        'Melodic workspace provides tf/moveit for Python 2.\n'
    )
    sys.exit(1)

import rospy
import tf
import moveit_commander
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray


PRESET_VERSION = 3
DEFAULT_SEQUENCE = '1,2,3,4'
DEFAULT_PRESET_FILE = '/home/eaibot/handeye-calib/config/tag_pick_place_presets.json'
DEFAULT_ASSIST_FRONT_GAP = 0.065
DEFAULT_ASSIST_ORIENTATION_XYZW = '0,0,0,1'
DEFAULT_TEACH_SETTLE_SECONDS = 0.8
DEFAULT_MOTION_SETTLE_SECONDS = 0.25
DEFAULT_TAG_MIN_SAMPLES = 3
DEFAULT_TAG_MAX_MAD_M = 0.005
DEFAULT_TAG_MAX_AGE_SECONDS = 2.0
DEFAULT_PICKUP_APPROACH_AXIS_BASE = '-1,0,0'
DEFAULT_SHARED_GRASP_REFERENCE_TAG = 2
DEFAULT_VELOCITY_SCALE = 0.4
DEFAULT_ACCELERATION_SCALE = 0.4
DEFAULT_APPROACH_GAP = 0.030
DEFAULT_PLACE_APPROACH_GAP = 0.05
CONTACT_STAGING_STEP_M = 0.005
CONTACT_PROBE_STEP_M = 0.002
CONTACT_PROBE_MAX_TRAVEL_M = 0.065
CONTACT_RETREAT_EXTRA_M = 0.030
CONTACT_PROBE_POLL_SECONDS = 0.02
CONTACT_PROBE_EXPECTED_POINT_SECONDS = 0.5
CONTACT_PROBE_ENABLE_SERVICE = '/mirobot_contact_probe_enable'
CONTACT_STATE_SERVICE = '/mirobot_contact_state'
CONTACT_PROBE_MISS_EXIT_CODE = 4
POSE_DONE_POSITION_TOLERANCE = 0.015
POSE_DONE_ORIENTATION_TOLERANCE_RAD = 0.35
DEFAULT_STARTUP_HOME_SERVICE = '/mirobot_startup_home'
MOTION_SETTLE_SECONDS = DEFAULT_MOTION_SETTLE_SECONDS
STAGING_FALLBACK_GAPS_M = (0.020, 0.010, 0.0)
STAGING_FALLBACK_INTERVAL_SECONDS = 0.1

try:
    STRING_TYPES = (basestring,)
except NameError:
    STRING_TYPES = (str,)


class UserAbort(Exception):
    pass


class ContactProbeIncomplete(Exception):
    def __init__(self, tag_ids):
        self.tag_ids = list(tag_ids)
        Exception.__init__(
            self,
            '限位探测未触发，未完成的 tag：%s。'
            % ','.join(str(tag_id) for tag_id in self.tag_ids))


def parse_sequence(text):
    if not isinstance(text, STRING_TYPES) or not text.strip():
        raise RuntimeError('--sequence must be a comma separated list of tag IDs.')
    result = []
    for item in text.split(','):
        value = item.strip()
        if not value:
            continue
        try:
            tag_id = int(value)
        except (TypeError, ValueError):
            raise RuntimeError('--sequence values must be positive integer tag IDs.')
        if tag_id <= 0:
            raise RuntimeError('--sequence values must be positive integer tag IDs.')
        result.append(tag_id)
    if not result:
        raise RuntimeError('--sequence must contain at least one tag ID.')
    return result


def parse_quaternion_text(text, option):
    if not isinstance(text, STRING_TYPES):
        raise RuntimeError('%s must be x,y,z,w.' % option)
    parts = [part.strip() for part in text.split(',')]
    if len(parts) != 4:
        raise RuntimeError('%s must be x,y,z,w.' % option)
    try:
        return normalize_quaternion([float(part) for part in parts])
    except (TypeError, ValueError):
        raise RuntimeError('%s must be x,y,z,w.' % option)


def _positive(value, option):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise RuntimeError('%s must be a positive finite number.' % option)
    if math.isnan(number) or math.isinf(number) or number <= 0.0:
        raise RuntimeError('%s must be a positive finite number.' % option)
    return number


def _nonnegative(value, option):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise RuntimeError('%s must be a non-negative finite number.' % option)
    if math.isnan(number) or math.isinf(number) or number < 0.0:
        raise RuntimeError('%s must be a non-negative finite number.' % option)
    return number


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description='RViz taught AprilTag pick and fixed bin placement helper.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Common modes:\n'
            '  teach_tag_grasp      record a tag pickup point\n'
            '  teach_place_start    record the shared place-teach start pose\n'
            '  teach_tag_place      record a fixed bin place point\n'
            '  teach_pre_pick_transit record the shared pre-pick transit pose\n'
            '  teach_carry          record the safe carry pose after pickup\n'
            '  teach_idle           record the idle/waiting pose\n'
            '  run_taught_sequence  pick and place taught tags\n\n'
            'Carry pose example:\n'
            '  python2 /home/eaibot/handeye-calib/src/mirobot_pick_test_tag.py \\\n'
            '    --mode teach_carry \\\n'
            '    --preset-file /home/eaibot/handeye-calib/config/tag_pick_place_presets.json \\\n'
            '    --overwrite'))
    parser.add_argument('--mode',
                        choices=['teach_tag_sequence', 'teach_tag_grasp',
                                 'teach_place_start', 'teach_tag_place',
                                 'teach_pre_pick_transit',
                                 'teach_carry',
                                 'teach_idle',
                                 'run_taught_sequence'],
                        required=True)
    parser.add_argument('--sequence', default=DEFAULT_SEQUENCE)
    parser.add_argument('--preset-file', default=DEFAULT_PRESET_FILE)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--camera-frame', default='camera_rgb_optical_frame')
    parser.add_argument('--base-frame', default='base')
    parser.add_argument('--group', default='manipulator')
    parser.add_argument('--tf-timeout', type=float, default=12.0)
    parser.add_argument('--tag-min-samples', type=int,
                        default=DEFAULT_TAG_MIN_SAMPLES)
    parser.add_argument('--tag-max-mad-m', type=float,
                        default=DEFAULT_TAG_MAX_MAD_M)
    parser.add_argument('--tag-max-age-seconds', type=float,
                        default=DEFAULT_TAG_MAX_AGE_SECONDS)
    parser.add_argument('--pickup-approach-axis-base',
                        default=DEFAULT_PICKUP_APPROACH_AXIS_BASE)
    parser.add_argument('--approach-gap', type=float,
                        default=DEFAULT_APPROACH_GAP,
                        help='Backoff behind the taught pre-grasp pose before the straight approach.')
    parser.add_argument('--place-approach-gap', type=float,
                        default=DEFAULT_PLACE_APPROACH_GAP)
    parser.add_argument('--planning-time', type=float, default=2.0)
    parser.add_argument('--disable-replanning', action='store_true')
    parser.add_argument('--velocity-scale', type=float,
                        default=DEFAULT_VELOCITY_SCALE)
    parser.add_argument('--acceleration-scale', type=float,
                        default=DEFAULT_ACCELERATION_SCALE)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--debug-hold-seconds', type=float, default=0.0)
    parser.add_argument('--assist-front-gap', type=float,
                        default=DEFAULT_ASSIST_FRONT_GAP)
    parser.add_argument('--assist-orientation-xyzw',
                        default=DEFAULT_ASSIST_ORIENTATION_XYZW)
    parser.add_argument('--disable-teach-assist', action='store_true')
    parser.add_argument('--teach-settle-seconds', type=float,
                        default=DEFAULT_TEACH_SETTLE_SECONDS)
    parser.add_argument('--motion-settle-seconds', type=float,
                        default=DEFAULT_MOTION_SETTLE_SECONDS)
    parser.add_argument('--home-after-idle', action='store_true',
                        help='After each successful tag, move to taught idle first and then run the controller startup homing service.')
    parser.add_argument('--startup-home-service',
                        default=DEFAULT_STARTUP_HOME_SERVICE)
    parser.add_argument('--startup-home-wait-seconds', type=float, default=8.0)
    parser.add_argument('--startup-home-settle-seconds', type=float, default=3.0)
    args = parser.parse_args(rospy.myargv(argv)[1:])
    args.sequence = parse_sequence(args.sequence)
    _positive(args.tf_timeout, '--tf-timeout')
    if args.tag_min_samples < 1:
        raise RuntimeError('--tag-min-samples must be at least 1.')
    _positive(args.tag_max_mad_m, '--tag-max-mad-m')
    _positive(args.tag_max_age_seconds, '--tag-max-age-seconds')
    _positive(args.approach_gap, '--approach-gap')
    _positive(args.place_approach_gap, '--place-approach-gap')
    _positive(args.planning_time, '--planning-time')
    _positive(args.velocity_scale, '--velocity-scale')
    _positive(args.acceleration_scale, '--acceleration-scale')
    _positive(args.assist_front_gap, '--assist-front-gap')
    _nonnegative(args.teach_settle_seconds, '--teach-settle-seconds')
    _nonnegative(args.motion_settle_seconds, '--motion-settle-seconds')
    _positive(args.startup_home_wait_seconds, '--startup-home-wait-seconds')
    _nonnegative(args.startup_home_settle_seconds,
                 '--startup-home-settle-seconds')
    args.assist_orientation_xyzw = parse_quaternion_text(
        args.assist_orientation_xyzw, '--assist-orientation-xyzw')
    axis_parts = [part.strip()
                  for part in args.pickup_approach_axis_base.split(',')]
    if len(axis_parts) != 3:
        raise RuntimeError('--pickup-approach-axis-base must be x,y,z.')
    try:
        args.pickup_approach_axis_base = normalize_axis(
            [float(part) for part in axis_parts])
    except (TypeError, ValueError):
        raise RuntimeError('--pickup-approach-axis-base must be x,y,z.')
    if args.debug_hold_seconds < 0.0:
        raise RuntimeError('--debug-hold-seconds must be non-negative.')
    return args


def ros_time_now():
    module_rospy = globals().get('rospy')
    if module_rospy is None or not hasattr(module_rospy, 'Time'):
        return None
    return module_rospy.Time.now()


def ros_is_shutdown():
    module_rospy = globals().get('rospy')
    return bool(module_rospy is not None and
                hasattr(module_rospy, 'is_shutdown') and
                module_rospy.is_shutdown())


def quaternion_msg_to_list(quaternion_msg):
    return [
        float(quaternion_msg.x),
        float(quaternion_msg.y),
        float(quaternion_msg.z),
        float(quaternion_msg.w),
    ]


def normalize_quaternion(values):
    norm = math.sqrt(sum(float(value) * float(value) for value in values))
    if norm <= 0.0 or math.isnan(norm) or math.isinf(norm):
        raise RuntimeError('Quaternion is invalid.')
    return [float(value) / norm for value in values]


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
    rotation = quaternion_to_matrix(quaternion_msg_to_list(pose_stamped.pose.orientation))
    position = pose_stamped.pose.position
    return [
        [rotation[0][0], rotation[0][1], rotation[0][2], float(position.x)],
        [rotation[1][0], rotation[1][1], rotation[1][2], float(position.y)],
        [rotation[2][0], rotation[2][1], rotation[2][2], float(position.z)],
        [0.0, 0.0, 0.0, 1.0],
    ]


def transform_to_matrix(transform):
    rotation = quaternion_to_matrix(transform['orientation_xyzw'])
    position = transform['position']
    return [
        [rotation[0][0], rotation[0][1], rotation[0][2], float(position[0])],
        [rotation[1][0], rotation[1][1], rotation[1][2], float(position[1])],
        [rotation[2][0], rotation[2][1], rotation[2][2], float(position[2])],
        [0.0, 0.0, 0.0, 1.0],
    ]


def matrix_multiply(first, second):
    result = [[0.0 for _ in range(4)] for _ in range(4)]
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


def matrix_to_transform(matrix):
    return {
        'position': [matrix[0][3], matrix[1][3], matrix[2][3]],
        'orientation_xyzw': quaternion_from_matrix(matrix),
    }


def transform_to_pose(frame_id, transform):
    matrix = transform_to_matrix(transform)
    return matrix_to_pose(frame_id, matrix)


def matrix_to_pose(frame_id, matrix):
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.header.stamp = ros_time_now()
    pose.pose.position.x = matrix[0][3]
    pose.pose.position.y = matrix[1][3]
    pose.pose.position.z = matrix[2][3]
    qx, qy, qz, qw = quaternion_from_matrix(matrix)
    pose.pose.orientation.x = qx
    pose.pose.orientation.y = qy
    pose.pose.orientation.z = qz
    pose.pose.orientation.w = qw
    return pose


def pose_to_transform(pose_stamped):
    pose = pose_stamped.pose
    return {
        'position': [
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
        ],
        'orientation_xyzw': quaternion_msg_to_list(pose.orientation),
    }


def compute_taught_pre_grasp_pose(tag_base_pose, pickup_model, entry,
                                  base_frame):
    offset = entry['grasp_offset_xyz_base']
    orientation = normalize_quaternion(
        pickup_model['orientation_xyzw_base'])
    pose = PoseStamped()
    pose.header.frame_id = base_frame
    pose.header.stamp = ros_time_now()
    pose.pose.position.x = (
        float(tag_base_pose.pose.position.x) + float(offset[0]))
    pose.pose.position.y = (
        float(tag_base_pose.pose.position.y) + float(offset[1]))
    pose.pose.position.z = (
        float(tag_base_pose.pose.position.z) + float(offset[2]))
    pose.pose.orientation.x = orientation[0]
    pose.pose.orientation.y = orientation[1]
    pose.pose.orientation.z = orientation[2]
    pose.pose.orientation.w = orientation[3]
    return pose


def build_backoff_pose(reference_pose, pickup_model,
                       distance_m, base_frame):
    axis = normalize_axis(pickup_model['approach_axis_xyz_base'])
    target = copy.deepcopy(reference_pose)
    target.header.frame_id = base_frame
    target.header.stamp = ros_time_now()
    target.pose.position.x += axis[0] * distance_m
    target.pose.position.y += axis[1] * distance_m
    target.pose.position.z += axis[2] * distance_m
    return target


def build_contact_probe_pose(probe_start_pose, pickup_model,
                             travel_m, base_frame):
    axis = normalize_axis(pickup_model['approach_axis_xyz_base'])
    target = copy.deepcopy(probe_start_pose)
    target.header.frame_id = base_frame
    target.header.stamp = ros_time_now()
    target.pose.position.x -= axis[0] * travel_m
    target.pose.position.y -= axis[1] * travel_m
    target.pose.position.z -= axis[2] * travel_m
    return target


def tag_plus_z_axis(tag_base_pose):
    matrix = pose_to_matrix(tag_base_pose)
    return [matrix[0][2], matrix[1][2], matrix[2][2]]


def horizontal_tag_outward_axis(tag_base_pose):
    normal = tag_plus_z_axis(tag_base_pose)
    return normalize_axis([normal[0], normal[1], 0.0])


def normalize_axis(axis):
    norm = math.sqrt(sum(float(value) * float(value) for value in axis))
    if norm <= 0.0 or math.isnan(norm) or math.isinf(norm):
        raise RuntimeError('Approach axis is invalid.')
    return [float(value) / norm for value in axis]


def build_pre_place_pose(place_pose, place_gap, base_frame):
    pre_place = copy.deepcopy(place_pose)
    pre_place.header.frame_id = base_frame
    pre_place.header.stamp = ros_time_now()
    pre_place.pose.position.z += place_gap
    return pre_place


def pose_position_distance(a, b):
    dx = a.pose.position.x - b.pose.position.x
    dy = a.pose.position.y - b.pose.position.y
    dz = a.pose.position.z - b.pose.position.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def pose_orientation_distance(a, b):
    qa = normalize_quaternion(quaternion_msg_to_list(a.pose.orientation))
    qb = normalize_quaternion(quaternion_msg_to_list(b.pose.orientation))
    dot = abs(qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3])
    dot = max(-1.0, min(1.0, dot))
    return 2.0 * math.acos(dot)


def pose_is_close(current_pose, target_pose):
    return (
        pose_position_distance(current_pose, target_pose) <=
        POSE_DONE_POSITION_TOLERANCE and
        pose_orientation_distance(current_pose, target_pose) <=
        POSE_DONE_ORIENTATION_TOLERANCE_RAD)


def current_pose_is_close_to_target(arm, target_pose, label):
    try:
        current_pose = arm.get_current_pose()
    except Exception as exc:
        rospy.logwarn('动作 %s 失败后读取当前末端位姿也失败：%s',
                      display_label(label), exc)
        return False
    if pose_is_close(current_pose, target_pose):
        rospy.logwarn(
            'MoveIt 报告 %s 失败，但当前末端已经接近目标点；接受这次动作，避免重复执行。',
            display_label(label))
        return True
    return False


def build_teach_assist_pose(tag_base_pose, front_gap, orientation_xyzw, base_frame):
    try:
        outward_axis = horizontal_tag_outward_axis(tag_base_pose)
    except RuntimeError:
        raise RuntimeError(
            'Tag +Z normal has no horizontal component for teach assist.')
    pose = PoseStamped()
    pose.header.frame_id = base_frame
    pose.header.stamp = ros_time_now()
    pose.pose.position.x = (
        tag_base_pose.pose.position.x + outward_axis[0] * front_gap)
    pose.pose.position.y = (
        tag_base_pose.pose.position.y + outward_axis[1] * front_gap)
    pose.pose.position.z = tag_base_pose.pose.position.z
    qx, qy, qz, qw = normalize_quaternion(orientation_xyzw)
    pose.pose.orientation.x = qx
    pose.pose.orientation.y = qy
    pose.pose.orientation.z = qz
    pose.pose.orientation.w = qw
    return pose


def median(values):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise RuntimeError('Cannot compute median from no values.')
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def append_unique_tag_sample(samples, seen_stamps, stamp_ns,
                             position, orientation_xyzw):
    stamp_ns = int(stamp_ns)
    if stamp_ns in seen_stamps:
        return False
    seen_stamps.add(stamp_ns)
    samples.append({
        'stamp_ns': stamp_ns,
        'position': [float(value) for value in position],
        'orientation_xyzw': normalize_quaternion(orientation_xyzw),
    })
    return True


def filter_tag_translation_samples(samples, min_samples=10,
                                   mad_scale=3.5,
                                   max_axis_mad_m=0.005):
    if len(samples) < int(min_samples):
        raise RuntimeError(
            'Only %d unique Tag samples were collected; need at least %d.'
            % (len(samples), min_samples))
    axes = list(zip(*[sample['position'] for sample in samples]))
    centers = [median(axis) for axis in axes]
    mads = [
        median([abs(float(value) - center) for value in axis])
        for axis, center in zip(axes, centers)
    ]
    max_axis_mad_m = float(max_axis_mad_m)
    if any(axis_mad > max_axis_mad_m for axis_mad in mads):
        raise RuntimeError(
            'Tag translation is unstable: axis MAD %s exceeds %.4f m.'
            % (mads, max_axis_mad_m))
    inliers = []
    for sample in samples:
        keep = True
        for value, center, axis_mad in zip(
                sample['position'], centers, mads):
            limit = max(float(axis_mad) * float(mad_scale), max_axis_mad_m)
            if abs(float(value) - center) > limit:
                keep = False
                break
        if keep:
            inliers.append(sample)
    if len(inliers) < int(min_samples):
        raise RuntimeError(
            'Only %d stable Tag samples remain after MAD filtering; need %d.'
            % (len(inliers), int(min_samples)))
    inlier_axes = list(zip(*[sample['position'] for sample in inliers]))
    newest = max(inliers, key=lambda sample: sample['stamp_ns'])
    return {
        'position': [median(axis) for axis in inlier_axes],
        'orientation_xyzw': newest['orientation_xyzw'],
        'sample_count': len(samples),
        'inlier_count': len(inliers),
        'axis_mad_m': mads,
    }


def read_preset_json(path):
    if not os.path.isfile(path):
        raise RuntimeError('Preset file does not exist: %s' % path)
    try:
        with open(path, 'r') as handle:
            preset = json.load(handle)
    except ValueError as exc:
        raise RuntimeError('无法解析 preset JSON：%s' % exc)
    except IOError as exc:
        raise RuntimeError('无法读取 preset 文件：%s' % exc)
    if not isinstance(preset.get('tags'), dict):
        raise RuntimeError('Preset file must contain a tags object.')
    return preset


def load_preset(path):
    preset = read_preset_json(path)
    if preset.get('version') != PRESET_VERSION:
        raise RuntimeError(
            'Preset version 3 is required for robust pickup; found version %r. '
            'Re-teach tag grasps to migrate while preserving place points.'
            % preset.get('version'))
    return preset


def migrate_legacy_preset_for_teach(preset):
    if preset.get('version') == PRESET_VERSION:
        return copy.deepcopy(preset)
    if preset.get('version') not in (1, 2):
        raise RuntimeError(
            'Cannot migrate unsupported preset version: %r.'
            % preset.get('version'))
    migrated = {
        'version': PRESET_VERSION,
        'base_frame': preset.get('base_frame', 'base'),
        'camera_frame': preset.get(
            'camera_frame', 'camera_rgb_optical_frame'),
        'tags': {},
    }
    if 'idle_joint_values' in preset:
        migrated['idle_joint_values'] = [
            float(value) for value in preset['idle_joint_values']
        ]
    if 'carry_joint_values' in preset:
        migrated['carry_joint_values'] = [
            float(value) for value in preset['carry_joint_values']
        ]
    if 'place_teach_start_ee_in_base' in preset:
        migrated['place_teach_start_ee_in_base'] = copy.deepcopy(
            preset['place_teach_start_ee_in_base'])
    for tag_id, entry in preset.get('tags', {}).items():
        migrated_entry = {}
        if 'place_ee_in_base' in entry:
            migrated_entry['place_ee_in_base'] = copy.deepcopy(
                entry['place_ee_in_base'])
        migrated['tags'][str(tag_id)] = migrated_entry
    return migrated


def load_preset_for_grasp_teach(path):
    return migrate_legacy_preset_for_teach(read_preset_json(path))


def make_empty_preset(base_frame, camera_frame):
    return {
        'version': PRESET_VERSION,
        'base_frame': base_frame,
        'camera_frame': camera_frame,
        'tags': {},
    }


def load_or_create_preset(args):
    if os.path.exists(args.preset_file):
        return load_preset_for_grasp_teach(args.preset_file), True
    return make_empty_preset(args.base_frame, args.camera_frame), False


def require_teach_overwrite_for_existing_tags(preset, sequence, overwrite):
    if overwrite:
        return
    existing = [
        str(tag_id) for tag_id in sequence
        if str(tag_id) in preset.get('tags', {})
    ]
    if existing:
        raise RuntimeError(
            'Preset already contains tag(s) %s. Use --overwrite to update them.'
            % ','.join(existing))


def save_preset(path, preset, overwrite=False):
    if os.path.exists(path) and not overwrite:
        raise RuntimeError('Preset file already exists: %s. Use --overwrite to replace it.' % path)
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    tmp_path = path + '.tmp'
    try:
        with open(tmp_path, 'w') as handle:
            json.dump(preset, handle, indent=2, sort_keys=True)
            handle.write('\n')
        if hasattr(os, 'replace'):
            os.replace(tmp_path, path)
        else:
            os.rename(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def require_preset_tags(preset, sequence):
    tags = preset.get('tags')
    if not isinstance(tags, dict):
        raise RuntimeError('Preset file must contain a tags object.')
    for tag_id in sequence:
        if str(tag_id) not in tags:
            raise RuntimeError('Preset file is missing tag %d.' % tag_id)


def require_pickup_model(preset):
    model = preset.get('pickup_model')
    if not isinstance(model, dict):
        raise RuntimeError(
            'Preset version 3 is missing pickup_model. Re-teach at least '
            'one tag grasp before running.')
    for field in (
            'orientation_xyzw_base',
            'approach_axis_xyz_base'):
        if field not in model:
            raise RuntimeError(
                'Preset pickup_model is missing %s. Re-teach tag grasps.'
                % field)
    normalize_quaternion(model['orientation_xyzw_base'])
    normalize_axis(model['approach_axis_xyz_base'])
    return model


def require_tag_fields(preset, tag_id, fields, mode):
    tags = preset.get('tags')
    if not isinstance(tags, dict) or str(tag_id) not in tags:
        raise RuntimeError(
            'Preset file is missing tag %d. Use teach_tag_sequence first.'
            % tag_id)
    entry = tags[str(tag_id)]
    for field in fields:
        if field not in entry:
            raise RuntimeError(
                'Preset tag %d is missing %s. Use teach_tag_sequence first.'
                % (tag_id, field))
    return entry


def require_shared_grasp_offset(preset):
    reference_tag = DEFAULT_SHARED_GRASP_REFERENCE_TAG
    entry = require_tag_fields(
        preset, reference_tag, ['grasp_offset_xyz_base'],
        'shared Tag grasp')
    offset = entry['grasp_offset_xyz_base']
    if not isinstance(offset, (list, tuple)) or len(offset) != 3:
        raise RuntimeError(
            'Tag %d grasp_offset_xyz_base must contain three values.'
            % reference_tag)
    values = [float(value) for value in offset]
    if any(math.isnan(value) or math.isinf(value) for value in values):
        raise RuntimeError(
            'Tag %d grasp_offset_xyz_base must contain finite values.'
            % reference_tag)
    return values


def require_joint_values(preset, field):
    values = preset.get(field)
    if not isinstance(values, (list, tuple)) or len(values) != 6:
        raise RuntimeError(
            'Preset is missing %s with six joint values. Run '
            '--mode teach_pre_pick_transit first.' % field)
    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError, OverflowError):
        raise RuntimeError('%s must contain finite joint values.' % field)
    if any(math.isnan(value) or math.isinf(value) for value in result):
        raise RuntimeError('%s must contain finite joint values.' % field)
    return result


def require_field_overwrite(preset, sequence, field, overwrite):
    if overwrite:
        return
    existing = []
    for tag_id in sequence:
        entry = preset.get('tags', {}).get(str(tag_id), {})
        if field in entry:
            existing.append(str(tag_id))
    if existing:
        raise RuntimeError(
            'Preset already contains %s for tag(s) %s. Use --overwrite to update it.'
            % (field, ','.join(existing)))


def record_tag_grasp_in_preset(preset, tag_id, tag_pose, taught_pre_grasp_pose,
                               approach_axis_base=None,
                               replace_pickup_model=False):
    entry = preset.setdefault('tags', {}).setdefault(str(tag_id), {})
    entry['grasp_offset_xyz_base'] = [
        float(taught_pre_grasp_pose.pose.position.x -
              tag_pose.pose.position.x),
        float(taught_pre_grasp_pose.pose.position.y -
              tag_pose.pose.position.y),
        float(taught_pre_grasp_pose.pose.position.z -
              tag_pose.pose.position.z),
    ]
    if 'pickup_model' not in preset or replace_pickup_model:
        preset['pickup_model'] = {
            'orientation_xyzw_base': normalize_quaternion(
                quaternion_msg_to_list(taught_pre_grasp_pose.pose.orientation)),
            'approach_axis_xyz_base': normalize_axis(
                approach_axis_base or [-1.0, 0.0, 0.0]),
        }
    for legacy_field in (
            'grasp_ee_in_tag',
            'grasp_position_offset_in_base',
            'grasp_orientation_in_base',
            'grasp_approach_axis_in_base',
            'grasp_joint_values',
            'grasp_offset_xy_base'):
        entry.pop(legacy_field, None)
    return entry['grasp_offset_xyz_base']


def record_tag_place_in_preset(preset, tag_id, place_pose):
    entry = preset.setdefault('tags', {}).setdefault(str(tag_id), {})
    entry['place_ee_in_base'] = pose_to_transform(place_pose)
    entry['place_orientation_in_base'] = normalize_quaternion(
        quaternion_msg_to_list(place_pose.pose.orientation))
    entry['place_approach_axis_in_base'] = [0.0, 0.0, 1.0]
    entry.pop('place_joint_values', None)
    return entry['place_ee_in_base']


def require_place_teach_start(preset):
    transform = preset.get('place_teach_start_ee_in_base')
    if not isinstance(transform, dict):
        raise RuntimeError(
            'Preset is missing place_teach_start_ee_in_base. Run '
            '--mode teach_place_start first.')
    transform_to_matrix(transform)
    return transform


def record_place_teach_start_in_preset(preset, pose):
    preset['place_teach_start_ee_in_base'] = pose_to_transform(pose)
    return preset['place_teach_start_ee_in_base']


def record_idle_in_preset(preset, arm):
    preset['idle_joint_values'] = [
        float(value) for value in arm.get_current_joint_values()
    ]
    return preset['idle_joint_values']


def record_carry_in_preset(preset, arm):
    preset['carry_joint_values'] = [
        float(value) for value in arm.get_current_joint_values()
    ]
    return preset['carry_joint_values']


def record_pre_pick_transit_in_preset(preset, arm):
    values = [float(value) for value in arm.get_current_joint_values()]
    if len(values) != 6 or any(
            math.isnan(value) or math.isinf(value) for value in values):
        raise RuntimeError(
            'Current pre-pick transit pose must contain six finite joints.')
    preset['pre_pick_transit_joint_values'] = values
    return values


def build_move_group(group_name, base_frame, velocity_scale, acceleration_scale,
                     planning_time, allow_replanning):
    if not rospy.has_param('robot_description'):
        raise RuntimeError(
            'robot_description is missing from the ROS parameter server. '
            'Make sure terminal 2 is running: '
            'roslaunch mirobot_moveit_config mirobot.launch start_rviz:=false')
    if not rospy.has_param('robot_description_semantic'):
        raise RuntimeError(
            'robot_description_semantic is missing from the ROS parameter server. '
            'MoveIt may not be running. Start terminal 2 with: '
            'roslaunch mirobot_moveit_config mirobot.launch start_rviz:=false')
    try:
        arm = moveit_commander.MoveGroupCommander(group_name)
    except RuntimeError as exc:
        raise RuntimeError(
            'Unable to construct MoveIt move group %r. '
            'Check that terminal 2 is still running and publishing '
            'robot_description / robot_description_semantic. Original error: %s'
            % (group_name, exc))
    arm.set_pose_reference_frame(base_frame)
    arm.allow_replanning(allow_replanning)
    arm.set_max_velocity_scaling_factor(velocity_scale)
    arm.set_max_acceleration_scaling_factor(acceleration_scale)
    arm.set_planning_time(planning_time)
    return arm


def get_mirobot_pump_type():
    try:
        from mirobot_urdf_2.srv import mirobotPump
        return mirobotPump
    except ImportError:
        raise RuntimeError('mirobot pump service type is unavailable. Source the ROS workspaces first.')


def get_pump_proxy():
    rospy.loginfo('等待吸泵服务：switch_pump_status')
    rospy.wait_for_service('switch_pump_status', timeout=5.0)
    return rospy.ServiceProxy('switch_pump_status', get_mirobot_pump_type())


def get_startup_home_type():
    try:
        from std_srvs.srv import Trigger
        return Trigger
    except ImportError:
        raise RuntimeError(
            'std_srvs/Trigger is unavailable. Source the ROS workspaces first.')


def get_contact_service_types():
    try:
        from std_srvs.srv import SetBool, Trigger
        return SetBool, Trigger
    except ImportError:
        raise RuntimeError(
            'std_srvs contact service types are unavailable. Source the ROS workspaces first.')


def get_contact_proxies():
    enable_type, state_type = get_contact_service_types()
    rospy.loginfo('等待限位探测服务：%s、%s',
                  CONTACT_PROBE_ENABLE_SERVICE, CONTACT_STATE_SERVICE)
    rospy.wait_for_service(CONTACT_PROBE_ENABLE_SERVICE, timeout=5.0)
    rospy.wait_for_service(CONTACT_STATE_SERVICE, timeout=5.0)
    return (
        rospy.ServiceProxy(CONTACT_PROBE_ENABLE_SERVICE, enable_type),
        rospy.ServiceProxy(CONTACT_STATE_SERVICE, state_type),
    )


def run_startup_home(args):
    service_name = getattr(
        args, 'startup_home_service', DEFAULT_STARTUP_HOME_SERVICE)
    wait_seconds = getattr(args, 'startup_home_wait_seconds', 8.0)
    settle_seconds = getattr(args, 'startup_home_settle_seconds', 3.0)
    rospy.loginfo('执行启动回零服务：%s', service_name)
    rospy.wait_for_service(service_name, timeout=wait_seconds)
    response = rospy.ServiceProxy(service_name, get_startup_home_type())()
    if not response.success:
        raise RuntimeError(
            '启动回零服务执行失败：%s' % response.message)
    rospy.sleep(settle_seconds)


def set_pump(pump_proxy, enabled):
    rospy.loginfo('吸泵%s', '开启' if enabled else '关闭')
    response = pump_proxy(enabled)
    if not response.Sucess:
        raise RuntimeError('吸泵服务返回失败。')


def set_contact_probe_enabled(enable_proxy, enabled):
    response = enable_proxy(bool(enabled))
    if not response.success:
        raise RuntimeError(
            '限位探测%s失败：%s'
            % ('启用' if enabled else '关闭', response.message))


def contact_is_triggered(state_proxy):
    response = state_proxy()
    message = str(response.message or '')
    if not response.success and message.startswith('ERROR:'):
        raise RuntimeError('读取限位开关失败：%s' % message)
    return bool(response.success)


def pose_from_tf_sample(frame_id, trans, rot):
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.header.stamp = rospy.Time.now()
    pose.pose.position.x = trans[0]
    pose.pose.position.y = trans[1]
    pose.pose.position.z = trans[2]
    orientation = normalize_quaternion(rot)
    pose.pose.orientation.x = orientation[0]
    pose.pose.orientation.y = orientation[1]
    pose.pose.orientation.z = orientation[2]
    pose.pose.orientation.w = orientation[3]
    return pose


def wait_for_tag_pose_in_base(listener, args, tag_id):
    tag_frame = 'tag_%d' % tag_id
    deadline = rospy.Time.now() + rospy.Duration(args.tf_timeout)
    min_samples = getattr(
        args, 'tag_min_samples', DEFAULT_TAG_MIN_SAMPLES)
    max_mad_m = getattr(
        args, 'tag_max_mad_m', DEFAULT_TAG_MAX_MAD_M)
    max_age_seconds = getattr(
        args, 'tag_max_age_seconds', DEFAULT_TAG_MAX_AGE_SECONDS)
    samples = []
    seen_stamps = set()
    latest_pose = None
    latest_age_seconds = None
    latest_filter_error = None
    last_reported_sample_count = 0
    last_filter_report_count = 0
    rospy.loginfo(
        '等待 tag_%d 稳定位姿：需要 %d 个新 TF，过滤后至少保留 %d 个内点。'
        '连续 %.1fs 收不到新的有效 TF 才超时。',
        tag_id, min_samples, min_samples, args.tf_timeout)
    while not rospy.is_shutdown() and rospy.Time.now() < deadline:
        try:
            common_time = listener.getLatestCommonTime(
                args.base_frame, tag_frame)
            age_seconds = (rospy.Time.now() - common_time).to_sec()
            trans, rot = listener.lookupTransform(
                args.base_frame, tag_frame, common_time)
            latest_pose = pose_from_tf_sample(args.base_frame, trans, rot)
            latest_age_seconds = age_seconds
            if age_seconds < 0.0 or age_seconds > max_age_seconds:
                rospy.sleep(0.02)
                continue
            stamp_ns = int(common_time.to_nsec())
            if append_unique_tag_sample(samples, seen_stamps, stamp_ns, trans, rot):
                # Treat tf_timeout as an inactivity timeout. A slow detector may
                # need longer than tf_timeout in total to produce enough unique
                # frames, but every fresh frame proves that the pipeline is alive.
                deadline = rospy.Time.now() + rospy.Duration(args.tf_timeout)
                if len(samples) != last_reported_sample_count:
                    last_reported_sample_count = len(samples)
                    rospy.loginfo(
                        'tag_%d 已采集新 TF：%d/%d，源帧 stamp=%.3f，age=%.2fs。',
                        tag_id, min(len(samples), min_samples),
                        min_samples, stamp_ns / 1000000000.0, age_seconds)
            if len(samples) >= min_samples:
                try:
                    filtered = filter_tag_translation_samples(
                        samples, min_samples=min_samples,
                        max_axis_mad_m=max_mad_m)
                except RuntimeError as exc:
                    latest_filter_error = exc
                    if len(samples) != last_filter_report_count:
                        last_filter_report_count = len(samples)
                        rospy.logwarn(
                            'tag_%d 当前采样过滤后内点不足，继续等待新 TF：%s',
                            tag_id, exc)
                    rospy.sleep(0.02)
                    continue
                pose = PoseStamped()
                pose.header.frame_id = args.base_frame
                pose.header.stamp = rospy.Time.now()
                pose.pose.position.x = filtered['position'][0]
                pose.pose.position.y = filtered['position'][1]
                pose.pose.position.z = filtered['position'][2]
                orientation = filtered['orientation_xyzw']
                pose.pose.orientation.x = orientation[0]
                pose.pose.orientation.y = orientation[1]
                pose.pose.orientation.z = orientation[2]
                pose.pose.orientation.w = orientation[3]
                rospy.loginfo(
                    'tag_%d 稳定位姿锁存：内点=%d/%d，MAD_mm=[%.2f, %.2f, %.2f]',
                    tag_id, filtered['inlier_count'],
                    filtered['sample_count'],
                    filtered['axis_mad_m'][0] * 1000.0,
                    filtered['axis_mad_m'][1] * 1000.0,
                    filtered['axis_mad_m'][2] * 1000.0)
                return pose
        except (tf.Exception, tf.LookupException, tf.ConnectivityException,
                tf.ExtrapolationException):
            pass
        rospy.sleep(0.02)
    if samples:
        if latest_filter_error is not None:
            raise RuntimeError(
                'tag_%d did not provide %d stable inlier TF samples before timeout. '
                'Collected %d unique fresh samples. Last filter error: %s'
                % (tag_id, min_samples, len(samples), latest_filter_error))
        raise RuntimeError(
            'tag_%d did not provide enough stable, fresh, unique TF samples: '
            'collected %d, need %d; no new valid TF arrived for %.1fs.'
            % (tag_id, len(samples), min_samples, args.tf_timeout))
    if latest_pose is not None:
        raise RuntimeError(
            'TF for %s was visible but too old for robust pickup '
            '(latest_age=%s, allowed<=%.1fs). Check AprilTag publish rate or '
            'increase --tag-max-age-seconds.'
            % (tag_frame,
               'unknown' if latest_age_seconds is None else '%.2fs' % latest_age_seconds,
               max_age_seconds))
    tag_exists = False
    camera_exists = False
    try:
        tag_exists = listener.frameExists(tag_frame)
        camera_exists = listener.frameExists(args.camera_frame)
    except Exception:
        pass
    if tag_exists and camera_exists:
        try:
            now = rospy.Time(0)
            listener.waitForTransform(args.base_frame, args.camera_frame, now,
                                      rospy.Duration(0.1))
        except (tf.Exception, tf.LookupException, tf.ConnectivityException,
                tf.ExtrapolationException):
            raise RuntimeError(
                'Detected %s under %s, but %s and %s are not connected in TF. '
                'AprilTag detection is running, but the hand-eye result is not being '
                'published. Start terminal 3: roslaunch easy_handeye publish.launch '
                'eye_on_hand:=false tracking_base_frame:=camera_link'
                % (tag_frame, args.camera_frame, args.base_frame, args.camera_frame))
    raise RuntimeError(
        'TF for %s was not found. Make sure AprilTag detection is running and the tag '
        'is visible in the terminal 5 window.' % tag_frame)


def prompt_enter(message):
    print('')
    print(message)
    print('按 Enter 继续；输入 q 再回车退出。')
    if hasattr(sys, 'stdout') and hasattr(sys.stdout, 'flush'):
        sys.stdout.flush()
    while True:
        if ros_is_shutdown():
            raise UserAbort('等待 Enter 时 ROS 已中断。')
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
        except (select.error, IOError) as exc:
            if exc.args and exc.args[0] == errno.EINTR:
                continue
            raise
        except KeyboardInterrupt:
            raise UserAbort('用户中断。')
        if not ready:
            continue
        try:
            line = sys.stdin.readline()
        except EOFError:
            raise RuntimeError('Input closed while waiting for Enter.')
        except KeyboardInterrupt:
            raise UserAbort('用户中断。')
        if line == '':
            raise RuntimeError('Input closed while waiting for Enter.')
        value = line.strip().lower()
        if value in ('q', 'quit', 'exit'):
            raise UserAbort('Aborted by user.')
        return


def pose_to_text(name, pose_stamped):
    pose = pose_stamped.pose
    return ('%s 位置=(%.4f, %.4f, %.4f) 姿态xyzw=(%.4f, %.4f, %.4f, %.4f)'
            % (name,
               pose.position.x, pose.position.y, pose.position.z,
               pose.orientation.x, pose.orientation.y,
               pose.orientation.z, pose.orientation.w))


DISPLAY_LABELS = {
    'tag_in_base': 'tag在base下位姿',
    'place_teach_start': '共用放置示教起点',
    'pre_pick_transit': '抓取前共享中转点',
    'approach_staging': '预抓点后方安全点',
    'taught_pre_grasp': '示教预抓点',
    'pickup_retreat': '吸附后退点',
    'carry': '搬运中间姿态',
    'taught_pre_place': '放置上方点',
    'taught_place': '放置接触点',
    'taught_place_retreat': '放置后退点',
    'idle': '空闲姿态',
}


def display_label(label):
    return DISPLAY_LABELS.get(label, label)


def create_debug_marker(marker_id, pose_stamped, rgb, scale):
    marker = Marker()
    marker.header.frame_id = pose_stamped.header.frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = 'mirobot_pick_debug'
    marker.id = marker_id
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose = copy.deepcopy(pose_stamped.pose)
    marker.scale.x = scale
    marker.scale.y = scale
    marker.scale.z = scale
    marker.color.r = rgb[0]
    marker.color.g = rgb[1]
    marker.color.b = rgb[2]
    marker.color.a = 0.85
    return marker


def publish_debug_geometry(base_frame, poses):
    publishers = {}
    for name in poses:
        publishers[name] = rospy.Publisher(
            'mirobot_pick_debug/%s' % name, PoseStamped, queue_size=1, latch=True)
    marker_pub = rospy.Publisher('mirobot_pick_debug/markers', MarkerArray,
                                 queue_size=1, latch=True)
    rospy.sleep(0.1)
    marker_colors = {
        'tag_in_base': (0.1, 0.4, 0.9),
        'teach_assist_front': (0.2, 0.95, 0.35),
        'approach_staging': (0.95, 0.75, 0.1),
        'taught_pre_grasp': (0.95, 0.75, 0.1),
        'taught_pre_place': (0.8, 0.3, 0.95),
        'taught_place': (0.25, 0.95, 0.95),
    }
    markers = MarkerArray()
    marker_id = 0
    for name, pose in poses.items():
        pose.header.frame_id = pose.header.frame_id or base_frame
        pose.header.stamp = rospy.Time.now()
        publishers[name].publish(pose)
        markers.markers.append(create_debug_marker(
            marker_id, pose, marker_colors.get(name, (0.8, 0.8, 0.8)), 0.02))
        marker_id += 1
    marker_pub.publish(markers)


def settle_after_motion():
    if MOTION_SETTLE_SECONDS > 0.0:
        rospy.sleep(MOTION_SETTLE_SECONDS)


def execute_pose(arm, target_pose, label):
    rospy.loginfo('执行动作：%s', pose_to_text(display_label(label), target_pose))
    for attempt in range(2):
        arm.set_start_state_to_current_state()
        arm.set_pose_target(target_pose)
        success = arm.go(wait=True)
        arm.stop()
        arm.clear_pose_targets()
        if success:
            settle_after_motion()
            return
        if current_pose_is_close_to_target(arm, target_pose, label):
            settle_after_motion()
            return
        if attempt == 0:
            rospy.logwarn(
                'MoveIt 执行 %s 失败，等待关节状态稳定后重试一次。',
                display_label(label))
            rospy.sleep(0.3)
    raise RuntimeError('MoveIt 执行 %s 失败。' % display_label(label))


def move_to_staging_with_fallback(primary_gap_m, build_pose, move_pose,
                                  sleep_fn, logwarn_fn, label,
                                  abort_exceptions=()):
    """安全点不可达时逐步靠近 P；主距离成功时不增加额外动作。"""
    gaps = [float(primary_gap_m)]
    for gap in STAGING_FALLBACK_GAPS_M:
        if gap < gaps[0] - 1e-9:
            gaps.append(gap)
    for index, gap in enumerate(gaps):
        pose = build_pose(gap)
        attempt_label = label if index == 0 else '%s_fallback_%dmm' % (
            label, int(round(gap * 1000.0)))
        try:
            move_pose(pose, attempt_label)
            return pose, gap
        except RuntimeError as exc:
            if abort_exceptions and isinstance(exc, abort_exceptions):
                raise
            if index + 1 >= len(gaps):
                raise
            logwarn_fn(
                '%s 后方 %.0fmm 规划失败：%s；0.1秒后改试 %.0fmm。' % (
                    label, gap * 1000.0, exc,
                    gaps[index + 1] * 1000.0))
            sleep_fn(STAGING_FALLBACK_INTERVAL_SECONDS)


def execute_joint_values(arm, joint_values, label):
    rospy.loginfo('执行关节动作：%s joint_values=%s',
                  display_label(label), joint_values)
    target_values = [float(value) for value in joint_values]
    for attempt in range(2):
        arm.set_start_state_to_current_state()
        arm.set_joint_value_target(target_values)
        success = arm.go(wait=True)
        arm.stop()
        arm.clear_pose_targets()
        if success:
            settle_after_motion()
            return
        if attempt == 0:
            rospy.logwarn(
                'MoveIt 执行 %s 失败，等待关节状态稳定后重试一次。',
                display_label(label))
            rospy.sleep(0.5)
    raise RuntimeError('MoveIt 执行 %s 失败。' % display_label(label))


def log_current_pose(arm, label):
    try:
        current_pose = arm.get_current_pose()
    except Exception as exc:
        rospy.logwarn('%s 后读取当前末端位姿失败：%s', label, exc)
        return None
    rospy.loginfo(pose_to_text(label, current_pose))
    return current_pose


def execute_cartesian_pose(arm, target_pose, label, eef_step=0.005,
                           jump_threshold=0.0, retry_without_collisions=False,
                           fallback_to_pose=False, quiet=False,
                           settle=True, stop_after=True,
                           min_point_interval=0.0):
    if not quiet:
        rospy.loginfo('执行直线动作：%s',
                      pose_to_text(display_label(label), target_pose))
    for attempt in range(2):
        arm.set_start_state_to_current_state()
        plan, fraction = arm.compute_cartesian_path(
            [copy.deepcopy(target_pose.pose)], eef_step, jump_threshold, True)
        if fraction < 0.999:
            if retry_without_collisions:
                rospy.logwarn(
                    'MoveIt 规划 %s 直线路径只完成 %.3f，尝试关闭碰撞检查后重算。',
                    display_label(label), fraction)
                arm.set_start_state_to_current_state()
                plan, fraction = arm.compute_cartesian_path(
                    [copy.deepcopy(target_pose.pose)], eef_step, jump_threshold,
                    False)
            if fraction < 0.999 and fallback_to_pose:
                rospy.logwarn(
                    'MoveIt 规划 %s 直线路径仍只完成 %.3f，改用普通位姿规划。',
                    display_label(label), fraction)
                execute_pose(arm, target_pose, label + '_pose_fallback')
                return
            if fraction < 0.999:
                raise RuntimeError(
                    'MoveIt 无法完整规划 %s 直线路径，完成比例=%.3f。'
                    % (display_label(label), fraction))
        if not plan.joint_trajectory.points:
            raise RuntimeError('MoveIt 为 %s 返回了空直线轨迹。' %
                               display_label(label))
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
            if settle:
                settle_after_motion()
            return
        if attempt == 0:
            rospy.logwarn(
                'MoveIt 执行 %s 直线动作失败，保持当前姿态，等待关节状态稳定后重试一次。',
                display_label(label))
            rospy.sleep(0.5)
    raise RuntimeError('MoveIt 执行 %s 直线动作失败。' % display_label(label))


def run_contact_approach(arm, taught_pre_grasp_pose, pickup_model, base_frame,
                         enable_proxy, state_proxy,
                         skip_staging_motion=False,
                         staging_step_m=CONTACT_STAGING_STEP_M,
                         probe_step_m=CONTACT_PROBE_STEP_M,
                         max_travel_m=CONTACT_PROBE_MAX_TRAVEL_M,
                         poll_seconds=CONTACT_PROBE_POLL_SECONDS,
                         point_interval=CONTACT_PROBE_EXPECTED_POINT_SECONDS):
    set_contact_probe_enabled(enable_proxy, True)
    try:
        if contact_is_triggered(state_proxy):
            return True
        rospy.loginfo(
            '限位已在预抓点后方安全点开启：先以 %.0fmm 步长受保护地'
            '直线移动到 P，未触发再从 P 前探 %.0fmm，步长 %.0fmm。',
            staging_step_m * 1000.0,
            max_travel_m * 1000.0,
            probe_step_m * 1000.0,
            )
        if not skip_staging_motion:
            execute_cartesian_pose(
                arm, taught_pre_grasp_pose, 'guarded_to_taught_pre_grasp',
                eef_step=staging_step_m,
                quiet=True, settle=False, stop_after=False,
                min_point_interval=point_interval)
            rospy.sleep(poll_seconds)
            if contact_is_triggered(state_proxy):
                rospy.loginfo('到达示教预抓点 P 之前已触发限位，停止继续前探。')
                return True
        probe_end_pose = build_contact_probe_pose(
            taught_pre_grasp_pose, pickup_model,
            max_travel_m, base_frame)
        execute_cartesian_pose(
            arm, probe_end_pose, 'contact_probe_path',
            eef_step=probe_step_m,
            quiet=True, settle=False, stop_after=False,
            min_point_interval=point_interval)
        rospy.sleep(poll_seconds)
        triggered = contact_is_triggered(state_proxy)
        if triggered:
            rospy.loginfo('限位开关已触发，底层已停止后续探测路点。')
        return triggered
    finally:
        set_contact_probe_enabled(enable_proxy, False)


def cache_tag_pose_for_teach(listener, args, tag_id, index, total_tags):
    prompt_enter(
        '步骤 1：缓存 tag_%d 位姿（当前第 %d/%d 个）\n'
        '请确认 AprilTag 窗口里 tag_%d 正在稳定画框。\n'
        '按 Enter 后脚本会读取当前 tag 位姿，然后自动尝试到 tag 前方安全点。'
        % (tag_id, index, total_tags, tag_id))
    tag_pose = wait_for_tag_pose_in_base(listener, args, tag_id)
    rospy.loginfo('步骤 1 完成：已缓存 tag_%d 位姿。', tag_id)
    rospy.loginfo(pose_to_text('tag_%d_in_base' % tag_id, tag_pose))
    return tag_pose


def move_to_teach_assist(args, arm, tag_id, tag_pose):
    if args.disable_teach_assist:
        rospy.logwarn('步骤 2 已跳过：--disable-teach-assist 已开启。')
        return
    try:
        assist_pose = build_teach_assist_pose(
            tag_pose, args.assist_front_gap,
            args.assist_orientation_xyzw, args.base_frame)
        rospy.loginfo(
            '步骤 2：自动移动到 tag_%d 前方安全点。距离 tag 约 %.3fm，Z 高度保持和 tag 一样，泵头使用水平姿态。',
            tag_id, args.assist_front_gap)
        rospy.loginfo(pose_to_text('tag_%d_teach_assist_front' % tag_id,
                                   assist_pose))
        publish_debug_geometry(args.base_frame, {
            'tag_in_base': tag_pose,
            'teach_assist_front': assist_pose,
        })
        if args.dry_run:
            rospy.logwarn(
                '当前是 dry-run，已跳过 tag_%d 的示教辅助移动。',
                tag_id)
        else:
            execute_pose(arm, assist_pose, 'tag_%d_teach_assist_front' % tag_id)
            log_current_pose(arm, '步骤 2 完成：MoveIt 当前末端位姿')
    except RuntimeError as exc:
        rospy.logwarn(
            '步骤 2 失败：tag_%d 自动到前方安全点失败：%s。请在 RViz 里手动移动到 tag 前方。',
            tag_id, exc)


def prompt_and_record_grasp(args, arm, preset, tag_id, tag_pose,
                            update_pickup_model=False):
    approach_gap = getattr(args, 'approach_gap', DEFAULT_APPROACH_GAP)
    prompt_enter(
        '步骤 3：记录 tag_%d 的预抓姿态\n'
        '请在 RViz 里微调 XY/Z 和吸盘朝向，使吸盘近距离正对物块，但不要贴住。\n'
        'Plan/Execute 到位后回这里按 Enter 保存。正式抓取会先停在该预抓点后方 %.0fmm，'
        '再直线伸到预抓点并启动限位慢速前探。'
        % (tag_id, approach_gap * 1000.0))
    settle_seconds = getattr(args, 'teach_settle_seconds', 0.0)
    if settle_seconds > 0.0:
        rospy.loginfo('等待 %.2fs，让关节状态刷新稳定后再记录抓取点。', settle_seconds)
        rospy.sleep(settle_seconds)
    taught_pre_grasp_pose = arm.get_current_pose()
    if update_pickup_model or 'place_teach_start_ee_in_base' not in preset:
        record_place_teach_start_in_preset(preset, taught_pre_grasp_pose)
    if update_pickup_model:
        approach_axis = horizontal_tag_outward_axis(tag_pose)
    elif isinstance(preset.get('pickup_model'), dict):
        approach_axis = normalize_axis(
            preset['pickup_model']['approach_axis_xyz_base'])
    else:
        approach_axis = args.pickup_approach_axis_base
    record_tag_grasp_in_preset(
        preset, tag_id, tag_pose, taught_pre_grasp_pose,
        approach_axis_base=approach_axis,
        replace_pickup_model=update_pickup_model)
    if update_pickup_model:
        rospy.loginfo(
            '已更新所有 tag 共用的吸盘姿态和限位前进轴：[%.4f, %.4f, %.4f]。',
            approach_axis[0], approach_axis[1], approach_axis[2])
    rospy.loginfo('步骤 3 完成：已记录 tag_%d 预抓姿态。', tag_id)
    if update_pickup_model:
        rospy.loginfo(
            '该预抓 Link6 位姿已同时更新为有 Tag/无 Tag 共用的放置示教起点。')
    rospy.loginfo(pose_to_text(
        'tag_%d_taught_pre_grasp_in_base' % tag_id,
        taught_pre_grasp_pose))


def move_to_place_teach_start(args, arm, preset):
    start_pose = transform_to_pose(
        args.base_frame, require_place_teach_start(preset))
    rospy.loginfo(
        '放置示教前先移动到有 Tag/无 Tag 共用的预抓姿态。')
    rospy.loginfo(pose_to_text('place_teach_start', start_pose))
    execute_pose(arm, start_pose, 'place_teach_start')


def teach_place_start(args, arm):
    preset = load_preset(args.preset_file)
    if ('place_teach_start_ee_in_base' in preset and not args.overwrite):
        raise RuntimeError(
            'Preset already contains place_teach_start_ee_in_base. '
            'Use --overwrite to update it.')
    prompt_enter(
        '示教有 Tag/无 Tag 共用的放置示教起点\n'
        '请在 RViz 中 Plan/Execute 到稳定的向前伸展预抓姿态。\n'
        '这里只记录当前 Link6 完整位姿，不会强制修改后续放置姿态。')
    settle_seconds = getattr(args, 'teach_settle_seconds', 0.0)
    if settle_seconds > 0.0:
        rospy.sleep(settle_seconds)
    start_pose = arm.get_current_pose()
    record_place_teach_start_in_preset(preset, start_pose)
    save_preset(args.preset_file, preset, overwrite=True)
    rospy.loginfo('共用放置示教起点已保存：%s', args.preset_file)
    rospy.loginfo(pose_to_text('place_teach_start', start_pose))


def prompt_and_record_place(args, arm, preset, tag_id):
    move_to_place_teach_start(args, arm, preset)
    prompt_enter(
        '步骤 4：记录 tag_%d 的载物仓释放姿态\n'
        '机械臂已到共用的预抓姿态。\n'
        '请从这里在 RViz 里移动吸盘到对应载物仓释放位置，Plan/Execute 到位后回这里按 Enter。\n'
        '最终保存你实际调整后的 Link6 完整位姿，不锁定姿态四元数。'
        % tag_id)
    settle_seconds = getattr(args, 'teach_settle_seconds', 0.0)
    if settle_seconds > 0.0:
        rospy.loginfo('等待 %.2fs，让关节状态刷新稳定后再记录放置点。', settle_seconds)
        rospy.sleep(settle_seconds)
    place_pose = arm.get_current_pose()
    record_tag_place_in_preset(preset, tag_id, place_pose)
    rospy.loginfo('步骤 4 完成：已记录 tag_%d 载物仓释放姿态。', tag_id)
    rospy.loginfo(pose_to_text('tag_%d_place_ee_in_base' % tag_id, place_pose))


def teach_tag_sequence(args, arm):
    listener = tf.TransformListener()
    rospy.sleep(0.5)
    preset, preset_existed = load_or_create_preset(args)
    require_teach_overwrite_for_existing_tags(
        preset, args.sequence, args.overwrite)
    total_tags = len(args.sequence)
    for index, tag_id in enumerate(args.sequence, 1):
        tag_pose = cache_tag_pose_for_teach(
            listener, args, tag_id, index, total_tags)
        move_to_teach_assist(args, arm, tag_id, tag_pose)
        prompt_and_record_grasp(
            args, arm, preset, tag_id, tag_pose,
            update_pickup_model=(index == 1))
        prompt_and_record_place(args, arm, preset, tag_id)
    save_preset(args.preset_file, preset,
                overwrite=(preset_existed or args.overwrite))
    rospy.loginfo('已保存 tag 示教参数：%s', args.preset_file)


def teach_tag_grasp(args, arm):
    if args.sequence != [DEFAULT_SHARED_GRASP_REFERENCE_TAG]:
        raise RuntimeError(
            'teach_tag_grasp uses tag_%d as the single shared reference. '
            'Run with --sequence %d.'
            % (DEFAULT_SHARED_GRASP_REFERENCE_TAG,
               DEFAULT_SHARED_GRASP_REFERENCE_TAG))
    preset = load_preset_for_grasp_teach(args.preset_file)
    require_field_overwrite(preset, args.sequence, 'grasp_offset_xyz_base',
                            args.overwrite)
    for tag_id in args.sequence:
        require_tag_fields(preset, tag_id, ['place_ee_in_base'],
                           'teach_tag_grasp')
    listener = tf.TransformListener()
    rospy.sleep(0.5)
    total_tags = len(args.sequence)
    for index, tag_id in enumerate(args.sequence, 1):
        tag_pose = cache_tag_pose_for_teach(
            listener, args, tag_id, index, total_tags)
        move_to_teach_assist(args, arm, tag_id, tag_pose)
        prompt_and_record_grasp(
            args, arm, preset, tag_id, tag_pose,
            update_pickup_model=(index == 1))
        rospy.loginfo('tag_%d 原来的载物仓释放姿态已保留，不会覆盖。', tag_id)
    save_preset(args.preset_file, preset, overwrite=True)
    rospy.loginfo('已保存 tag 示教参数：%s', args.preset_file)


def teach_tag_place(args, arm):
    preset = load_preset(args.preset_file)
    require_place_teach_start(preset)
    require_field_overwrite(preset, args.sequence, 'place_ee_in_base',
                            args.overwrite)
    require_shared_grasp_offset(preset)
    require_pickup_model(preset)
    total_tags = len(args.sequence)
    for index, tag_id in enumerate(args.sequence, 1):
        rospy.loginfo('准备重采 tag_%d 放置点（当前第 %d/%d 个），抓取姿态会保留。',
                      tag_id, index, total_tags)
        prompt_and_record_place(args, arm, preset, tag_id)
    save_preset(args.preset_file, preset, overwrite=True)
    rospy.loginfo('已保存 tag 示教参数：%s', args.preset_file)


def teach_idle(args, arm):
    preset, preset_existed = load_or_create_preset(args)
    if 'idle_joint_values' in preset and not args.overwrite:
        raise RuntimeError(
            'Preset already contains idle_joint_values. Use --overwrite to update it.')
    prompt_enter(
        '记录机械臂空闲姿态\n'
        '请在 RViz 里把机械臂移动到比赛等待/空闲姿态，Plan/Execute 到位后回这里按 Enter。')
    idle_joint_values = record_idle_in_preset(preset, arm)
    save_preset(args.preset_file, preset,
                overwrite=(preset_existed or args.overwrite))
    rospy.loginfo('已保存空闲姿态关节值：%s', idle_joint_values)
    rospy.loginfo('已保存 tag 示教参数：%s', args.preset_file)


def teach_carry(args, arm):
    preset, preset_existed = load_or_create_preset(args)
    if 'carry_joint_values' in preset and not args.overwrite:
        raise RuntimeError(
            'Preset already contains carry_joint_values. Use --overwrite to update it.')
    prompt_enter(
        '记录抓取后搬运中间姿态\n'
        '请在 RViz 里把机械臂移动到抓起物块后、去放置点前的安全中转姿态。\n'
        '建议姿态：物块离开料仓，关节不过极限，吸泵管线不缠绕。Plan/Execute 到位后回这里按 Enter。')
    carry_joint_values = record_carry_in_preset(preset, arm)
    save_preset(args.preset_file, preset,
                overwrite=(preset_existed or args.overwrite))
    rospy.loginfo('已保存搬运中间姿态关节值：%s', carry_joint_values)
    rospy.loginfo('已保存 tag 示教参数：%s', args.preset_file)


def teach_pre_pick_transit(args, arm):
    preset = load_preset(args.preset_file)
    field = 'pre_pick_transit_joint_values'
    if field in preset and not args.overwrite:
        raise RuntimeError(
            'Preset already contains %s. Use --overwrite to update it.'
            % field)
    prompt_enter(
        '记录 B 点有 Tag 抓取前共享中转点\n'
        '请在 RViz 中把机械臂移动到安全、不会碰撞的抓取前中间姿态。\n'
        '四个 Tag 每次都按“该中转点 -> 当前 Tag 抓取安全点 -> 抓取”执行；'
        '它与抓取完成后到达的 idle 是两个不同的点。'
        'Plan/Execute 到位后回这里按 Enter。')
    joint_values = record_pre_pick_transit_in_preset(preset, arm)
    save_preset(args.preset_file, preset, overwrite=True)
    rospy.loginfo('已保存 B 点抓取前共享中转点：%s', joint_values)
    rospy.loginfo('已保存 tag 示教参数：%s', args.preset_file)


def run_taught_sequence(args, arm, pump_proxy, contact_proxies=None):
    preset = load_preset(args.preset_file)
    require_preset_tags(preset, args.sequence)
    pickup_model = require_pickup_model(preset)
    shared_grasp_offset = require_shared_grasp_offset(preset)
    pre_pick_transit = require_joint_values(
        preset, 'pre_pick_transit_joint_values')
    rospy.loginfo(
        '四个 Tag 共用 ID%d 的示教抓取偏移。',
        DEFAULT_SHARED_GRASP_REFERENCE_TAG)
    listener = tf.TransformListener()
    rospy.sleep(0.5)
    if not args.dry_run and contact_proxies is None:
        contact_proxies = get_contact_proxies()
    enable_contact_proxy = None if args.dry_run else contact_proxies[0]
    contact_state_proxy = None if args.dry_run else contact_proxies[1]
    total_tags = len(args.sequence)
    incomplete_tags = []
    for index, tag_id in enumerate(args.sequence, 1):
        rospy.loginfo('开始处理 tag_%d（%d/%d）。', tag_id, index, total_tags)
        entry = preset['tags'][str(tag_id)]
        require_tag_fields(
            preset, tag_id,
            ['place_ee_in_base'],
            'run_taught_sequence')
        tag_pose = wait_for_tag_pose_in_base(listener, args, tag_id)
        taught_pre_grasp_pose = compute_taught_pre_grasp_pose(
            tag_pose, pickup_model,
            {'grasp_offset_xyz_base': shared_grasp_offset},
            args.base_frame)
        approach_staging_pose = build_backoff_pose(
            taught_pre_grasp_pose, pickup_model,
            args.approach_gap, args.base_frame)
        place_pose = transform_to_pose(args.base_frame, entry['place_ee_in_base'])
        pre_place_pose = build_pre_place_pose(
            place_pose, args.place_approach_gap, args.base_frame)
        probe_end_pose = build_contact_probe_pose(
            taught_pre_grasp_pose, pickup_model,
            CONTACT_PROBE_MAX_TRAVEL_M, args.base_frame)
        retreat_pose = build_backoff_pose(
            taught_pre_grasp_pose, pickup_model,
            CONTACT_RETREAT_EXTRA_M, args.base_frame)

        rospy.loginfo(pose_to_text('tag_%d在base下位姿' % tag_id, tag_pose))
        rospy.loginfo(pose_to_text('预抓点后方安全点', approach_staging_pose))
        rospy.loginfo(pose_to_text('示教预抓点', taught_pre_grasp_pose))
        rospy.loginfo(pose_to_text('限位探测最远点', probe_end_pose))
        rospy.loginfo(pose_to_text('抓取后退点', retreat_pose))
        rospy.loginfo(pose_to_text('放置上方点', pre_place_pose))
        rospy.loginfo(pose_to_text('放置接触点', place_pose))
        publish_debug_geometry(args.base_frame, {
            'tag_in_base': tag_pose,
            'approach_staging': approach_staging_pose,
            'taught_pre_grasp': taught_pre_grasp_pose,
            'contact_probe_end': probe_end_pose,
            'pickup_retreat': retreat_pose,
            'taught_pre_place': pre_place_pose,
            'taught_place': place_pose,
        })

        if args.dry_run:
            rospy.logwarn('tag_%d 当前是 dry-run，只打印/发布调试位姿，不执行机械臂动作。', tag_id)
            continue

        holding_object = False
        try:
            set_pump(pump_proxy, False)
            execute_joint_values(
                arm, pre_pick_transit, 'pre_pick_transit')
            approach_staging_pose, selected_staging_gap = \
                move_to_staging_with_fallback(
                    args.approach_gap,
                    lambda gap: build_backoff_pose(
                        taught_pre_grasp_pose, pickup_model,
                        gap, args.base_frame),
                    lambda pose, label: execute_pose(arm, pose, label),
                    rospy.sleep, rospy.logwarn, 'approach_staging')
            if not run_contact_approach(
                    arm, taught_pre_grasp_pose, pickup_model, args.base_frame,
                    enable_contact_proxy, contact_state_proxy,
                    selected_staging_gap <= 1e-9):
                rospy.logwarn(
                    'CONTACT_PROBE_MISS tag_%d：前进 %.0fmm 仍未触发限位，退回并跳过当前物块。',
                    tag_id, CONTACT_PROBE_MAX_TRAVEL_M * 1000.0)
                set_pump(pump_proxy, False)
                execute_cartesian_pose(
                    arm, retreat_pose, 'contact_probe_miss_retreat')
                if preset.get('idle_joint_values'):
                    execute_joint_values(
                        arm, preset['idle_joint_values'], 'idle')
                    rospy.sleep(0.5)
                incomplete_tags.append(tag_id)
                continue
            set_pump(pump_proxy, True)
            holding_object = True
            rospy.sleep(0.8)
            rospy.loginfo(
                '吸附完成，沿原接近路径直线退过预抓点 %.0fmm，再进入搬运规划。',
                CONTACT_RETREAT_EXTRA_M * 1000.0)
            execute_cartesian_pose(arm, retreat_pose, 'pickup_retreat')
            if preset.get('carry_joint_values'):
                execute_joint_values(arm, preset['carry_joint_values'], 'carry')
                rospy.sleep(0.5)
            execute_pose(arm, pre_place_pose, 'taught_pre_place')
            execute_cartesian_pose(
                arm, place_pose, 'taught_place',
                retry_without_collisions=True, fallback_to_pose=True)
            set_pump(pump_proxy, False)
            holding_object = False
            rospy.sleep(0.5)
            execute_cartesian_pose(
                arm, pre_place_pose, 'taught_place_retreat',
                retry_without_collisions=True, fallback_to_pose=True)
            if preset.get('idle_joint_values'):
                execute_joint_values(arm, preset['idle_joint_values'], 'idle')
                rospy.sleep(0.5)
            if args.home_after_idle:
                run_startup_home(args)
        except Exception:
            if holding_object:
                try:
                    rospy.logwarn(
                        '吸泵已开启后动作失败，退出前先关闭吸泵。')
                    set_pump(pump_proxy, False)
                except Exception as pump_exc:
                    rospy.logerr('动作失败后关闭吸泵也失败：%s',
                                 pump_exc)
            raise

    if args.dry_run and args.debug_hold_seconds > 0.0:
        rospy.loginfo('保持调试位姿话题 %.1f 秒。',
                      args.debug_hold_seconds)
        rospy.sleep(args.debug_hold_seconds)
    if incomplete_tags:
        raise ContactProbeIncomplete(incomplete_tags)


def main():
    args = parse_args(sys.argv)
    global MOTION_SETTLE_SECONDS
    MOTION_SETTLE_SECONDS = args.motion_settle_seconds
    rospy.init_node('mirobot_pick_test_tag', anonymous=False)
    moveit_commander.roscpp_initialize(sys.argv)
    arm = None
    exit_code = 0
    try:
        arm = build_move_group(
            args.group, args.base_frame, args.velocity_scale,
            args.acceleration_scale, args.planning_time,
            not args.disable_replanning)
        if args.mode == 'teach_tag_sequence':
            teach_tag_sequence(args, arm)
        elif args.mode == 'teach_tag_grasp':
            teach_tag_grasp(args, arm)
        elif args.mode == 'teach_place_start':
            teach_place_start(args, arm)
        elif args.mode == 'teach_tag_place':
            teach_tag_place(args, arm)
        elif args.mode == 'teach_pre_pick_transit':
            teach_pre_pick_transit(args, arm)
        elif args.mode == 'teach_carry':
            teach_carry(args, arm)
        elif args.mode == 'teach_idle':
            teach_idle(args, arm)
        else:
            pump_proxy = None if args.dry_run else get_pump_proxy()
            contact_proxies = None if args.dry_run else get_contact_proxies()
            run_taught_sequence(
                args, arm, pump_proxy, contact_proxies=contact_proxies)
        rospy.loginfo('Tag 示教/抓取流程结束。')
    except ContactProbeIncomplete as exc:
        rospy.logwarn(str(exc))
        exit_code = CONTACT_PROBE_MISS_EXIT_CODE
    except UserAbort as exc:
        rospy.logwarn(str(exc))
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        rospy.logwarn('用户中断。')
    except Exception as exc:
        rospy.logerr(str(exc))
        raise
    finally:
        moveit_commander.roscpp_shutdown()
    if exit_code:
        sys.exit(exit_code)


if __name__ == '__main__':
    main()
