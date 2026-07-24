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


PRESET_VERSION = 2
DEFAULT_SEQUENCE = '1,2,3,4'
DEFAULT_PRESET_FILE = '/home/eaibot/handeye-calib/config/tag_pick_place_presets.json'
DEFAULT_ASSIST_FRONT_GAP = 0.03
DEFAULT_ASSIST_ORIENTATION_XYZW = '0,0,0,1'
DEFAULT_TEACH_SETTLE_SECONDS = 0.8
DEFAULT_MOTION_SETTLE_SECONDS = 0.25
DEFAULT_TAG_SAMPLE_SECONDS = 1.5
DEFAULT_TAG_MIN_SAMPLES = 10
DEFAULT_TAG_MAX_MAD_M = 0.005
DEFAULT_TAG_MAX_AGE_SECONDS = 0.5
DEFAULT_PICKUP_APPROACH_AXIS_BASE = '-1,0,0'
POSE_DONE_POSITION_TOLERANCE = 0.015
POSE_DONE_ORIENTATION_TOLERANCE_RAD = 0.35
DEFAULT_STARTUP_HOME_SERVICE = '/mirobot_startup_home'
MOTION_SETTLE_SECONDS = DEFAULT_MOTION_SETTLE_SECONDS

try:
    STRING_TYPES = (basestring,)
except NameError:
    STRING_TYPES = (str,)


class UserAbort(Exception):
    pass


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
        description='RViz taught AprilTag pick and fixed bin placement helper.')
    parser.add_argument('--mode',
                        choices=['teach_tag_sequence', 'teach_tag_grasp',
                                 'teach_tag_place', 'teach_idle',
                                 'run_taught_sequence'],
                        required=True)
    parser.add_argument('--sequence', default=DEFAULT_SEQUENCE)
    parser.add_argument('--preset-file', default=DEFAULT_PRESET_FILE)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--camera-frame', default='camera_rgb_optical_frame')
    parser.add_argument('--base-frame', default='base')
    parser.add_argument('--group', default='manipulator')
    parser.add_argument('--tf-timeout', type=float, default=5.0)
    parser.add_argument('--tag-sample-seconds', type=float,
                        default=DEFAULT_TAG_SAMPLE_SECONDS)
    parser.add_argument('--tag-min-samples', type=int,
                        default=DEFAULT_TAG_MIN_SAMPLES)
    parser.add_argument('--tag-max-mad-m', type=float,
                        default=DEFAULT_TAG_MAX_MAD_M)
    parser.add_argument('--tag-max-age-seconds', type=float,
                        default=DEFAULT_TAG_MAX_AGE_SECONDS)
    parser.add_argument('--pickup-approach-axis-base',
                        default=DEFAULT_PICKUP_APPROACH_AXIS_BASE)
    parser.add_argument('--approach-gap', type=float, default=0.03)
    parser.add_argument('--place-approach-gap', type=float, default=0.02)
    parser.add_argument('--planning-time', type=float, default=2.0)
    parser.add_argument('--disable-replanning', action='store_true')
    parser.add_argument('--velocity-scale', type=float, default=0.1)
    parser.add_argument('--acceleration-scale', type=float, default=0.1)
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
    _positive(args.tag_sample_seconds, '--tag-sample-seconds')
    if args.tag_min_samples < 3:
        raise RuntimeError('--tag-min-samples must be at least 3.')
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


def compute_v2_grasp_pose(tag_base_pose, pickup_model, entry, base_frame):
    offset = entry['grasp_offset_xy_base']
    orientation = normalize_quaternion(
        pickup_model['orientation_xyzw_base'])
    pose = PoseStamped()
    pose.header.frame_id = base_frame
    pose.header.stamp = ros_time_now()
    pose.pose.position.x = (
        float(tag_base_pose.pose.position.x) + float(offset[0]))
    pose.pose.position.y = (
        float(tag_base_pose.pose.position.y) + float(offset[1]))
    pose.pose.position.z = float(pickup_model['contact_z_base'])
    pose.pose.orientation.x = orientation[0]
    pose.pose.orientation.y = orientation[1]
    pose.pose.orientation.z = orientation[2]
    pose.pose.orientation.w = orientation[3]
    return pose


def build_v2_pre_grasp_pose(grasp_pose, pickup_model,
                            approach_gap, base_frame):
    axis = normalize_axis(pickup_model['approach_axis_xyz_base'])
    pre_grasp = copy.deepcopy(grasp_pose)
    pre_grasp.header.frame_id = base_frame
    pre_grasp.header.stamp = ros_time_now()
    pre_grasp.pose.position.x += axis[0] * approach_gap
    pre_grasp.pose.position.y += axis[1] * approach_gap
    pre_grasp.pose.position.z += axis[2] * approach_gap
    return pre_grasp


def tag_plus_z_axis(tag_base_pose):
    matrix = pose_to_matrix(tag_base_pose)
    return [matrix[0][2], matrix[1][2], matrix[2][2]]


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
        rospy.logwarn('Could not read current pose after failed %s: %s',
                      label, exc)
        return False
    if pose_is_close(current_pose, target_pose):
        rospy.logwarn(
            'MoveIt reported failure during %s, but current pose is already close to target. '
            'Accepting this motion to avoid duplicate execution.',
            label)
        return True
    return False


def build_teach_assist_pose(tag_base_pose, front_gap, orientation_xyzw, base_frame):
    normal = tag_plus_z_axis(tag_base_pose)
    horizontal_norm = math.sqrt(normal[0] * normal[0] + normal[1] * normal[1])
    if horizontal_norm < 1e-6:
        raise RuntimeError('Tag +Z normal has no horizontal component for teach assist.')
    unit_x = normal[0] / horizontal_norm
    unit_y = normal[1] / horizontal_norm
    pose = PoseStamped()
    pose.header.frame_id = base_frame
    pose.header.stamp = ros_time_now()
    pose.pose.position.x = tag_base_pose.pose.position.x + unit_x * front_gap
    pose.pose.position.y = tag_base_pose.pose.position.y + unit_y * front_gap
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
    if max(mads) > float(max_axis_mad_m):
        raise RuntimeError(
            'Tag translation is unstable: axis MAD=%s m exceeds %.4f m.'
            % ([round(value, 6) for value in mads], max_axis_mad_m))
    inliers = []
    for sample in samples:
        keep = True
        for value, center, axis_mad in zip(
                sample['position'], centers, mads):
            limit = max(float(axis_mad) * float(mad_scale), 1e-6)
            if abs(float(value) - center) > limit:
                keep = False
                break
        if keep:
            inliers.append(sample)
    minimum_inliers = max(3, int(math.ceil(float(min_samples) * 0.7)))
    if len(inliers) < minimum_inliers:
        raise RuntimeError(
            'Only %d stable Tag samples remain after filtering; need %d.'
            % (len(inliers), minimum_inliers))
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
        raise RuntimeError('Could not parse preset JSON: %s' % exc)
    except IOError as exc:
        raise RuntimeError('Could not read preset file: %s' % exc)
    if not isinstance(preset.get('tags'), dict):
        raise RuntimeError('Preset file must contain a tags object.')
    return preset


def load_preset(path):
    preset = read_preset_json(path)
    if preset.get('version') != PRESET_VERSION:
        raise RuntimeError(
            'Preset version 2 is required for robust pickup; found version %r. '
            'Re-teach tag grasps to migrate while preserving place points.'
            % preset.get('version'))
    return preset


def migrate_legacy_preset_for_teach(preset):
    if preset.get('version') == PRESET_VERSION:
        return copy.deepcopy(preset)
    if preset.get('version') != 1:
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


def require_v2_pickup_model(preset):
    model = preset.get('pickup_model')
    if not isinstance(model, dict):
        raise RuntimeError(
            'Preset version 2 is missing pickup_model. Re-teach at least '
            'one tag grasp before running.')
    for field in (
            'orientation_xyzw_base',
            'approach_axis_xyz_base',
            'contact_z_base'):
        if field not in model:
            raise RuntimeError(
                'Preset pickup_model is missing %s. Re-teach tag grasps.'
                % field)
    normalize_quaternion(model['orientation_xyzw_base'])
    normalize_axis(model['approach_axis_xyz_base'])
    float(model['contact_z_base'])
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


def record_tag_grasp_in_preset(preset, tag_id, tag_pose, grasp_pose,
                               approach_axis_base=None):
    entry = preset.setdefault('tags', {}).setdefault(str(tag_id), {})
    entry['grasp_offset_xy_base'] = [
        float(grasp_pose.pose.position.x - tag_pose.pose.position.x),
        float(grasp_pose.pose.position.y - tag_pose.pose.position.y),
    ]
    if 'pickup_model' not in preset:
        preset['pickup_model'] = {
            'orientation_xyzw_base': normalize_quaternion(
                quaternion_msg_to_list(grasp_pose.pose.orientation)),
            'approach_axis_xyz_base': normalize_axis(
                approach_axis_base or [-1.0, 0.0, 0.0]),
            'contact_z_base': float(grasp_pose.pose.position.z),
        }
    for legacy_field in (
            'grasp_ee_in_tag',
            'grasp_position_offset_in_base',
            'grasp_orientation_in_base',
            'grasp_approach_axis_in_base',
            'grasp_joint_values'):
        entry.pop(legacy_field, None)
    return entry['grasp_offset_xy_base']


def record_tag_place_in_preset(preset, tag_id, place_pose):
    entry = preset.setdefault('tags', {}).setdefault(str(tag_id), {})
    entry['place_ee_in_base'] = pose_to_transform(place_pose)
    entry['place_orientation_in_base'] = normalize_quaternion(
        quaternion_msg_to_list(place_pose.pose.orientation))
    entry['place_approach_axis_in_base'] = [0.0, 0.0, 1.0]
    entry.pop('place_joint_values', None)
    return entry['place_ee_in_base']


def record_idle_in_preset(preset, arm):
    preset['idle_joint_values'] = [
        float(value) for value in arm.get_current_joint_values()
    ]
    return preset['idle_joint_values']


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
    rospy.loginfo('Waiting for pump service: switch_pump_status')
    rospy.wait_for_service('switch_pump_status', timeout=5.0)
    return rospy.ServiceProxy('switch_pump_status', get_mirobot_pump_type())


def get_startup_home_type():
    try:
        from std_srvs.srv import Trigger
        return Trigger
    except ImportError:
        raise RuntimeError(
            'std_srvs/Trigger is unavailable. Source the ROS workspaces first.')


def run_startup_home(args):
    service_name = getattr(
        args, 'startup_home_service', DEFAULT_STARTUP_HOME_SERVICE)
    wait_seconds = getattr(args, 'startup_home_wait_seconds', 8.0)
    settle_seconds = getattr(args, 'startup_home_settle_seconds', 3.0)
    rospy.loginfo('Running startup homing through %s.', service_name)
    rospy.wait_for_service(service_name, timeout=wait_seconds)
    response = rospy.ServiceProxy(service_name, get_startup_home_type())()
    if not response.success:
        raise RuntimeError(
            'Startup homing service failed: %s' % response.message)
    rospy.sleep(settle_seconds)


def set_pump(pump_proxy, enabled):
    rospy.loginfo('Pump %s', 'ON' if enabled else 'OFF')
    response = pump_proxy(enabled)
    if not response.Sucess:
        raise RuntimeError('Pump service returned failure.')


def wait_for_tag_pose_in_base(listener, args, tag_id):
    tag_frame = 'tag_%d' % tag_id
    deadline = rospy.Time.now() + rospy.Duration(args.tf_timeout)
    sample_seconds = getattr(
        args, 'tag_sample_seconds', DEFAULT_TAG_SAMPLE_SECONDS)
    min_samples = getattr(
        args, 'tag_min_samples', DEFAULT_TAG_MIN_SAMPLES)
    max_mad_m = getattr(
        args, 'tag_max_mad_m', DEFAULT_TAG_MAX_MAD_M)
    max_age_seconds = getattr(
        args, 'tag_max_age_seconds', DEFAULT_TAG_MAX_AGE_SECONDS)
    samples = []
    seen_stamps = set()
    first_sample_wall_time = None
    while not rospy.is_shutdown() and rospy.Time.now() < deadline:
        try:
            common_time = listener.getLatestCommonTime(
                args.base_frame, tag_frame)
            age_seconds = (rospy.Time.now() - common_time).to_sec()
            if age_seconds < 0.0 or age_seconds > max_age_seconds:
                rospy.sleep(0.02)
                continue
            trans, rot = listener.lookupTransform(
                args.base_frame, tag_frame, common_time)
            stamp_ns = int(common_time.to_nsec())
            if append_unique_tag_sample(
                    samples, seen_stamps, stamp_ns, trans, rot):
                if first_sample_wall_time is None:
                    first_sample_wall_time = rospy.Time.now()
            enough_time = (
                first_sample_wall_time is not None and
                (rospy.Time.now() - first_sample_wall_time).to_sec() >=
                sample_seconds)
            if enough_time and len(samples) >= min_samples:
                filtered = filter_tag_translation_samples(
                    samples, min_samples=min_samples,
                    max_axis_mad_m=max_mad_m)
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
                    'tag_%d stable TF: inliers=%d/%d MAD_mm=[%.2f, %.2f, %.2f]',
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
        raise RuntimeError(
            'tag_%d did not provide enough stable, fresh, unique TF samples: '
            'collected %d, need %d within %.1fs.'
            % (tag_id, len(samples), min_samples, args.tf_timeout))
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
            raise UserAbort('Interrupted while waiting for Enter.')
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
        except (select.error, IOError) as exc:
            if exc.args and exc.args[0] == errno.EINTR:
                continue
            raise
        except KeyboardInterrupt:
            raise UserAbort('Interrupted by user.')
        if not ready:
            continue
        try:
            line = sys.stdin.readline()
        except EOFError:
            raise RuntimeError('Input closed while waiting for Enter.')
        except KeyboardInterrupt:
            raise UserAbort('Interrupted by user.')
        if line == '':
            raise RuntimeError('Input closed while waiting for Enter.')
        value = line.strip().lower()
        if value in ('q', 'quit', 'exit'):
            raise UserAbort('Aborted by user.')
        return


def pose_to_text(name, pose_stamped):
    pose = pose_stamped.pose
    return ('%s position=(%.4f, %.4f, %.4f) orientation=(%.4f, %.4f, %.4f, %.4f)'
            % (name,
               pose.position.x, pose.position.y, pose.position.z,
               pose.orientation.x, pose.orientation.y,
               pose.orientation.z, pose.orientation.w))


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
        'taught_pre_grasp': (0.95, 0.75, 0.1),
        'taught_grasp': (0.95, 0.2, 0.2),
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
    rospy.loginfo('Executing %s', pose_to_text(label, target_pose))
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
                'MoveIt failed during %s. Waiting for joint state to settle and retrying once.',
                label)
            rospy.sleep(0.3)
    raise RuntimeError('MoveIt failed during %s.' % label)


def execute_joint_values(arm, joint_values, label):
    rospy.loginfo('Executing %s joint_values=%s', label, joint_values)
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
                'MoveIt failed during %s. Waiting for joint state to settle and retrying once.',
                label)
            rospy.sleep(0.5)
    raise RuntimeError('MoveIt failed during %s.' % label)


def log_current_pose(arm, label):
    try:
        current_pose = arm.get_current_pose()
    except Exception as exc:
        rospy.logwarn('Could not read current end-effector pose after %s: %s',
                      label, exc)
        return None
    rospy.loginfo(pose_to_text(label, current_pose))
    return current_pose


def execute_cartesian_pose(arm, target_pose, label, eef_step=0.005,
                           jump_threshold=0.0, retry_without_collisions=False,
                           fallback_to_pose=False):
    rospy.loginfo('Executing cartesian %s', pose_to_text(label, target_pose))
    for attempt in range(2):
        arm.set_start_state_to_current_state()
        plan, fraction = arm.compute_cartesian_path(
            [copy.deepcopy(target_pose.pose)], eef_step, jump_threshold, True)
        if fraction < 0.999:
            if retry_without_collisions:
                rospy.logwarn(
                    'MoveIt cartesian path during %s reached only %.3f with collision checking. '
                    'Retrying without collision checking.',
                    label, fraction)
                arm.set_start_state_to_current_state()
                plan, fraction = arm.compute_cartesian_path(
                    [copy.deepcopy(target_pose.pose)], eef_step, jump_threshold,
                    False)
            if fraction < 0.999 and fallback_to_pose:
                rospy.logwarn(
                    'MoveIt cartesian path during %s still reached only %.3f. '
                    'Falling back to normal pose planning.',
                    label, fraction)
                execute_pose(arm, target_pose, label + '_pose_fallback')
                return
            if fraction < 0.999:
                raise RuntimeError(
                    'MoveIt failed to compute a full cartesian path during %s (fraction=%.3f).'
                    % (label, fraction))
        if not plan.joint_trajectory.points:
            raise RuntimeError('MoveIt returned an empty cartesian trajectory during %s.' % label)
        success = arm.execute(plan, wait=True)
        arm.stop()
        arm.clear_pose_targets()
        if success:
            settle_after_motion()
            return
        if attempt == 0:
            rospy.logwarn(
                'MoveIt failed during cartesian %s. Holding current pose, waiting for '
                'joint state to settle, replanning from current state and retrying once.',
                label)
            rospy.sleep(0.5)
    raise RuntimeError('MoveIt failed during cartesian %s.' % label)


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
                'Dry run enabled. Teach assist motion for tag_%d was skipped.',
                tag_id)
        else:
            execute_pose(arm, assist_pose, 'tag_%d_teach_assist_front' % tag_id)
            log_current_pose(arm, '步骤 2 完成：MoveIt 当前末端位姿')
    except RuntimeError as exc:
        rospy.logwarn(
            '步骤 2 失败：tag_%d 自动到前方安全点失败：%s。请在 RViz 里手动移动到 tag 前方。',
            tag_id, exc)


def prompt_and_record_grasp(args, arm, preset, tag_id, tag_pose):
    prompt_enter(
        '步骤 3：记录 tag_%d 的抓取接触姿态\n'
        '现在请在 RViz 里微调吸盘末端，主要调 XY，必要时小幅调 Z。\n'
        '目标：泵头水平、吸盘刚好贴住 tag 面。RViz Plan/Execute 到位后回这里按 Enter。'
        % tag_id)
    settle_seconds = getattr(args, 'teach_settle_seconds', 0.0)
    if settle_seconds > 0.0:
        rospy.loginfo('等待 %.2fs，让关节状态刷新稳定后再记录抓取点。', settle_seconds)
        rospy.sleep(settle_seconds)
    grasp_pose = arm.get_current_pose()
    record_tag_grasp_in_preset(
        preset, tag_id, tag_pose, grasp_pose,
        approach_axis_base=args.pickup_approach_axis_base)
    rospy.loginfo('步骤 3 完成：已记录 tag_%d 抓取接触姿态。', tag_id)
    rospy.loginfo(pose_to_text('tag_%d_grasp_ee_in_base' % tag_id, grasp_pose))


def prompt_and_record_place(args, arm, preset, tag_id):
    prompt_enter(
        '步骤 4：记录 tag_%d 的载物仓释放姿态\n'
        '请在 RViz 里移动吸盘到对应载物仓释放位置，Plan/Execute 到位后回这里按 Enter。'
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
        prompt_and_record_grasp(args, arm, preset, tag_id, tag_pose)
        prompt_and_record_place(args, arm, preset, tag_id)
    save_preset(args.preset_file, preset,
                overwrite=(preset_existed or args.overwrite))
    rospy.loginfo('Saved taught tag preset: %s', args.preset_file)


def teach_tag_grasp(args, arm):
    preset = load_preset_for_grasp_teach(args.preset_file)
    require_field_overwrite(preset, args.sequence, 'grasp_offset_xy_base',
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
        prompt_and_record_grasp(args, arm, preset, tag_id, tag_pose)
        rospy.loginfo('tag_%d 原来的载物仓释放姿态已保留，不会覆盖。', tag_id)
    save_preset(args.preset_file, preset, overwrite=True)
    rospy.loginfo('Saved taught tag preset: %s', args.preset_file)


def teach_tag_place(args, arm):
    preset = load_preset(args.preset_file)
    require_field_overwrite(preset, args.sequence, 'place_ee_in_base',
                            args.overwrite)
    for tag_id in args.sequence:
        require_tag_fields(preset, tag_id, ['grasp_offset_xy_base'],
                           'teach_tag_place')
    require_v2_pickup_model(preset)
    total_tags = len(args.sequence)
    for index, tag_id in enumerate(args.sequence, 1):
        rospy.loginfo('准备重采 tag_%d 放置点（当前第 %d/%d 个），抓取姿态会保留。',
                      tag_id, index, total_tags)
        prompt_and_record_place(args, arm, preset, tag_id)
    save_preset(args.preset_file, preset, overwrite=True)
    rospy.loginfo('Saved taught tag preset: %s', args.preset_file)


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
    rospy.loginfo('Saved idle joint values: %s', idle_joint_values)
    rospy.loginfo('Saved taught tag preset: %s', args.preset_file)


def run_taught_sequence(args, arm, pump_proxy):
    preset = load_preset(args.preset_file)
    require_preset_tags(preset, args.sequence)
    pickup_model = require_v2_pickup_model(preset)
    listener = tf.TransformListener()
    rospy.sleep(0.5)
    for tag_id in args.sequence:
        entry = preset['tags'][str(tag_id)]
        require_tag_fields(
            preset, tag_id,
            ['grasp_offset_xy_base', 'place_ee_in_base'],
            'run_taught_sequence')
        tag_pose = wait_for_tag_pose_in_base(listener, args, tag_id)
        grasp_pose = compute_v2_grasp_pose(
            tag_pose, pickup_model, entry, args.base_frame)
        pre_grasp_pose = build_v2_pre_grasp_pose(
            grasp_pose, pickup_model,
            args.approach_gap, args.base_frame)
        place_pose = transform_to_pose(args.base_frame, entry['place_ee_in_base'])
        pre_place_pose = build_pre_place_pose(
            place_pose, args.place_approach_gap, args.base_frame)

        rospy.loginfo(pose_to_text('tag_%d_in_base' % tag_id, tag_pose))
        rospy.loginfo(pose_to_text('taught_pre_grasp', pre_grasp_pose))
        rospy.loginfo(pose_to_text('taught_grasp', grasp_pose))
        rospy.loginfo(pose_to_text('taught_pre_place', pre_place_pose))
        rospy.loginfo(pose_to_text('taught_place', place_pose))
        publish_debug_geometry(args.base_frame, {
            'tag_in_base': tag_pose,
            'taught_pre_grasp': pre_grasp_pose,
            'taught_grasp': grasp_pose,
            'taught_pre_place': pre_place_pose,
            'taught_place': place_pose,
        })

        if args.dry_run:
            rospy.logwarn('Dry run enabled for tag_%d. No arm motion will be executed.', tag_id)
            continue

        holding_object = False
        try:
            execute_pose(arm, pre_grasp_pose, 'taught_pre_grasp')
            execute_cartesian_pose(arm, grasp_pose, 'taught_grasp')
            set_pump(pump_proxy, True)
            holding_object = True
            rospy.sleep(0.8)
            execute_cartesian_pose(arm, pre_grasp_pose, 'taught_grasp_retreat')
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
                        'Motion failed after pump was enabled. Turning pump OFF before aborting.')
                    set_pump(pump_proxy, False)
                except Exception as pump_exc:
                    rospy.logerr('Failed to turn pump OFF after motion error: %s',
                                 pump_exc)
            raise

    if args.dry_run and args.debug_hold_seconds > 0.0:
        rospy.loginfo('Holding debug pose topics for %.1f seconds.',
                      args.debug_hold_seconds)
        rospy.sleep(args.debug_hold_seconds)


def main():
    args = parse_args(sys.argv)
    global MOTION_SETTLE_SECONDS
    MOTION_SETTLE_SECONDS = args.motion_settle_seconds
    rospy.init_node('mirobot_pick_test_tag', anonymous=False)
    moveit_commander.roscpp_initialize(sys.argv)
    arm = None
    try:
        arm = build_move_group(
            args.group, args.base_frame, args.velocity_scale,
            args.acceleration_scale, args.planning_time,
            not args.disable_replanning)
        if args.mode == 'teach_tag_sequence':
            teach_tag_sequence(args, arm)
        elif args.mode == 'teach_tag_grasp':
            teach_tag_grasp(args, arm)
        elif args.mode == 'teach_tag_place':
            teach_tag_place(args, arm)
        elif args.mode == 'teach_idle':
            teach_idle(args, arm)
        else:
            pump_proxy = None if args.dry_run else get_pump_proxy()
            run_taught_sequence(args, arm, pump_proxy)
        rospy.loginfo('Tag taught sequence finished.')
    except UserAbort as exc:
        rospy.logwarn(str(exc))
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        rospy.logwarn('Interrupted by user.')
    except Exception as exc:
        rospy.logerr(str(exc))
        raise
    finally:
        moveit_commander.roscpp_shutdown()


if __name__ == '__main__':
    main()
