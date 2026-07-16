#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import copy
import math
import os
import sys
import tempfile
import threading

if sys.version_info[0] != 2:
    sys.stderr.write(
        'mirobot_pick_test.py 必须使用 Python 2 运行，因为当前 ROS Melodic 的 tf/moveit 模块是 Python 2 版本。\n'
        '请先 source ROS 工作空间，再用下面的命令运行：\n'
        'source /opt/ros/melodic/setup.bash\n'
        'source /home/eaibot/mirobot_ws/devel/setup.bash\n'
        'source /home/eaibot/handeye-calib/devel/setup.bash\n'
        'python2 /home/eaibot/handeye-calib/src/mirobot_pick_test.py --mode home\n'
    )
    sys.exit(1)

import rospy
import tf
import moveit_commander
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray

from block_detector_protocol import DetectorClient
from block_grasp_sequence import run_block_sequence
from block_grasp_vision import (
    LocalizationError,
    compute_link_targets,
    deproject_pixel,
    find_block_quadrilateral,
    render_debug_image,
    rotate_vector_by_quaternion,
    sample_depth_m,
    tool_axis_vector,
    undistort_pixel,
    validate_axis_alignment,
    validate_rgbd_metadata,
    validate_workspace_points,
)


SUPPLY_TAG_MAP = {
    'basic': {
        'tag_id': 1,
        'label': '基本生活物资',
        'grasp_offsets': {
            'grasp_x': 0.0,
            'y_offset': 0.0,
            'z_offset': 0.09,
        },
    },
    'medical': {'tag_id': 2, 'label': '医疗包'},
    'recyclable': {'tag_id': 3, 'label': '常规消杀剂'},
    'hazardous': {'tag_id': 4, 'label': '生物危害专用消杀剂'},
}

DEFAULT_GRASP_OFFSETS = {
    'grasp_x': -0.045,
    'y_offset': 0.0,
    'z_offset': 0.003,
}

WRIST_FORWARD_JOINT5 = -1.5709534265016345
TF_LISTENER_WARMUP_SECONDS = 0.2

TOOL_AXES = ('x', '-x', 'y', '-y', 'z', '-z')
BLOCK_TARGETS = ('power', 'fire', 'gas', 'support')

try:
    STRING_TYPES = (basestring,)
except NameError:
    STRING_TYPES = (str,)


def _normalize_signed_args(argv):
    signed_options = set([
        '--tool-offset', '--max-tool-camera-angle-deg', '--approach-gap',
        '--base-min-z', '--base-max-radius', '--wrist-forward-tolerance',
    ])
    normalized = []
    index = 0
    while index < len(argv):
        token = argv[index]
        next_value = argv[index + 1] if index + 1 < len(argv) else None
        if (next_value is not None and
                ((token in signed_options and next_value.startswith('-')) or
                 (token == '--tool-axis' and next_value in ('-x', '-y', '-z')))):
            normalized.append(token + '=' + next_value)
            index += 2
            continue
        normalized.append(token)
        index += 1
    return normalized


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Mirobot test helper for home, pump and tag grasp.')
    parser.add_argument('--mode', choices=['home', 'pump', 'grasp', 'place', 'pick_place', 'pick_lift_place', 'current_pose', 'wrist_forward', 'block_grasp'], default='grasp')
    parser.add_argument('--tag-id', type=int, default=0,
                        help='AprilTag ID。比赛物资可直接用 --supply basic|medical|recyclable|hazardous。')
    parser.add_argument('--supply', choices=sorted(SUPPLY_TAG_MAP.keys()),
                        help='按比赛物资类型自动映射到 AprilTag：basic=1, medical=2, recyclable=3, hazardous=4。')
    parser.add_argument('--camera-frame', default='camera_rgb_optical_frame')
    parser.add_argument('--base-frame', default='base')
    parser.add_argument('--group', default='manipulator')
    parser.add_argument('--pre-x', type=float,
                        help='Base-frame x offset used by the pre-grasp pose. If omitted, it is derived from --grasp-x and --approach-gap.')
    parser.add_argument('--grasp-x', type=float,
                        help='Base-frame x offset used by the grasp pose. 不填时会先尝试用物资预设，再回退到通用默认值。')
    parser.add_argument('--approach-axis', choices=['x', 'z', 'front'], default='x',
                        help='接近抓取点的方向。x 保持原来的前后接近；z 表示先到抓取点正上方，再竖直下压；front 表示面朝 AprilTag 正面接近。')
    parser.add_argument('--approach-gap', type=float, default=0.005,
                        help='pre_grasp 和 grasp 之间的间距。approach-axis=x 时表示 x 方向距离；approach-axis=z 时表示上方高度；approach-axis=front 时表示离标签面更远的正面预留距离。')
    parser.add_argument('--y-offset', type=float,
                        help='Base-frame lateral offset used by both pre-grasp and grasp poses. 不填时会先尝试用物资预设，再回退到通用默认值。')
    parser.add_argument('--z-offset', type=float,
                        help='Base-frame vertical offset used by both pre-grasp and grasp poses. 不填时会先尝试用物资预设，再回退到通用默认值。')
    parser.add_argument('--front-tool-roll-deg', type=float, default=0.0,
                        help='approach-axis=front 时，末端相对 AprilTag 姿态的附加 roll 角度，单位度。')
    parser.add_argument('--front-tool-pitch-deg', type=float, default=0.0,
                        help='approach-axis=front 时，末端相对 AprilTag 姿态的附加 pitch 角度，单位度。')
    parser.add_argument('--front-tool-yaw-deg', type=float, default=0.0,
                        help='approach-axis=front 时，末端相对 AprilTag 姿态的附加 yaw 角度，单位度。')
    parser.add_argument('--place-x', type=float,
                        help='放置点在 base 坐标系下的绝对 x 坐标。')
    parser.add_argument('--place-y', type=float,
                        help='放置点在 base 坐标系下的绝对 y 坐标。')
    parser.add_argument('--place-z', type=float,
                        help='放置点在 base 坐标系下的绝对 z 坐标。')
    parser.add_argument('--place-approach-gap', type=float, default=0.02,
                        help='放置时先停在目标点正上方多高的位置，再竖直下放。')
    parser.add_argument('--lift-height', type=float, default=0.05,
                        help='pick_lift_place 模式下，抓住后先向上抬高多少米，再原地下放释放。')
    parser.add_argument('--velocity-scale', type=float, default=0.2)
    parser.add_argument('--acceleration-scale', type=float, default=0.2)
    parser.add_argument('--planning-time', type=float, default=5.0)
    parser.add_argument('--disable-replanning', action='store_true',
                        help='Fail fast while tuning offsets instead of retrying multiple planning attempts.')
    parser.add_argument('--tf-timeout', type=float, default=5.0)
    parser.add_argument('--pump-seconds', type=float, default=2.0)
    parser.add_argument('--dry-run', action='store_true',
                        help='Only print the computed poses without moving the arm.')
    parser.add_argument('--debug-hold-seconds', type=float, default=0.0,
                        help='When used with --dry-run, keep debug pose topics alive for RViz inspection.')
    parser.add_argument('--skip-home', action='store_true')
    parser.add_argument('--keep-pump-on', action='store_true',
                        help='Keep the pump enabled after the script finishes.')
    parser.add_argument('--skip-post-grasp-retreat', action='store_true',
                        help='抓住后不回到 pre_grasp。只建议在 --mode grasp 的调试场景使用。')
    parser.add_argument('--use-tag-orientation', action='store_true',
                        help='Use the transformed tag orientation for grasp poses.')
    parser.add_argument('--wrist-forward', action='store_true',
                        help='抓取前先把 joint5 转到吸盘水平朝前的已验证姿态。')
    parser.add_argument('--wrist-forward-joint5', type=float, default=WRIST_FORWARD_JOINT5,
                        help='--wrist-forward / --mode wrist_forward 使用的 joint5 目标弧度。')
    parser.add_argument('--block-target', choices=BLOCK_TARGETS)
    parser.add_argument('--detector-request-fd', type=int)
    parser.add_argument('--detector-response-fd', type=int)
    parser.add_argument('--rgb-topic', default='/camera/rgb/image_raw')
    parser.add_argument('--registered-depth-topic', default='/camera/depth_registered/image_raw')
    parser.add_argument('--rgb-camera-info-topic', default='/camera/rgb/camera_info')
    parser.add_argument('--rgbd-timeout', type=float, default=5.0)
    parser.add_argument('--rgbd-slop', type=float, default=0.05)
    parser.add_argument('--depth-radius', type=int, default=3)
    parser.add_argument('--depth-min-m', type=float, default=0.10)
    parser.add_argument('--depth-max-m', type=float, default=2.00)
    parser.add_argument('--depth-min-valid-ratio', type=float, default=0.50)
    parser.add_argument('--depth-max-mad-m', type=float, default=0.010)
    parser.add_argument('--roi-margin', type=float, default=0.40)
    parser.add_argument('--roi-min-area-pixels', type=float, default=1000.0)
    parser.add_argument('--roi-max-aspect-error', type=float, default=0.25)
    parser.add_argument('--roi-min-rectangularity', type=float, default=0.75)
    parser.add_argument('--roi-ambiguity-ratio', type=float, default=0.90)
    parser.add_argument('--tool-offset', type=float)
    parser.add_argument('--tool-axis', choices=TOOL_AXES)
    parser.add_argument('--max-tool-camera-angle-deg', type=float, default=20.0)
    parser.add_argument('--stop-at-pre-grasp', action='store_true')
    parser.add_argument('--debug-image', default='/tmp/block_grasp_debug.png')
    parser.add_argument('--base-min-z', type=float, default=0.04)
    parser.add_argument('--base-max-radius', type=float, default=0.50)
    parser.add_argument('--wrist-forward-tolerance', type=float, default=0.03)
    ros_argv = rospy.myargv(argv)[1:]
    return parser.parse_args(_normalize_signed_args(ros_argv))


def _require_finite(value, option):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise RuntimeError('{} must be a finite number.'.format(option))
    if math.isnan(number) or math.isinf(number):
        raise RuntimeError('{} must be a finite number.'.format(option))
    return number


def require_block_args(args):
    """Fail closed before opening a detector pipe or touching ROS topics."""
    if not isinstance(args.block_target, STRING_TYPES) or not args.block_target.strip():
        raise RuntimeError('--block-target is required for block_grasp mode.')
    if args.block_target not in BLOCK_TARGETS:
        raise RuntimeError('--block-target is unsupported: {}'.format(args.block_target))
    for option in ('rgb_topic', 'registered_depth_topic', 'rgb_camera_info_topic'):
        value = getattr(args, option)
        if not isinstance(value, STRING_TYPES) or not value.strip():
            raise RuntimeError('--{} must be non-empty.'.format(option.replace('_', '-')))
    if not isinstance(args.debug_image, STRING_TYPES) or not args.debug_image.strip():
        raise RuntimeError('--debug-image must be non-empty.')
    for option in ('detector_request_fd', 'detector_response_fd'):
        value = getattr(args, option)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError('--{} must be a non-negative file descriptor.'.format(
                option.replace('_', '-')))
    if args.detector_request_fd == args.detector_response_fd:
        raise RuntimeError('Detector request and response file descriptors must differ.')

    numeric = (
        ('rgbd_timeout', '--rgbd-timeout'),
        ('rgbd_slop', '--rgbd-slop'),
        ('depth_min_m', '--depth-min-m'),
        ('depth_max_m', '--depth-max-m'),
        ('depth_min_valid_ratio', '--depth-min-valid-ratio'),
        ('depth_max_mad_m', '--depth-max-mad-m'),
        ('roi_margin', '--roi-margin'),
        ('roi_min_area_pixels', '--roi-min-area-pixels'),
        ('roi_max_aspect_error', '--roi-max-aspect-error'),
        ('roi_min_rectangularity', '--roi-min-rectangularity'),
        ('roi_ambiguity_ratio', '--roi-ambiguity-ratio'),
        ('approach_gap', '--approach-gap'),
        ('velocity_scale', '--velocity-scale'),
        ('acceleration_scale', '--acceleration-scale'),
        ('tf_timeout', '--tf-timeout'),
        ('wrist_forward_joint5', '--wrist-forward-joint5'),
        ('max_tool_camera_angle_deg', '--max-tool-camera-angle-deg'),
        ('base_min_z', '--base-min-z'),
        ('base_max_radius', '--base-max-radius'),
        ('wrist_forward_tolerance', '--wrist-forward-tolerance'),
    )
    values = {}
    for attribute, option in numeric:
        values[attribute] = _require_finite(getattr(args, attribute), option)
    if args.tool_offset is not None:
        values['tool_offset'] = _require_finite(args.tool_offset, '--tool-offset')

    if values['rgbd_timeout'] <= 0.0:
        raise RuntimeError('--rgbd-timeout must be positive.')
    if values['rgbd_slop'] < 0.0 or values['rgbd_slop'] > 1.0:
        raise RuntimeError('--rgbd-slop must be in [0, 1].')
    if isinstance(args.depth_radius, bool) or args.depth_radius < 0 or args.depth_radius > 100:
        raise RuntimeError('--depth-radius must be an integer in [0, 100].')
    if values['depth_min_m'] <= 0.0 or values['depth_max_m'] <= values['depth_min_m']:
        raise RuntimeError('Depth range must be positive and increasing.')
    if not 0.0 < values['depth_min_valid_ratio'] <= 1.0:
        raise RuntimeError('--depth-min-valid-ratio must be in (0, 1].')
    if values['depth_max_mad_m'] < 0.0:
        raise RuntimeError('--depth-max-mad-m must be non-negative.')
    if not 0.0 <= values['roi_margin'] <= 2.0:
        raise RuntimeError('--roi-margin must be in [0, 2].')
    if values['roi_min_area_pixels'] <= 0.0:
        raise RuntimeError('--roi-min-area-pixels must be positive.')
    if not 0.0 <= values['roi_max_aspect_error'] <= 1.0:
        raise RuntimeError('--roi-max-aspect-error must be in [0, 1].')
    if not 0.0 < values['roi_min_rectangularity'] <= 1.0:
        raise RuntimeError('--roi-min-rectangularity must be in (0, 1].')
    if not 0.0 < values['roi_ambiguity_ratio'] <= 1.0:
        raise RuntimeError('--roi-ambiguity-ratio must be in (0, 1].')
    if not 0.0 < values['approach_gap'] <= 0.15:
        raise RuntimeError('--approach-gap must be in (0, 0.15].')
    if not 0.0 < values['velocity_scale'] <= 1.0:
        raise RuntimeError('--velocity-scale must be in (0, 1].')
    if not 0.0 < values['acceleration_scale'] <= 1.0:
        raise RuntimeError('--acceleration-scale must be in (0, 1].')
    if values['tf_timeout'] <= 0.0:
        raise RuntimeError('--tf-timeout must be positive.')
    if not 0.0 < values['max_tool_camera_angle_deg'] < 90.0:
        raise RuntimeError('--max-tool-camera-angle-deg must be in (0, 90).')
    if values['base_min_z'] < 0.0 or values['base_max_radius'] <= 0.0:
        raise RuntimeError('Base workspace z/radius limits are invalid.')
    if values['wrist_forward_tolerance'] <= 0.0:
        raise RuntimeError('--wrist-forward-tolerance must be positive.')

    if (args.tool_offset is None) != (args.tool_axis is None):
        raise RuntimeError('--tool-offset and --tool-axis must be provided together.')
    if args.tool_offset is not None and not 0.0 < values['tool_offset'] <= 0.30:
        raise RuntimeError('--tool-offset must be in (0, 0.30].')
    if (not args.dry_run or args.stop_at_pre_grasp) and args.tool_offset is None:
        raise RuntimeError('Tool offset and axis are required outside surface-only dry-run.')
    return args


def _unsubscribe_message_filter(subscriber):
    if subscriber is None:
        return
    unregister = getattr(subscriber, 'unregister', None)
    if callable(unregister):
        unregister()
        return
    wrapped = getattr(subscriber, 'sub', None)
    unregister = getattr(wrapped, 'unregister', None)
    if callable(unregister):
        unregister()


def capture_rgbd_once(args):
    """Capture exactly the first synchronized registered RGB-D pair."""
    try:
        import message_filters
        from cv_bridge import CvBridge
        from sensor_msgs.msg import CameraInfo, Image
    except ImportError as exc:
        raise RuntimeError('RGB-D ROS dependencies are unavailable: {}'.format(exc))

    bridge = CvBridge()
    ready = threading.Event()
    lock = threading.Lock()
    state = {'capture': None, 'error': None}
    rgb_subscriber = None
    depth_subscriber = None

    def synchronized_callback(rgb_message, depth_message):
        with lock:
            if ready.is_set():
                return
            try:
                rgb = bridge.imgmsg_to_cv2(rgb_message, desired_encoding='bgr8')
                depth = bridge.imgmsg_to_cv2(depth_message, desired_encoding='passthrough')
                state['capture'] = {
                    'rgb': rgb,
                    'depth': depth,
                    'rgb_header': copy.deepcopy(rgb_message.header),
                    'depth_header': copy.deepcopy(depth_message.header),
                    'depth_encoding': depth_message.encoding,
                }
            except Exception as exc:
                state['error'] = 'cv_bridge conversion failed: {}'.format(exc)
            ready.set()

    try:
        rgb_subscriber = message_filters.Subscriber(args.rgb_topic, Image)
        depth_subscriber = message_filters.Subscriber(args.registered_depth_topic, Image)
        synchronizer = message_filters.ApproximateTimeSynchronizer(
            [rgb_subscriber, depth_subscriber], queue_size=10, slop=args.rgbd_slop)
        synchronizer.registerCallback(synchronized_callback)
        try:
            camera_info = rospy.wait_for_message(
                args.rgb_camera_info_topic, CameraInfo, timeout=args.rgbd_timeout)
        except rospy.ROSException as exc:
            raise RuntimeError('Timed out waiting for RGB CameraInfo: {}'.format(exc))
        if not ready.wait(args.rgbd_timeout):
            raise RuntimeError('Timed out waiting for a synchronized registered RGB-D pair.')
        if state['error'] is not None:
            raise RuntimeError(state['error'])
        capture = state['capture']
        if capture is None:
            raise RuntimeError('RGB-D callback completed without a capture.')
        metadata = validate_rgbd_metadata(
            capture['rgb'], capture['depth'], capture['rgb_header'],
            capture['depth_header'], camera_info, capture['depth_encoding'],
            args.rgbd_slop, lambda stamp: stamp.to_sec())
        capture['camera_info'] = copy.deepcopy(camera_info)
        capture['metadata'] = metadata
        return capture
    except LocalizationError as exc:
        raise RuntimeError('Registered RGB-D validation failed: {}'.format(exc))
    finally:
        for subscriber in (rgb_subscriber, depth_subscriber):
            try:
                _unsubscribe_message_filter(subscriber)
            except Exception as exc:
                rospy.logwarn('Failed to unsubscribe RGB-D subscriber: %s', exc)


def _open_detector_streams(args):
    request_stream = response_stream = None
    try:
        request_stream = os.fdopen(args.detector_request_fd, 'w', 1)
        response_stream = os.fdopen(args.detector_response_fd, 'r', 1)
        return request_stream, response_stream
    except Exception:
        if request_stream is not None:
            request_stream.close()
        if response_stream is not None:
            response_stream.close()
        raise


def localize_block(args, capture):
    """Ask the Python 3 detector, refine geometry, then deproject the surface."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError('OpenCV/numpy unavailable for block localization: {}'.format(exc))

    request_stream = response_stream = None
    image_fd = None
    image_path = None
    try:
        request_stream, response_stream = _open_detector_streams(args)
        detector = DetectorClient(request_stream, response_stream)
        image_fd, image_path = tempfile.mkstemp(prefix='block_rgb_', suffix='.png')
        os.close(image_fd)
        image_fd = None
        if not cv2.imwrite(image_path, capture['rgb']):
            raise RuntimeError('Failed to write exact RGB detector PNG: {}'.format(image_path))
        response = detector.detect(image_path, args.block_target)

        localization = find_block_quadrilateral(
            capture['rgb'], response['box'], args.roi_margin,
            args.roi_min_area_pixels, args.roi_max_aspect_error,
            args.roi_min_rectangularity, args.roi_ambiguity_ratio)
        depth_m, depth_stats = sample_depth_m(
            capture['depth'], localization['center'], capture['depth_encoding'],
            args.depth_radius, args.depth_min_m, args.depth_max_m,
            args.depth_min_valid_ratio, args.depth_max_mad_m)
        info = capture['camera_info']
        matrix = np.asarray(info.K, dtype=np.float64).reshape(3, 3)
        corrected_center = undistort_pixel(
            localization['center'][0], localization['center'][1], matrix,
            info.D, info.distortion_model)
        camera_xyz = deproject_pixel(
            corrected_center[0], corrected_center[1], depth_m,
            matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2])

        debug = render_debug_image(
            capture['rgb'], response['box'], localization, args.depth_radius)
        if not cv2.imwrite(args.debug_image, debug):
            raise RuntimeError('Failed to write block debug image: {}'.format(args.debug_image))
        return {
            'target': response['target'],
            'class_id': response['class_id'],
            'class_name': response['class_name'],
            'confidence': response['confidence'],
            'box': tuple(response['box']),
            'corners': tuple(tuple(float(value) for value in corner)
                             for corner in localization['corners']),
            'center': tuple(float(value) for value in localization['center']),
            'undistorted_center': tuple(corrected_center),
            'depth_m': depth_m,
            'depth_stats': depth_stats,
            'camera_xyz': tuple(camera_xyz),
            'rgb_header': copy.deepcopy(capture['rgb_header']),
        }
    except (LocalizationError, OSError) as exc:
        raise RuntimeError('Block localization failed: {}'.format(exc))
    finally:
        if image_fd is not None:
            os.close(image_fd)
        if image_path is not None:
            try:
                os.unlink(image_path)
            except OSError:
                pass
        for stream in (request_stream, response_stream):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass


def resolve_tag_id(args):
    if args.supply:
        supply_info = SUPPLY_TAG_MAP[args.supply]
        rospy.loginfo('Using supply %s -> tag_%d (%s)',
                      args.supply, supply_info['tag_id'], supply_info['label'])
        return supply_info['tag_id']
    return args.tag_id


def resolve_grasp_offsets(args):
    supply_defaults = {}
    if args.supply:
        supply_defaults = SUPPLY_TAG_MAP[args.supply].get('grasp_offsets', {})

    applied_sources = []
    for field_name, fallback_value in DEFAULT_GRASP_OFFSETS.items():
        if getattr(args, field_name) is not None:
            continue

        if field_name in supply_defaults:
            resolved_value = supply_defaults[field_name]
            applied_sources.append('%s=%.4f(supply preset)' % (field_name, resolved_value))
        else:
            resolved_value = fallback_value
            applied_sources.append('%s=%.4f(global default)' % (field_name, resolved_value))

        setattr(args, field_name, resolved_value)

    if applied_sources:
        rospy.loginfo('Resolved grasp offsets: %s', ', '.join(applied_sources))


def build_move_group(group_name, base_frame, velocity_scale, acceleration_scale,
                     planning_time, allow_replanning):
    arm = moveit_commander.MoveGroupCommander(group_name)
    arm.set_pose_reference_frame(base_frame)
    arm.allow_replanning(allow_replanning)
    arm.set_max_velocity_scaling_factor(velocity_scale)
    arm.set_max_acceleration_scaling_factor(acceleration_scale)
    arm.set_planning_time(planning_time)
    return arm


def is_named_target_reached(arm, target_name, tolerance=1e-3):
    current_values = arm.get_current_joint_values()
    target_values = arm.get_named_target_values(target_name)
    active_joints = arm.get_active_joints()

    for index, joint_name in enumerate(active_joints):
        target_value = target_values.get(joint_name)
        if target_value is None:
            continue
        if abs(current_values[index] - target_value) > tolerance:
            return False
    return True


def go_home(arm):
    if is_named_target_reached(arm, 'home'):
        rospy.loginfo('Arm is already at named target: home, skipping execution.')
        return

    rospy.loginfo('Moving arm to named target: home')
    arm.set_start_state_to_current_state()
    arm.set_named_target('home')
    success = arm.go(wait=True)
    arm.stop()
    arm.clear_pose_targets()
    if not success:
        raise RuntimeError('Failed to move to home target.')


def format_joint_values(joint_values):
    return '[' + ', '.join('{:.4f}'.format(value) for value in joint_values) + ']'


def go_wrist_forward(arm, joint5_target):
    current_values = arm.get_current_joint_values()
    if len(current_values) < 5:
        raise RuntimeError('Expected at least 5 active joints, got {}.'.format(len(current_values)))

    target_values = list(current_values)
    target_values[4] = joint5_target
    rospy.loginfo('Moving wrist forward: current=%s target=%s',
                  format_joint_values(current_values),
                  format_joint_values(target_values))

    arm.set_start_state_to_current_state()
    arm.set_joint_value_target(target_values)
    success = arm.go(wait=True)
    arm.stop()
    arm.clear_pose_targets()
    if not success:
        raise RuntimeError('Failed to move wrist forward.')


def get_mirobot_pump_type():
    try:
        from mirobot_urdf_2.srv import mirobotPump
        return mirobotPump
    except ImportError:
        raise RuntimeError(
            '当前环境未加载 mirobot_urdf_2.srv，pump/grasp 模式需要先 source 机械臂工作空间。\n'
            '请执行：\n'
            'source /opt/ros/melodic/setup.bash\n'
            'source /home/eaibot/mirobot_ws/devel/setup.bash\n'
            'source /home/eaibot/handeye-calib/devel/setup.bash\n'
            '然后再运行该脚本。'
        )


def get_pump_proxy():
    rospy.loginfo('Waiting for pump service: switch_pump_status')
    rospy.wait_for_service('switch_pump_status', timeout=5.0)
    return rospy.ServiceProxy('switch_pump_status', get_mirobot_pump_type())


def set_pump(pump_proxy, enabled):
    rospy.loginfo('Pump %s', 'ON' if enabled else 'OFF')
    response = pump_proxy(enabled)
    if not response.Sucess:
        raise RuntimeError('Pump service returned failure.')


def get_apriltag_detection_array_type():
    try:
        from apriltag_ros.msg import AprilTagDetectionArray
        return AprilTagDetectionArray
    except ImportError:
        raise RuntimeError(
            '当前环境未加载 apriltag_ros.msg，无法读取 /tag_detections。\n'
            '请执行：\n'
            'source /opt/ros/melodic/setup.bash\n'
            'source /home/eaibot/mirobot_ws/devel/setup.bash\n'
            'source /home/eaibot/handeye-calib/devel/setup.bash\n'
            '然后再运行该脚本。'
        )


def detection_to_pose_stamped(detection):
    pose = PoseStamped()
    pose.header = detection.pose.header
    pose.pose = copy.deepcopy(detection.pose.pose.pose)
    return pose


def extract_visible_tag_ids(detection_array):
    visible_tag_ids = set()
    for detection in detection_array.detections:
        for tag_id in detection.id:
            visible_tag_ids.add(tag_id)
    return sorted(visible_tag_ids)


def wait_for_tag_pose(listener, camera_frame, tag_frame, tag_id, timeout_sec):
    deadline = rospy.Time.now() + rospy.Duration(timeout_sec)
    detection_array_type = get_apriltag_detection_array_type()
    visible_tag_ids = []

    while not rospy.is_shutdown() and rospy.Time.now() < deadline:
        try:
            now = rospy.Time(0)
            listener.waitForTransform(camera_frame, tag_frame, now, rospy.Duration(0.3))
            trans, rot = listener.lookupTransform(camera_frame, tag_frame, now)
            pose = PoseStamped()
            pose.header.frame_id = camera_frame
            pose.header.stamp = rospy.Time.now()
            pose.pose.position.x = trans[0]
            pose.pose.position.y = trans[1]
            pose.pose.position.z = trans[2]
            pose.pose.orientation.x = rot[0]
            pose.pose.orientation.y = rot[1]
            pose.pose.orientation.z = rot[2]
            pose.pose.orientation.w = rot[3]
            return pose
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            remaining = (deadline - rospy.Time.now()).to_sec()
            if remaining <= 0.0:
                break

            try:
                detection_array = rospy.wait_for_message(
                    '/tag_detections', detection_array_type, timeout=min(0.25, remaining))
                visible_tag_ids = extract_visible_tag_ids(detection_array)
                for detection in detection_array.detections:
                    if tag_id in detection.id:
                        rospy.loginfo('Resolved tag_%d pose from /tag_detections because TF %s was unavailable.',
                                      tag_id, tag_frame)
                        return detection_to_pose_stamped(detection)
            except rospy.ROSException:
                pass

            rospy.sleep(0.05)

    if visible_tag_ids:
        raise RuntimeError(
            'TF for {} was not found. /tag_detections 当前能看到的标签 ID: {}。'
            '请确认目标 tag_{} 在画面里，或者把命令里的 --supply / --tag-id 改对。'
            .format(tag_frame, ', '.join(str(tag) for tag in visible_tag_ids), tag_id)
        )

    raise RuntimeError(
        'TF for {} was not found, and /tag_detections is empty. '
        'Make sure apriltag_ros is running and the target tag is visible.'.format(tag_frame)
    )


def transform_pose(listener, target_frame, pose_stamped, timeout_sec):
    deadline = rospy.Time.now() + rospy.Duration(timeout_sec)
    while not rospy.is_shutdown() and rospy.Time.now() < deadline:
        try:
            listener.waitForTransform(target_frame, pose_stamped.header.frame_id,
                                      rospy.Time(0), rospy.Duration(0.3))
            return listener.transformPose(target_frame, pose_stamped)
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.sleep(0.1)

    raise RuntimeError('Failed to transform pose from {} to {}.'.format(
        pose_stamped.header.frame_id, target_frame))


def transform_pose_at_stamp(listener, target_frame, pose_stamped, timeout_sec):
    """Transform a measured point without silently replacing its capture time."""
    timeout_sec = _require_finite(timeout_sec, '--tf-timeout')
    if timeout_sec <= 0.0:
        raise RuntimeError('--tf-timeout must be positive.')
    if not pose_stamped.header.frame_id:
        raise RuntimeError('Measured pose has an empty source frame.')
    stamp = pose_stamped.header.stamp
    try:
        listener.waitForTransform(
            target_frame, pose_stamped.header.frame_id, stamp,
            rospy.Duration(timeout_sec))
        return listener.transformPose(target_frame, pose_stamped)
    except (tf.Exception, tf.LookupException, tf.ConnectivityException,
            tf.ExtrapolationException) as exc:
        raise RuntimeError(
            'Failed to transform captured pose from {} to {} at RGB stamp: {}'
            .format(pose_stamped.header.frame_id, target_frame, exc))


def warmup_transform_listener(listener, seconds=TF_LISTENER_WARMUP_SECONDS):
    if listener is None:
        raise RuntimeError('TF listener warmup requires a listener.')
    seconds = _require_finite(seconds, 'TF listener warmup seconds')
    if seconds <= 0.0:
        raise RuntimeError('TF listener warmup seconds must be positive.')
    rospy.loginfo('Warming TF listener cache for %.3f seconds before RGB-D capture.',
                  seconds)
    rospy.sleep(seconds)


def make_camera_point_pose(rgb_header, camera_xyz):
    values = []
    for index, value in enumerate(camera_xyz):
        values.append(_require_finite(value, 'camera_xyz[{}]'.format(index)))
    if len(values) != 3:
        raise RuntimeError('camera_xyz must contain exactly three values.')
    if not rgb_header.frame_id:
        raise RuntimeError('RGB header frame must be non-empty.')
    point = PoseStamped()
    point.header = copy.deepcopy(rgb_header)
    point.pose.position.x = values[0]
    point.pose.position.y = values[1]
    point.pose.position.z = values[2]
    point.pose.orientation.x = 0.0
    point.pose.orientation.y = 0.0
    point.pose.orientation.z = 0.0
    point.pose.orientation.w = 1.0
    return point


def is_wrist_forward_reached(arm, joint5_target, tolerance):
    joint5_target = _require_finite(joint5_target, '--wrist-forward-joint5')
    tolerance = _require_finite(tolerance, '--wrist-forward-tolerance')
    if tolerance <= 0.0:
        raise RuntimeError('Wrist-forward tolerance must be positive.')
    active_joints = list(arm.get_active_joints())
    current_values = list(arm.get_current_joint_values())
    if len(active_joints) != len(current_values):
        raise RuntimeError('Active joint names and current joint values differ in length.')
    matches = [index for index, name in enumerate(active_joints)
               if name.split('/')[-1].lower() == 'joint5']
    if len(matches) != 1:
        raise RuntimeError('Could not uniquely locate joint5 in active joints: {}'.format(
            active_joints))
    actual = _require_finite(current_values[matches[0]], 'current joint5')
    return abs(actual - joint5_target) <= tolerance


def _pose_position_tuple(pose_stamped):
    position = pose_stamped.pose.position
    return (float(position.x), float(position.y), float(position.z))


def build_block_poses(args, listener, current_pose, localization,
                      surface_camera_pose=None, surface_base_pose=None):
    if args.tool_offset is None or args.tool_axis is None:
        raise RuntimeError('Tool geometry is required to build block grasp poses.')
    if surface_camera_pose is None:
        surface_camera_pose = make_camera_point_pose(
            localization['rgb_header'], localization['camera_xyz'])
    if surface_base_pose is None:
        surface_base_pose = transform_pose_at_stamp(
            listener, args.base_frame, surface_camera_pose, args.tf_timeout)

    reference_xyz = list(localization['camera_xyz'])
    reference_xyz[2] += 0.01
    reference_camera_pose = make_camera_point_pose(
        localization['rgb_header'], reference_xyz)
    reference_base_pose = transform_pose_at_stamp(
        listener, args.base_frame, reference_camera_pose, args.tf_timeout)
    surface_base = _pose_position_tuple(surface_base_pose)
    reference_base = _pose_position_tuple(reference_base_pose)
    camera_forward_base = tuple(reference_base[index] - surface_base[index]
                                for index in range(3))

    orientation = current_pose.pose.orientation
    quaternion = quaternion_msg_to_list(orientation)
    tcp_local = tool_axis_vector(args.tool_axis, args.tool_offset)
    tcp_base = rotate_vector_by_quaternion(tcp_local, quaternion)
    alignment_deg = validate_axis_alignment(
        tcp_base, camera_forward_base, args.max_tool_camera_angle_deg)
    contact, precontact = compute_link_targets(
        surface_base, tcp_base, args.approach_gap)
    validate_workspace_points(
        contact, precontact, args.base_min_z, args.base_max_radius)
    grasp_pose = build_absolute_pose(
        args.base_frame, contact[0], contact[1], contact[2], orientation)
    pre_grasp_pose = build_absolute_pose(
        args.base_frame, precontact[0], precontact[1], precontact[2], orientation)

    for label, pose in (('grasp', grasp_pose), ('pre_grasp', pre_grasp_pose)):
        values = _pose_position_tuple(pose) + tuple(quaternion_msg_to_list(
            pose.pose.orientation))
        if any(math.isnan(value) or math.isinf(value) for value in values):
            raise RuntimeError('{} pose contains non-finite values.'.format(label))
    return {
        'surface_camera': surface_camera_pose,
        'surface_base': surface_base_pose,
        'pre_grasp': pre_grasp_pose,
        'grasp': grasp_pose,
        'camera_forward_base': camera_forward_base,
        'tcp_vector_base': tcp_base,
        'alignment_deg': alignment_deg,
    }


def compute_block_context(args, arm):
    """Compute block localization and poses only; never moves the arm or pump."""
    require_block_args(args)
    listener = tf.TransformListener()
    warmup_transform_listener(listener)
    capture = capture_rgbd_once(args)
    localization = localize_block(args, capture)
    current_pose = arm.get_current_pose()
    surface_camera = make_camera_point_pose(
        localization['rgb_header'], localization['camera_xyz'])
    surface_base = transform_pose_at_stamp(
        listener, args.base_frame, surface_camera, args.tf_timeout)
    rospy.loginfo(
        'Block %s (%s) confidence=%.3f center=(%.2f, %.2f) depth=%.4fm',
        localization['target'], localization['class_name'],
        localization['confidence'], localization['center'][0],
        localization['center'][1], localization['depth_m'])
    rospy.loginfo(pose_to_text('block_surface_camera', surface_camera))
    rospy.loginfo(pose_to_text('block_surface_base', surface_base))

    context = {
        'localization': localization,
        'current_pose': current_pose,
        'surface_camera': surface_camera,
        'surface_base': surface_base,
        'pre_grasp': None,
        'grasp': None,
    }
    if args.tool_offset is None and args.dry_run:
        rospy.logwarn('Surface-only dry run: no tool geometry, so grasp poses are omitted.')
        publish_debug_geometry(
            args.base_frame, current_pose, None, None, None,
            extra_pose_topics={'block_surface_base': surface_base})
        return context
    if not is_wrist_forward_reached(
            arm, args.wrist_forward_joint5, args.wrist_forward_tolerance):
        raise RuntimeError(
            'Wrist is not at the verified forward joint5 value; no motion was sent.')
    poses = build_block_poses(
        args, listener, current_pose, localization, surface_camera, surface_base)
    context.update(poses)
    rospy.loginfo(pose_to_text('block_pre_grasp', poses['pre_grasp']))
    rospy.loginfo(pose_to_text('block_grasp', poses['grasp']))
    publish_debug_geometry(
        args.base_frame, current_pose, None, None, None,
        extra_pose_topics={
            'block_surface_base': surface_base,
            'block_pre_grasp': poses['pre_grasp'],
            'block_grasp': poses['grasp'],
        })
    return context


def build_target_pose(base_pose, x_offset, y_offset, z_offset, orientation=None):
    return offset_pose(base_pose, x_offset=x_offset, y_offset=y_offset,
                       z_offset=z_offset, orientation=orientation)


def offset_pose(base_pose, x_offset=0.0, y_offset=0.0, z_offset=0.0, orientation=None):
    target = PoseStamped()
    target.header.frame_id = base_pose.header.frame_id
    target.header.stamp = rospy.Time.now()
    target.pose.position.x = base_pose.pose.position.x + x_offset
    target.pose.position.y = base_pose.pose.position.y + y_offset
    target.pose.position.z = base_pose.pose.position.z + z_offset
    if orientation is None:
        target.pose.orientation = copy.deepcopy(base_pose.pose.orientation)
    else:
        target.pose.orientation = copy.deepcopy(orientation)
    return target


def quaternion_msg_to_list(quaternion_msg):
    return [quaternion_msg.x, quaternion_msg.y, quaternion_msg.z, quaternion_msg.w]


def quaternion_list_to_msg(quaternion_values):
    quaternion_msg = copy.deepcopy(PoseStamped().pose.orientation)
    quaternion_msg.x = quaternion_values[0]
    quaternion_msg.y = quaternion_values[1]
    quaternion_msg.z = quaternion_values[2]
    quaternion_msg.w = quaternion_values[3]
    return quaternion_msg


def offset_pose_in_local_frame(base_pose, local_x=0.0, local_y=0.0, local_z=0.0, orientation=None):
    rotation = tf.transformations.quaternion_matrix(quaternion_msg_to_list(base_pose.pose.orientation))
    world_x = rotation[0][0] * local_x + rotation[0][1] * local_y + rotation[0][2] * local_z
    world_y = rotation[1][0] * local_x + rotation[1][1] * local_y + rotation[1][2] * local_z
    world_z = rotation[2][0] * local_x + rotation[2][1] * local_y + rotation[2][2] * local_z
    return offset_pose(base_pose, x_offset=world_x, y_offset=world_y,
                       z_offset=world_z, orientation=orientation)


def apply_orientation_offset(base_orientation, roll_deg, pitch_deg, yaw_deg):
    base_quaternion = quaternion_msg_to_list(base_orientation)
    offset_quaternion = tf.transformations.quaternion_from_euler(
        math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg))
    result_quaternion = tf.transformations.quaternion_multiply(base_quaternion, offset_quaternion)
    return quaternion_list_to_msg(result_quaternion)


def build_absolute_pose(frame_id, x_value, y_value, z_value, orientation):
    target = PoseStamped()
    target.header.frame_id = frame_id
    target.header.stamp = rospy.Time.now()
    target.pose.position.x = x_value
    target.pose.position.y = y_value
    target.pose.position.z = z_value
    target.pose.orientation = copy.deepcopy(orientation)
    return target


def resolve_pre_x(args):
    if args.pre_x is not None:
        return args.pre_x
    return args.grasp_x - args.approach_gap


def require_place_target(args):
    missing = []
    if args.place_x is None:
        missing.append('--place-x')
    if args.place_y is None:
        missing.append('--place-y')
    if args.place_z is None:
        missing.append('--place-z')

    if missing:
        raise RuntimeError('place / pick_place 模式需要同时提供 {}。'.format(', '.join(missing)))


def pose_to_text(name, pose_stamped):
    pose = pose_stamped.pose
    return ('{} position=({:.4f}, {:.4f}, {:.4f}) orientation=({:.4f}, {:.4f}, {:.4f}, {:.4f})'
            .format(name,
                    pose.position.x, pose.position.y, pose.position.z,
                    pose.orientation.x, pose.orientation.y,
                    pose.orientation.z, pose.orientation.w))


def pose_to_place_args(pose_stamped):
    pose = pose_stamped.pose.position
    return '--place-x {:.4f} --place-y {:.4f} --place-z {:.4f}'.format(
        pose.x, pose.y, pose.z)


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


def publish_debug_geometry(base_frame, current_pose, tag_pose, pre_grasp_pose, grasp_pose,
                           extra_pose_topics=None):
    pose_topics = {}
    if current_pose is not None:
        pose_topics['current_pose'] = copy.deepcopy(current_pose)
    if tag_pose is not None:
        pose_topics['tag_in_base'] = copy.deepcopy(tag_pose)
    if pre_grasp_pose is not None:
        pose_topics['pre_grasp'] = copy.deepcopy(pre_grasp_pose)
    if grasp_pose is not None:
        pose_topics['grasp'] = copy.deepcopy(grasp_pose)
    if extra_pose_topics:
        for name, pose in extra_pose_topics.items():
            pose_topics[name] = copy.deepcopy(pose)

    publishers = {}

    for name in pose_topics:
        publishers[name] = rospy.Publisher('mirobot_pick_debug/{}'.format(name),
                                           PoseStamped, queue_size=1, latch=True)

    marker_pub = rospy.Publisher('mirobot_pick_debug/markers', MarkerArray,
                                 queue_size=1, latch=True)
    rospy.sleep(0.1)

    markers = MarkerArray()
    marker_specs = []
    if current_pose is not None:
        marker_specs.append((0, pose_topics['current_pose'], (0.1, 0.8, 0.1), 0.018))
    if tag_pose is not None:
        marker_specs.append((1, pose_topics['tag_in_base'], (0.1, 0.4, 0.9), 0.018))
    if pre_grasp_pose is not None:
        marker_specs.append((2, pose_topics['pre_grasp'], (0.95, 0.75, 0.1), 0.02))
    if grasp_pose is not None:
        marker_specs.append((3, pose_topics['grasp'], (0.95, 0.2, 0.2), 0.02))
    if extra_pose_topics:
        extra_marker_colors = {
            'pre_place': (0.8, 0.3, 0.95),
            'place': (0.25, 0.95, 0.95),
        }
        marker_id = 4
        for name in extra_pose_topics:
            marker_specs.append((marker_id, pose_topics[name],
                                 extra_marker_colors.get(name, (0.8, 0.8, 0.8)),
                                 0.02))
            marker_id += 1

    for name, pose in pose_topics.items():
        pose.header.frame_id = pose.header.frame_id or base_frame
        pose.header.stamp = rospy.Time.now()
        publishers[name].publish(pose)

    for marker_id, pose, rgb, scale in marker_specs:
        markers.markers.append(create_debug_marker(marker_id, pose, rgb, scale))
    marker_pub.publish(markers)


def execute_pose(arm, target_pose, label):
    rospy.loginfo('Executing %s', pose_to_text(label, target_pose))
    arm.set_start_state_to_current_state()
    arm.set_pose_target(target_pose)
    success = arm.go(wait=True)
    arm.stop()
    arm.clear_pose_targets()
    if not success:
        raise RuntimeError('MoveIt failed during {}.'.format(label))


def execute_cartesian_pose(arm, target_pose, label, eef_step=0.005, jump_threshold=0.0):
    rospy.loginfo('Executing cartesian %s', pose_to_text(label, target_pose))
    arm.set_start_state_to_current_state()
    plan, fraction = arm.compute_cartesian_path([copy.deepcopy(target_pose.pose)],
                                                eef_step, jump_threshold, True)
    if fraction < 0.999:
        raise RuntimeError('MoveIt failed to compute a full cartesian path during {} (fraction={:.3f}).'.format(
            label, fraction))

    if not plan.joint_trajectory.points:
        raise RuntimeError('MoveIt returned an empty cartesian trajectory during {}.'.format(label))

    success = arm.execute(plan, wait=True)
    arm.stop()
    arm.clear_pose_targets()
    if not success:
        raise RuntimeError('MoveIt failed during cartesian {}.'.format(label))


def do_pump_test(pump_proxy, seconds):
    set_pump(pump_proxy, True)
    rospy.sleep(seconds)
    set_pump(pump_proxy, False)


def resolve_target_orientation(args, current_pose, base_pose):
    if args.approach_axis == 'front':
        return apply_orientation_offset(
            base_pose.pose.orientation,
            args.front_tool_roll_deg,
            args.front_tool_pitch_deg,
            args.front_tool_yaw_deg)
    if args.use_tag_orientation:
        return base_pose.pose.orientation
    return current_pose.pose.orientation


def build_grasp_targets(args, base_pose, target_orientation):
    if args.approach_axis in ('z', 'front') and args.pre_x is not None:
        rospy.logwarn('--pre-x 在 approach-axis=%s 模式下不会生效，已忽略。', args.approach_axis)

    if args.approach_axis == 'front':
        grasp_pose = offset_pose_in_local_frame(
            base_pose,
            local_x=args.y_offset,
            local_y=-args.z_offset,
            local_z=args.grasp_x,
            orientation=target_orientation)
        pre_grasp_pose = offset_pose_in_local_frame(
            base_pose,
            local_x=args.y_offset,
            local_y=-args.z_offset,
            local_z=args.grasp_x + args.approach_gap,
            orientation=target_orientation)
        return pre_grasp_pose, grasp_pose

    grasp_pose = build_target_pose(base_pose, args.grasp_x, args.y_offset, args.z_offset,
                                   orientation=target_orientation)

    if args.approach_axis == 'z':
        pre_grasp_pose = offset_pose(grasp_pose, z_offset=args.approach_gap,
                                     orientation=target_orientation)
    else:
        pre_grasp_pose = build_target_pose(base_pose, resolve_pre_x(args), args.y_offset,
                                           args.z_offset, orientation=target_orientation)

    return pre_grasp_pose, grasp_pose


def warn_if_grasp_target_is_too_low(pre_grasp_pose, grasp_pose):
    if pre_grasp_pose.pose.position.z < 0.06 or grasp_pose.pose.position.z < 0.04:
        rospy.logwarn('Computed grasp target is very low: pre_grasp_z=%.4f grasp_z=%.4f. '
                      '这通常说明 --z-offset 太小，或者相机/手眼标定位置已经变了。',
                      pre_grasp_pose.pose.position.z, grasp_pose.pose.position.z)


def build_place_targets(args, base_frame, target_orientation):
    require_place_target(args)
    place_pose = build_absolute_pose(base_frame, args.place_x, args.place_y, args.place_z,
                                     target_orientation)
    pre_place_pose = offset_pose(place_pose, z_offset=args.place_approach_gap,
                                 orientation=target_orientation)
    return pre_place_pose, place_pose


def build_lift_place_targets(args, grasp_pose):
    if args.lift_height <= 0.0:
        raise RuntimeError('--lift-height 必须大于 0。')

    lift_pose = offset_pose(grasp_pose, z_offset=args.lift_height,
                            orientation=grasp_pose.pose.orientation)
    release_pose = offset_pose(grasp_pose, z_offset=0.0,
                               orientation=grasp_pose.pose.orientation)
    return lift_pose, release_pose


def run_grasp_motion(args, arm, pump_proxy, pre_grasp_pose, grasp_pose,
                     retreat_after_grasp=True):
    execute_pose(arm, pre_grasp_pose, 'pre_grasp')
    rospy.sleep(0.5)
    if args.approach_axis in ('z', 'front'):
        execute_cartesian_pose(arm, grasp_pose, 'grasp_descend')
    else:
        execute_pose(arm, grasp_pose, 'grasp')
    rospy.sleep(0.5)
    set_pump(pump_proxy, True)
    rospy.sleep(0.8)

    if not retreat_after_grasp:
        rospy.loginfo('Skipping post-grasp retreat; arm will remain at the grasp pose.')
        return

    if args.approach_axis in ('z', 'front'):
        execute_cartesian_pose(arm, pre_grasp_pose, 'grasp_retreat')
    else:
        execute_pose(arm, pre_grasp_pose, 'retreat')


def run_place_motion(args, arm, pump_proxy, pre_place_pose, place_pose,
                     retreat_after_release=True):
    execute_pose(arm, pre_place_pose, 'pre_place')
    rospy.sleep(0.5)
    execute_cartesian_pose(arm, place_pose, 'place_descend')
    rospy.sleep(0.5)
    set_pump(pump_proxy, False)
    rospy.sleep(0.5)

    if not retreat_after_release:
        rospy.loginfo('Skipping post-release retreat; arm will remain at the release pose.')
        return

    execute_cartesian_pose(arm, pre_place_pose, 'place_retreat')


def compute_grasp_context(args, arm):
    listener = tf.TransformListener()
    tag_frame = 'tag_{}'.format(args.tag_id)
    current_pose = arm.get_current_pose()

    if not args.skip_home and not args.dry_run:
        go_home(arm)
        rospy.sleep(1.0)
        current_pose = arm.get_current_pose()

    if args.wrist_forward:
        if args.dry_run:
            rospy.logwarn('--wrist-forward 与 --dry-run 同时使用时不会实际转动腕部；目标姿态仍按当前机械臂姿态计算。')
        else:
            go_wrist_forward(arm, args.wrist_forward_joint5)
            rospy.sleep(0.5)
            current_pose = arm.get_current_pose()

    rospy.loginfo(pose_to_text('current_pose', current_pose))
    rospy.loginfo('Waiting for tag frame: %s', tag_frame)
    camera_pose = wait_for_tag_pose(listener, args.camera_frame, tag_frame, args.tag_id, args.tf_timeout)
    base_pose = transform_pose(listener, args.base_frame, camera_pose, args.tf_timeout)
    current_pose = arm.get_current_pose()
    target_orientation = resolve_target_orientation(args, current_pose, base_pose)
    pre_grasp_pose, grasp_pose = build_grasp_targets(args, base_pose, target_orientation)
    warn_if_grasp_target_is_too_low(pre_grasp_pose, grasp_pose)
    return current_pose, base_pose, pre_grasp_pose, grasp_pose


def do_grasp(args, arm, pump_proxy):
    current_pose, base_pose, pre_grasp_pose, grasp_pose = compute_grasp_context(args, arm)
    rospy.loginfo(pose_to_text('tag_in_base', base_pose))
    rospy.loginfo(pose_to_text('pre_grasp', pre_grasp_pose))
    rospy.loginfo(pose_to_text('grasp', grasp_pose))
    rospy.loginfo('Using grasp offsets: approach_axis=%s grasp_x=%.4f y_offset=%.4f z_offset=%.4f approach_gap=%.4f',
                  args.approach_axis, args.grasp_x, args.y_offset, args.z_offset, args.approach_gap)
    publish_debug_geometry(args.base_frame, current_pose, base_pose, pre_grasp_pose, grasp_pose)

    if args.dry_run:
        rospy.logwarn('Dry run enabled. No arm motion will be executed.')
        if args.debug_hold_seconds > 0.0:
            rospy.loginfo('Holding debug pose topics for %.1f seconds for RViz inspection.',
                          args.debug_hold_seconds)
            rospy.sleep(args.debug_hold_seconds)
        return

    if args.skip_post_grasp_retreat and not args.skip_home:
        rospy.logwarn('--skip-post-grasp-retreat 已开启，但当前没有 --skip-home，后续仍会执行 home。')

    run_grasp_motion(args, arm, pump_proxy, pre_grasp_pose, grasp_pose,
                     retreat_after_grasp=not args.skip_post_grasp_retreat)

    if not args.skip_home:
        go_home(arm)

    if not args.keep_pump_on:
        rospy.sleep(args.pump_seconds)
        set_pump(pump_proxy, False)


def do_block_grasp(args, arm, pump_proxy):
    """Localize once and execute the guarded front-suction sequence."""
    context = compute_block_context(args, arm)

    # compute_block_context already logs and publishes the exact stamped
    # localization/poses.  Do not publish them a second time here.
    if args.dry_run:
        rospy.logwarn('Dry run: no wrist, pump, or arm motion executed.')
        if args.debug_hold_seconds > 0.0:
            rospy.loginfo(
                'Holding block debug pose topics for %.1f seconds.',
                args.debug_hold_seconds)
            rospy.sleep(args.debug_hold_seconds)
        return 'dry_run'

    pre_grasp_pose = context['pre_grasp']
    grasp_pose = context['grasp']
    if pre_grasp_pose is None or grasp_pose is None:
        raise RuntimeError('Real block motion requires measured tool geometry.')
    if not args.stop_at_pre_grasp and pump_proxy is None:
        raise RuntimeError(
            'Full block grasp requires a confirmed pump service proxy before motion.')

    def move_pre():
        execute_pose(arm, pre_grasp_pose, 'block_pre_grasp')
        rospy.sleep(0.5)

    def confirm_pump_off():
        set_pump(pump_proxy, False)
        rospy.sleep(0.5)

    def move_contact():
        execute_cartesian_pose(arm, grasp_pose, 'block_grasp_contact')
        rospy.sleep(0.5)

    def pump_on():
        set_pump(pump_proxy, True)
        rospy.sleep(0.8)

    def retreat():
        execute_cartesian_pose(arm, pre_grasp_pose, 'block_grasp_retreat')

    return run_block_sequence(
        dry_run=False,
        stop_at_pre_grasp=args.stop_at_pre_grasp,
        confirm_pump_off=confirm_pump_off,
        move_pre=move_pre,
        move_contact=move_contact,
        pump_on=pump_on,
        retreat=retreat,
        log=rospy.logerr)


def do_place(args, arm, pump_proxy):
    current_pose = arm.get_current_pose()
    pre_place_pose, place_pose = build_place_targets(args, args.base_frame,
                                                     current_pose.pose.orientation)

    rospy.loginfo(pose_to_text('current_pose', current_pose))
    rospy.loginfo(pose_to_text('pre_place', pre_place_pose))
    rospy.loginfo(pose_to_text('place', place_pose))
    publish_debug_geometry(args.base_frame, current_pose, None, None, None,
                           extra_pose_topics={
                               'pre_place': pre_place_pose,
                               'place': place_pose,
                           })

    if args.dry_run:
        rospy.logwarn('Dry run enabled. No arm motion will be executed.')
        if args.debug_hold_seconds > 0.0:
            rospy.loginfo('Holding debug pose topics for %.1f seconds for RViz inspection.',
                          args.debug_hold_seconds)
            rospy.sleep(args.debug_hold_seconds)
        return

    run_place_motion(args, arm, pump_proxy, pre_place_pose, place_pose)

    if not args.skip_home:
        go_home(arm)


def do_pick_place(args, arm, pump_proxy):
    current_pose, base_pose, pre_grasp_pose, grasp_pose = compute_grasp_context(args, arm)
    pre_place_pose, place_pose = build_place_targets(args, args.base_frame,
                                                     grasp_pose.pose.orientation)

    if args.skip_post_grasp_retreat:
        rospy.logwarn('--skip-post-grasp-retreat 在 --mode pick_place 下会被忽略，放置前仍会先抬回安全高度。')

    rospy.loginfo(pose_to_text('tag_in_base', base_pose))
    rospy.loginfo(pose_to_text('pre_grasp', pre_grasp_pose))
    rospy.loginfo(pose_to_text('grasp', grasp_pose))
    rospy.loginfo(pose_to_text('pre_place', pre_place_pose))
    rospy.loginfo(pose_to_text('place', place_pose))
    publish_debug_geometry(args.base_frame, current_pose, base_pose, pre_grasp_pose, grasp_pose,
                           extra_pose_topics={
                               'pre_place': pre_place_pose,
                               'place': place_pose,
                           })

    if args.dry_run:
        rospy.logwarn('Dry run enabled. No arm motion will be executed.')
        if args.debug_hold_seconds > 0.0:
            rospy.loginfo('Holding debug pose topics for %.1f seconds for RViz inspection.',
                          args.debug_hold_seconds)
            rospy.sleep(args.debug_hold_seconds)
        return

    run_grasp_motion(args, arm, pump_proxy, pre_grasp_pose, grasp_pose)
    rospy.sleep(0.5)
    run_place_motion(args, arm, pump_proxy, pre_place_pose, place_pose)

    if not args.skip_home:
        go_home(arm)


def do_pick_lift_place(args, arm, pump_proxy):
    current_pose, base_pose, pre_grasp_pose, grasp_pose = compute_grasp_context(args, arm)
    lift_pose, release_pose = build_lift_place_targets(args, grasp_pose)

    rospy.loginfo(pose_to_text('tag_in_base', base_pose))
    rospy.loginfo(pose_to_text('pre_grasp', pre_grasp_pose))
    rospy.loginfo(pose_to_text('grasp', grasp_pose))
    rospy.loginfo(pose_to_text('lift_pose', lift_pose))
    rospy.loginfo(pose_to_text('release_pose', release_pose))
    publish_debug_geometry(args.base_frame, current_pose, base_pose, pre_grasp_pose, grasp_pose,
                           extra_pose_topics={
                               'lift_pose': lift_pose,
                               'release_pose': release_pose,
                           })

    if args.dry_run:
        rospy.logwarn('Dry run enabled. No arm motion will be executed.')
        if args.debug_hold_seconds > 0.0:
            rospy.loginfo('Holding debug pose topics for %.1f seconds for RViz inspection.',
                          args.debug_hold_seconds)
            rospy.sleep(args.debug_hold_seconds)
        return

    run_grasp_motion(args, arm, pump_proxy, pre_grasp_pose, grasp_pose,
                     retreat_after_grasp=False)
    rospy.sleep(0.5)
    run_place_motion(args, arm, pump_proxy, lift_pose, release_pose,
                     retreat_after_release=False)

    if not args.skip_home:
        go_home(arm)


def do_current_pose(args, arm):
    current_pose = arm.get_current_pose()
    rospy.loginfo(pose_to_text('current_pose', current_pose))
    rospy.loginfo('Copy these place args: %s', pose_to_place_args(current_pose))
    print(pose_to_place_args(current_pose))
    publish_debug_geometry(args.base_frame, current_pose, None, None, None)

    if args.debug_hold_seconds > 0.0:
        rospy.loginfo('Holding current_pose debug topic for %.1f seconds for RViz inspection.',
                      args.debug_hold_seconds)
        rospy.sleep(args.debug_hold_seconds)


def main():
    args = parse_args(sys.argv)
    rospy.init_node('mirobot_pick_test', anonymous=False)
    moveit_commander.roscpp_initialize(sys.argv)
    if args.mode == 'block_grasp':
        require_block_args(args)
    else:
        args.tag_id = resolve_tag_id(args)
    if args.mode in ('grasp', 'pick_place', 'pick_lift_place'):
        resolve_grasp_offsets(args)

    arm = None
    pump_proxy = None

    try:
        if args.mode in ('home', 'grasp', 'place', 'pick_place', 'pick_lift_place',
                         'current_pose', 'wrist_forward', 'block_grasp'):
            arm = build_move_group(args.group, args.base_frame, args.velocity_scale,
                                   args.acceleration_scale, args.planning_time,
                                   not args.disable_replanning)

        if (args.mode == 'pump' or
                (args.mode in ('grasp', 'place', 'pick_place', 'pick_lift_place')
                 and not args.dry_run) or
                (args.mode == 'block_grasp' and not args.dry_run
                 and not args.stop_at_pre_grasp)):
            pump_proxy = get_pump_proxy()

        if args.mode == 'home':
            go_home(arm)
        elif args.mode == 'wrist_forward':
            go_wrist_forward(arm, args.wrist_forward_joint5)
        elif args.mode == 'pump':
            do_pump_test(pump_proxy, args.pump_seconds)
        elif args.mode == 'grasp':
            do_grasp(args, arm, pump_proxy)
        elif args.mode == 'place':
            do_place(args, arm, pump_proxy)
        elif args.mode == 'pick_lift_place':
            do_pick_lift_place(args, arm, pump_proxy)
        elif args.mode == 'current_pose':
            do_current_pose(args, arm)
        elif args.mode == 'pick_place':
            do_pick_place(args, arm, pump_proxy)
        elif args.mode == 'block_grasp':
            do_block_grasp(args, arm, pump_proxy)
        else:
            raise RuntimeError('Unsupported mode: {}'.format(args.mode))

        rospy.loginfo('Test finished.')
    except rospy.ROSInterruptException:
        try:
            rospy.logerr(
                'CRITICAL: block grasp may be incomplete; pump state is UNKNOWN. '
                'Stop and recover manually.')
        except Exception:
            pass
        raise
    except Exception as exc:
        rospy.logerr(str(exc))
        raise
    finally:
        moveit_commander.roscpp_shutdown()


if __name__ == '__main__':
    main()
