#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import copy
import json
import math
import os
import sys

if sys.version_info[0] != 2:
    sys.stderr.write(
        'mirobot_delivery.py must run with Python 2 because ROS Melodic '
        'provides MoveIt for Python 2.\n')
    sys.exit(1)

import moveit_commander
import rospy

import mirobot_pick_test_tag as arm_api


PRESET_VERSION = 2
DEFAULT_SEQUENCE = '1,2,3,4'
DEFAULT_DELIVERY_FILE = (
    '/home/eaibot/handeye-calib/config/delivery_presets.json')
DEFAULT_TAG_PRESET_FILE = (
    '/home/eaibot/handeye-calib/config/tag_pick_place_presets.json')
HOME_READY_TIMEOUT_SECONDS = 30.0
HOME_JOINT_TOLERANCE_RAD = 0.08
HOME_STABLE_SAMPLES = 3
DELIVERY_LIFT_METERS = 0.05
COMPUTE_FK_SERVICE = '/compute_fk'
COMPUTE_FK_WAIT_SECONDS = 5.0
TEACH_POINTS = {
    'teach_cargo_pick': ('cargo_pick_joint_values', '载物仓抓取点'),
    'teach_transit': ('transit_joint_values', '中间过渡点'),
    'teach_release': ('delivery_joint_values', '投递点'),
}

try:
    STRING_TYPES = (basestring,)
except NameError:
    STRING_TYPES = (str,)


class UserAbort(Exception):
    pass


def parse_sequence(text):
    if not isinstance(text, STRING_TYPES) or not text.strip():
        raise RuntimeError('--sequence 必须是逗号分隔的正整数 ID。')
    result = []
    for item in text.split(','):
        try:
            value = int(item.strip())
        except (TypeError, ValueError):
            raise RuntimeError('--sequence 必须是逗号分隔的正整数 ID。')
        if value <= 0 or value in result:
            raise RuntimeError('--sequence 中的 ID 必须为正整数且不能重复。')
        result.append(value)
    return result


def positive(value, option):
    value = float(value)
    if math.isnan(value) or math.isinf(value) or value <= 0.0:
        raise RuntimeError('%s 必须为正数。' % option)
    return value


def nonnegative(value, option):
    value = float(value)
    if math.isnan(value) or math.isinf(value) or value < 0.0:
        raise RuntimeError('%s 必须为非负数。' % option)
    return value


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description='示教并执行 Mirobot 载物仓投递动作。')
    parser.add_argument(
        '--mode',
        choices=['teach_cargo_pick', 'teach_transit', 'teach_release',
                 'run_delivery'],
        required=True)
    parser.add_argument('--sequence', default=DEFAULT_SEQUENCE)
    parser.add_argument('--delivery-file', default=DEFAULT_DELIVERY_FILE)
    parser.add_argument('--tag-preset-file', default=DEFAULT_TAG_PRESET_FILE)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--group', default='manipulator')
    parser.add_argument('--base-frame', default='base')
    parser.add_argument('--planning-time', type=float, default=2.0)
    parser.add_argument('--disable-replanning', action='store_true')
    parser.add_argument('--velocity-scale', type=float, default=0.2)
    parser.add_argument('--acceleration-scale', type=float, default=0.2)
    parser.add_argument('--teach-settle-seconds', type=float, default=0.8)
    parser.add_argument('--motion-settle-seconds', type=float, default=0.25)
    parser.add_argument('--pump-on-settle-seconds', type=float, default=1.0)
    parser.add_argument('--pump-off-settle-seconds', type=float, default=0.7)
    parser.add_argument('--startup-home-service',
                        default=arm_api.DEFAULT_STARTUP_HOME_SERVICE)
    parser.add_argument('--startup-home-wait-seconds', type=float, default=8.0)
    parser.add_argument('--startup-home-settle-seconds', type=float, default=3.0)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(rospy.myargv(argv)[1:])
    args.sequence = parse_sequence(args.sequence)
    positive(args.planning_time, '--planning-time')
    positive(args.velocity_scale, '--velocity-scale')
    positive(args.acceleration_scale, '--acceleration-scale')
    nonnegative(args.teach_settle_seconds, '--teach-settle-seconds')
    nonnegative(args.motion_settle_seconds, '--motion-settle-seconds')
    nonnegative(args.pump_on_settle_seconds, '--pump-on-settle-seconds')
    nonnegative(args.pump_off_settle_seconds, '--pump-off-settle-seconds')
    positive(args.startup_home_wait_seconds, '--startup-home-wait-seconds')
    nonnegative(args.startup_home_settle_seconds,
                '--startup-home-settle-seconds')
    return args


def validate_joint_values(values, field):
    if not isinstance(values, list) or len(values) != 6:
        raise RuntimeError('%s 必须包含 6 个关节角。' % field)
    result = []
    for value in values:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            raise RuntimeError('%s 包含非法数值。' % field)
        result.append(number)
    return result


def empty_delivery_preset():
    return {
        'version': PRESET_VERSION,
        'cargo_pick_joint_values_by_id': {},
    }


def load_delivery_preset(path, allow_missing=False):
    if not os.path.isfile(path):
        if allow_missing:
            return empty_delivery_preset()
        raise RuntimeError('投递配置不存在：%s' % path)
    try:
        with open(path, 'r') as handle:
            preset = json.load(handle)
    except (IOError, ValueError) as exc:
        raise RuntimeError('无法读取投递配置：%s' % exc)
    if preset.get('version') != PRESET_VERSION:
        raise RuntimeError('不支持的投递配置版本：%r' % preset.get('version'))
    if not isinstance(preset.get('cargo_pick_joint_values_by_id'), dict):
        raise RuntimeError(
            '投递配置缺少 cargo_pick_joint_values_by_id 对象。')
    return preset


def save_delivery_preset(path, preset):
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


def load_idle_joint_values(path):
    try:
        with open(path, 'r') as handle:
            preset = json.load(handle)
    except (IOError, ValueError) as exc:
        raise RuntimeError('无法读取 Tag preset 中的 idle：%s' % exc)
    return validate_joint_values(
        preset.get('idle_joint_values'), 'idle_joint_values')


def require_delivery_items(preset, sequence):
    transit = validate_joint_values(
        preset.get('transit_joint_values'), 'transit_joint_values')
    release = validate_joint_values(
        preset.get('delivery_joint_values'), 'delivery_joint_values')
    cargo_points = preset.get('cargo_pick_joint_values_by_id', {})
    result = {}
    for item_id in sequence:
        key = str(item_id)
        result[key] = {
            'cargo_pick_joint_values': validate_joint_values(
                cargo_points.get(key),
                'ID%d.cargo_pick_joint_values' % item_id),
            'transit_joint_values': list(transit),
            'delivery_joint_values': list(release),
        }
    return result


def prompt_enter(message):
    print('')
    print(message)
    print('在 RViz Plan/Execute 到位后按 Enter；输入 q 再回车退出。')
    try:
        answer = raw_input().strip().lower()
    except EOFError:
        raise UserAbort('终端输入已关闭。')
    if answer in ('q', 'quit', 'exit'):
        raise UserAbort('用户取消投递示教。')


def record_current_joints(arm, settle_seconds):
    rospy.sleep(settle_seconds)
    return validate_joint_values(
        list(arm.get_current_joint_values()), '当前关节角')


def home_joint_state_is_ready(values, tolerance_rad=HOME_JOINT_TOLERANCE_RAD):
    try:
        joints = validate_joint_values(list(values), '回零关节状态')
    except (RuntimeError, TypeError):
        return False
    return max(abs(value) for value in joints) <= float(tolerance_rad)


def wait_for_home_joint_state(arm):
    deadline = (
        rospy.Time.now() + rospy.Duration(HOME_READY_TIMEOUT_SECONDS))
    stable = 0
    latest = None
    rospy.loginfo(
        '等待启动回零真正完成：需要连续 %d 次关节状态接近零位。',
        HOME_STABLE_SAMPLES)
    while not rospy.is_shutdown() and rospy.Time.now() < deadline:
        try:
            latest = list(arm.get_current_joint_values())
        except Exception:
            latest = None
        if home_joint_state_is_ready(latest or []):
            stable += 1
            if stable >= HOME_STABLE_SAMPLES:
                rospy.loginfo('启动回零已完成，当前关节角：%s', latest)
                arm.set_start_state_to_current_state()
                return
        else:
            stable = 0
        rospy.sleep(0.2)
    raise RuntimeError(
        '启动回零服务已返回，但 %.1fs 内 MoveIt 未收到稳定的零位关节状态。'
        '最新关节角=%s。请检查终端 2 的串口状态和 /joint_states。'
        % (HOME_READY_TIMEOUT_SECONDS, latest))


def build_vertical_offset_pose(pose, offset_z):
    target = copy.deepcopy(pose)
    target.header.stamp = rospy.Time.now()
    target.pose.position.z += float(offset_z)
    return target


def fill_fk_request(request, base_frame, joint_names, joint_values,
                    end_effector_link, stamp):
    request.header.frame_id = base_frame
    request.header.stamp = stamp
    request.fk_link_names = [end_effector_link]
    request.robot_state.joint_state.header.stamp = stamp
    request.robot_state.joint_state.name = list(joint_names)
    request.robot_state.joint_state.position = list(joint_values)
    return request


def compute_fk_pose(args, arm, joint_values):
    try:
        from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest
    except ImportError:
        raise RuntimeError(
            'moveit_msgs/GetPositionFK 不可用，请先 source MoveIt 工作空间。')
    joint_names = list(arm.get_active_joints())
    if len(joint_names) != len(joint_values):
        raise RuntimeError(
            'MoveIt 活动关节数 %d 与投递抓取点关节数 %d 不一致。'
            % (len(joint_names), len(joint_values)))
    end_effector_link = arm.get_end_effector_link()
    if not end_effector_link:
        raise RuntimeError('MoveIt 未配置末端 link，无法计算 FK。')
    rospy.wait_for_service(
        COMPUTE_FK_SERVICE, timeout=COMPUTE_FK_WAIT_SECONDS)
    request = fill_fk_request(
        GetPositionFKRequest(), args.base_frame, joint_names, joint_values,
        end_effector_link, rospy.Time.now())
    response = rospy.ServiceProxy(
        COMPUTE_FK_SERVICE, GetPositionFK)(request)
    if response.error_code.val != 1 or not response.pose_stamped:
        raise RuntimeError(
            'MoveIt FK 计算失败，error_code=%d。'
            % response.error_code.val)
    return response.pose_stamped[0]


def teach_delivery_point(args, arm):
    preset = load_delivery_preset(args.delivery_file, allow_missing=True)
    field, label = TEACH_POINTS[args.mode]
    if args.mode != 'teach_cargo_pick':
        if field in preset and not args.overwrite:
            raise RuntimeError(
                '已有共享%s，重采时请加 --overwrite。' % label)
        prompt_enter(
            '示教全部载物仓共享的%s\n'
            '请把机械臂移到该位置。' % label)
        preset[field] = record_current_joints(
            arm, args.teach_settle_seconds)
        save_delivery_preset(args.delivery_file, preset)
        rospy.loginfo('共享%s已保存：%s', label, preset[field])
        return

    cargo_points = preset['cargo_pick_joint_values_by_id']
    for index, item_id in enumerate(args.sequence, 1):
        key = str(item_id)
        if key in cargo_points and not args.overwrite:
            raise RuntimeError(
                'ID%d 已有%s，重采时请加 --overwrite。'
                % (item_id, label))
        prompt_enter(
            '示教 ID%d（%d/%d）：%s\n'
            '请把机械臂移到该位置。'
            % (item_id, index, len(args.sequence), label))
        cargo_points[key] = record_current_joints(
            arm, args.teach_settle_seconds)
        save_delivery_preset(args.delivery_file, preset)
        rospy.loginfo('ID%d %s已保存：%s', item_id, label,
                      cargo_points[key])
    rospy.loginfo('投递配置已保存：%s', args.delivery_file)


def build_delivery_actions(item, idle_joint_values):
    return [
        ('home', None),
        ('pump', False),
        ('pose_above_cargo', DELIVERY_LIFT_METERS),
        ('joint_to_cargo', item['cargo_pick_joint_values']),
        ('pump', True),
        ('cartesian_lift_z', DELIVERY_LIFT_METERS),
        ('joint', item['transit_joint_values']),
        ('joint', item['delivery_joint_values']),
        ('pump', False),
        ('joint', idle_joint_values),
    ]


def run_delivery(args, arm, pump_proxy):
    preset = load_delivery_preset(args.delivery_file)
    items = require_delivery_items(preset, args.sequence)
    idle_joint_values = load_idle_joint_values(args.tag_preset_file)
    if args.dry_run:
        for item_id in args.sequence:
            rospy.loginfo('Dry-run ID%d 投递动作：%s', item_id,
                          build_delivery_actions(items[str(item_id)],
                                                 idle_joint_values))
        rospy.logwarn('Dry-run：未回零、未移动机械臂、未操作吸泵。')
        return

    for index, item_id in enumerate(args.sequence, 1):
        entry = items[str(item_id)]
        holding_object = False
        rospy.loginfo('开始投递 ID%d（%d/%d）。', item_id, index,
                      len(args.sequence))
        try:
            cargo_pose = compute_fk_pose(
                args, arm, entry['cargo_pick_joint_values'])
            pre_pick_pose = build_vertical_offset_pose(
                cargo_pose, DELIVERY_LIFT_METERS)
            arm_api.run_startup_home(args)
            wait_for_home_joint_state(arm)
            arm_api.set_pump(pump_proxy, False)
            arm_api.execute_pose(
                arm, pre_pick_pose, 'delivery_%d_pre_pick_5cm' % item_id)
            arm_api.execute_joint_values(
                arm, entry['cargo_pick_joint_values'],
                'delivery_%d_cargo_pick' % item_id)
            arm_api.set_pump(pump_proxy, True)
            holding_object = True
            rospy.sleep(args.pump_on_settle_seconds)
            arm_api.execute_cartesian_pose(
                arm, pre_pick_pose, 'delivery_%d_lift_5cm' % item_id)
            arm_api.execute_joint_values(
                arm, entry['transit_joint_values'],
                'delivery_%d_transit' % item_id)
            arm_api.execute_joint_values(
                arm, entry['delivery_joint_values'],
                'delivery_%d_release' % item_id)
            arm_api.set_pump(pump_proxy, False)
            holding_object = False
            rospy.sleep(args.pump_off_settle_seconds)
            arm_api.execute_joint_values(
                arm, idle_joint_values, 'idle')
        except Exception:
            if holding_object:
                rospy.logerr(
                    'ID%d 携带物块时动作失败；为防止物块在未知位置掉落，'
                    '吸泵保持开启，请人工处理。', item_id)
            raise
        rospy.loginfo('ID%d 投递完成，已回到 idle。', item_id)


def main():
    args = parse_args(sys.argv)
    rospy.init_node('mirobot_delivery', anonymous=False)
    moveit_commander.roscpp_initialize(sys.argv)
    arm_api.MOTION_SETTLE_SECONDS = args.motion_settle_seconds
    try:
        arm = arm_api.build_move_group(
            args.group, args.base_frame, args.velocity_scale,
            args.acceleration_scale, args.planning_time,
            not args.disable_replanning)
        if args.mode in TEACH_POINTS:
            teach_delivery_point(args, arm)
        else:
            pump_proxy = None if args.dry_run else arm_api.get_pump_proxy()
            run_delivery(args, arm, pump_proxy)
    except UserAbort as exc:
        rospy.logwarn(str(exc))
    except (KeyboardInterrupt, rospy.ROSInterruptException):
        rospy.logwarn('用户中断。')
    except Exception as exc:
        rospy.logerr(str(exc))
        raise
    finally:
        moveit_commander.roscpp_shutdown()


if __name__ == '__main__':
    main()
