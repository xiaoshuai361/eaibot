#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import json
import math
import os
import subprocess
import sys

import rospy
import tf
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger


DEFAULT_SEQUENCE = '1,2,3,4'
DEFAULT_PRESET_FILE = '/home/eaibot/handeye-calib/config/tag_pick_place_presets.json'
DEFAULT_PICK_SCRIPT = '/home/eaibot/handeye-calib/src/mirobot_pick_test_tag.py'
DEFAULT_TARGET_ROI_RATIO = '0.06,0.00,0.24,1.00'
DEFAULT_STARTUP_HOME_SERVICE = '/mirobot_startup_home'

try:
    STRING_TYPES = (basestring,)
except NameError:
    STRING_TYPES = (str,)


class AlignmentResult(object):
    def __init__(self, linear_x, aligned, center_x, left, right):
        self.linear_x = float(linear_x)
        self.aligned = bool(aligned)
        self.center_x = float(center_x)
        self.left = float(left)
        self.right = float(right)


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


def finite(value, option):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise RuntimeError('%s must be finite.' % option)
    if math.isnan(number) or math.isinf(number):
        raise RuntimeError('%s must be finite.' % option)
    return number


def parse_roi_ratio(text):
    if not isinstance(text, STRING_TYPES):
        raise RuntimeError('target ROI must be x1,y1,x2,y2 ratios.')
    parts = [part.strip() for part in text.split(',')]
    if len(parts) != 4:
        raise RuntimeError('target ROI must be x1,y1,x2,y2 ratios.')
    values = [finite(part, 'target ROI') for part in parts]
    x1, y1, x2, y2 = values
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise RuntimeError('target ROI must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1.')
    return values


def roi_ratio_to_pixels(ratio, image_width, image_height):
    if len(ratio) != 4:
        raise RuntimeError('target ROI must contain four values.')
    width = finite(image_width, 'image_width')
    height = finite(image_height, 'image_height')
    if width <= 0.0 or height <= 0.0:
        raise RuntimeError('image size must be positive.')
    return [
        float(ratio[0]) * width,
        float(ratio[1]) * height,
        float(ratio[2]) * width,
        float(ratio[3]) * height,
    ]


def normalize_box(box):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise RuntimeError('Detection box must contain four values.')
    values = [finite(value, 'Detection box') for value in box]
    if values[2] <= values[0] or values[3] <= values[1]:
        raise RuntimeError('Detection box must have positive width and height.')
    return values


def box_center_x(box):
    values = normalize_box(box)
    return (values[0] + values[2]) / 2.0


def select_detection_for_tag(message, tag_id, min_confidence):
    detections = message.get('detections', []) if isinstance(message, dict) else []
    matches = []
    for detection in detections:
        try:
            detection_tag_id = int(detection.get('tag_id'))
            confidence = finite(detection.get('confidence', 0.0), 'Detection confidence')
            box = normalize_box(detection.get('box'))
        except Exception:
            continue
        if detection_tag_id == int(tag_id) and confidence >= float(min_confidence):
            selected = dict(detection)
            selected['tag_id'] = detection_tag_id
            selected['confidence'] = confidence
            selected['box'] = box
            matches.append(selected)
    if not matches:
        return None
    matches.sort(key=lambda item: item['confidence'], reverse=True)
    return matches[0]


def left_to_right_order(message, allowed_sequence, min_confidence):
    ordered = []
    for tag_id in allowed_sequence:
        detection = select_detection_for_tag(message, tag_id, min_confidence)
        if detection is not None:
            ordered.append((box_center_x(detection['box']), tag_id))
    ordered.sort()
    return [tag_id for _, tag_id in ordered]


def compute_drive_command(detection, roi_pixels, drive_speed,
                          tolerance_px, target_right_forward=True):
    box = normalize_box(detection.get('box'))
    roi = [finite(value, 'target ROI pixels') for value in roi_pixels]
    center_x = box_center_x(box)
    left = roi[0] + float(tolerance_px)
    right = roi[2] - float(tolerance_px)
    speed = abs(float(drive_speed))
    if center_x < left:
        return AlignmentResult(-speed if target_right_forward else speed,
                               False, center_x, left, right)
    if center_x > right:
        return AlignmentResult(speed if target_right_forward else -speed,
                               False, center_x, left, right)
    return AlignmentResult(0.0, True, center_x, left, right)


def build_pick_command(args, tag_id):
    command = [
        args.python2,
        args.pick_script,
        '--mode', 'run_taught_sequence',
        '--sequence', str(int(tag_id)),
        '--preset-file', args.preset_file,
        '--base-frame', args.base_frame,
        '--tf-timeout', str(args.tag_tf_wait_seconds),
        '--velocity-scale', str(args.pick_velocity_scale),
        '--acceleration-scale', str(args.pick_acceleration_scale),
        '--motion-settle-seconds', str(args.pick_motion_settle_seconds),
    ]
    if args.disable_replanning:
        command.append('--disable-replanning')
    return command


def pick_failure_is_missing_tf(output, tag_id):
    marker = 'TF for tag_%d was not found' % int(tag_id)
    return marker in output


def run_pick_command(command, tag_id):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True)
    output_lines = []
    while True:
        line = process.stdout.readline()
        if line:
            output_lines.append(line)
            sys.stdout.write(line)
            if hasattr(sys.stdout, 'flush'):
                sys.stdout.flush()
            continue
        if process.poll() is not None:
            break
    return_code = process.wait()
    output = ''.join(output_lines)
    if return_code == 0:
        return True
    if pick_failure_is_missing_tf(output, tag_id):
        rospy.logwarn(
            'Pick for ID%d skipped because tag TF disappeared before motion.',
            tag_id)
        return False
    raise subprocess.CalledProcessError(return_code, command)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description='Slow chassis alignment helper before taught AprilTag picking.')
    parser.add_argument('--sequence', default=DEFAULT_SEQUENCE)
    parser.add_argument('--order', choices=['left_to_right', 'sequence'],
                        default='left_to_right')
    parser.add_argument('--detections-topic', default='/tag_yolo_quiet/detections_json')
    parser.add_argument('--cmd-vel-topic', default='/cmd_vel')
    parser.add_argument('--debug-image-input-topic', default='/tag_detections_image')
    parser.add_argument('--debug-image-topic', default='/tag_chassis_align/debug_image')
    parser.add_argument('--target-roi-ratio', default=DEFAULT_TARGET_ROI_RATIO)
    parser.add_argument('--drive-speed', type=float, default=0.02)
    parser.add_argument('--align-tolerance-px', type=float, default=12.0)
    parser.add_argument('--stable-frames', type=int, default=2)
    parser.add_argument('--max-align-seconds', type=float, default=25.0)
    parser.add_argument('--control-hz', type=float, default=5.0)
    parser.add_argument('--min-confidence', type=float, default=0.1)
    parser.add_argument('--base-frame', default='base')
    parser.add_argument('--tag-tf-wait-seconds', type=float, default=10.0,
                        help='Wait this long for base->tag_N TF to stabilize before picking.')
    parser.add_argument('--tag-tf-stable-frames', type=int, default=3,
                        help='Required consecutive successful TF reads before picking.')
    parser.add_argument('--target-right-motion', choices=['forward', 'backward'],
                        default='forward')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--align-only', action='store_true')
    parser.add_argument('--wait-key-between-tags', action='store_true',
                        help='After each tag is aligned and handled, wait for Enter before continuing.')
    parser.add_argument('--python2', default=sys.executable)
    parser.add_argument('--pick-script', default=DEFAULT_PICK_SCRIPT)
    parser.add_argument('--preset-file', default=DEFAULT_PRESET_FILE)
    parser.add_argument('--pick-velocity-scale', type=float, default=0.1)
    parser.add_argument('--pick-acceleration-scale', type=float, default=0.1)
    parser.add_argument('--pick-motion-settle-seconds', type=float, default=0.25)
    parser.add_argument('--disable-replanning', action='store_true')
    parser.add_argument('--startup-home-service', default=DEFAULT_STARTUP_HOME_SERVICE,
                        help='Trigger service that sends the same $H homing command as mirobot.launch startup.')
    parser.add_argument('--startup-home-wait-seconds', type=float, default=8.0)
    parser.add_argument('--startup-home-settle-seconds', type=float, default=3.0,
                        help='Wait after startup homing before the next chassis/tag step.')
    parser.add_argument('--skip-startup-home', action='store_true',
                        help='Skip controller startup homing after each successful pick.')
    args = parser.parse_args(rospy.myargv(argv)[1:])
    args.sequence = parse_sequence(args.sequence)
    args.target_roi_ratio = parse_roi_ratio(args.target_roi_ratio)
    if args.drive_speed <= 0.0:
        raise RuntimeError('--drive-speed must be positive.')
    if args.align_tolerance_px < 0.0:
        raise RuntimeError('--align-tolerance-px must be non-negative.')
    if args.stable_frames <= 0:
        raise RuntimeError('--stable-frames must be positive.')
    if args.max_align_seconds <= 0.0:
        raise RuntimeError('--max-align-seconds must be positive.')
    if args.control_hz <= 0.0:
        raise RuntimeError('--control-hz must be positive.')
    if args.tag_tf_wait_seconds < 0.0:
        raise RuntimeError('--tag-tf-wait-seconds must be non-negative.')
    if args.tag_tf_stable_frames <= 0:
        raise RuntimeError('--tag-tf-stable-frames must be positive.')
    if args.startup_home_wait_seconds <= 0.0:
        raise RuntimeError('--startup-home-wait-seconds must be positive.')
    if args.startup_home_settle_seconds < 0.0:
        raise RuntimeError('--startup-home-settle-seconds must be non-negative.')
    return args


def make_twist(linear_x):
    message = Twist()
    message.linear.x = float(linear_x)
    message.linear.y = message.linear.z = 0.0
    message.angular.x = message.angular.y = message.angular.z = 0.0
    return message


class ChassisAlignPickSequence(object):
    def __init__(self, args):
        self.args = args
        self.latest_detections = None
        self.bridge = CvBridge()
        self.cmd_pub = rospy.Publisher(args.cmd_vel_topic, Twist, queue_size=1)
        self.debug_image_pub = rospy.Publisher(args.debug_image_topic, Image, queue_size=1)
        self.detections_sub = rospy.Subscriber(
            args.detections_topic, String, self.detections_callback, queue_size=1)
        self.image_sub = rospy.Subscriber(
            args.debug_image_input_topic, Image, self.image_callback, queue_size=1)
        self.tf_listener = None
        if not args.align_only and not args.dry_run:
            self.tf_listener = tf.TransformListener()

    def detections_callback(self, message):
        try:
            self.latest_detections = json.loads(message.data)
        except ValueError as exc:
            rospy.logwarn_throttle(2.0, 'Could not parse YOLO detections JSON: %s', exc)

    def image_callback(self, message):
        if self.latest_detections is None:
            return
        try:
            import cv2
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
            height, width = image.shape[:2]
            roi = roi_ratio_to_pixels(self.args.target_roi_ratio, width, height)
            x1, y1, x2, y2 = [int(round(value)) for value in roi]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            for detection in self.latest_detections.get('detections', []):
                box = normalize_box(detection.get('box'))
                bx1, by1, bx2, by2 = [int(round(value)) for value in box]
                cx = int(round((box[0] + box[2]) / 2.0))
                cy = int(round((box[1] + box[3]) / 2.0))
                cv2.rectangle(image, (bx1, by1), (bx2, by2), (255, 0, 0), 1)
                cv2.circle(image, (cx, cy), 3, (0, 255, 255), -1)
                cv2.putText(image, 'ID%d' % int(detection.get('tag_id', 0)),
                            (bx1, max(15, by1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            output = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
            output.header = message.header
            self.debug_image_pub.publish(output)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, 'Could not draw chassis alignment debug image: %s', exc)

    def publish_velocity(self, linear_x):
        if self.args.dry_run:
            return
        self.cmd_pub.publish(make_twist(linear_x))

    def stop_chassis(self):
        for _ in range(5):
            self.cmd_pub.publish(make_twist(0.0))
            rospy.sleep(0.03)

    def wait_for_detections(self, timeout):
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        rate = rospy.Rate(self.args.control_hz)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            if self.latest_detections is not None:
                return self.latest_detections
            rate.sleep()
        raise RuntimeError('Timed out waiting for YOLO detections JSON.')

    def select_next_tag(self, remaining_tags):
        remaining_tags = list(remaining_tags)
        if not remaining_tags:
            raise RuntimeError('No remaining tag IDs to process.')
        if self.args.order == 'sequence':
            return remaining_tags[0]
        deadline = rospy.Time.now() + rospy.Duration(self.args.max_align_seconds)
        rate = rospy.Rate(self.args.control_hz)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            try:
                message = self.wait_for_detections(0.3)
            except RuntimeError:
                rate.sleep()
                continue
            order = left_to_right_order(
                message, remaining_tags, self.args.min_confidence)
            if order:
                rospy.loginfo(
                    'Visible remaining tags left-to-right: %s. Next ID%d.',
                    order, order[0])
                return order[0]
            rospy.logwarn_throttle(
                2.0,
                'Waiting for at least one remaining requested tag. remaining=%s',
                remaining_tags)
            rate.sleep()
        raise RuntimeError(
            'No remaining requested tag IDs are visible for left-to-right ordering: %s.'
            % remaining_tags)

    def align_tag(self, tag_id):
        stable = 0
        deadline = rospy.Time.now() + rospy.Duration(self.args.max_align_seconds)
        rate = rospy.Rate(self.args.control_hz)
        target_right_forward = self.args.target_right_motion == 'forward'
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            message = self.latest_detections
            if message is None:
                self.publish_velocity(0.0)
                rate.sleep()
                continue
            detection = select_detection_for_tag(
                message, tag_id, self.args.min_confidence)
            if detection is None:
                stable = 0
                self.publish_velocity(0.0)
                rospy.logwarn_throttle(
                    2.0, 'Waiting for YOLO ID%d before chassis alignment.', tag_id)
                rate.sleep()
                continue
            roi = roi_ratio_to_pixels(
                self.args.target_roi_ratio,
                message.get('image_width'),
                message.get('image_height'))
            result = compute_drive_command(
                detection, roi, self.args.drive_speed,
                self.args.align_tolerance_px, target_right_forward)
            self.publish_velocity(result.linear_x)
            if result.aligned:
                stable += 1
                if stable >= self.args.stable_frames:
                    self.stop_chassis()
                    rospy.loginfo(
                        'ID%d aligned in target ROI. center_x=%.1f range=[%.1f, %.1f]',
                        tag_id, result.center_x, result.left, result.right)
                    return
            else:
                stable = 0
            rate.sleep()
        self.stop_chassis()
        raise RuntimeError('Timed out aligning ID%d into target ROI.' % tag_id)

    def run_pick(self, tag_id):
        if self.args.align_only:
            rospy.logwarn('Align-only enabled. Pick for ID%d is skipped.', tag_id)
            return
        if self.args.dry_run:
            rospy.logwarn('Dry run enabled. Pick for ID%d is skipped.', tag_id)
            return
        command = build_pick_command(self.args, tag_id)
        rospy.loginfo('Starting taught pick for ID%d.', tag_id)
        return run_pick_command(command, tag_id)

    def run_startup_home(self, tag_id):
        if (self.args.align_only or self.args.dry_run or
                getattr(self.args, 'skip_startup_home', False)):
            return
        rospy.loginfo(
            'Running startup homing after ID%d through %s.',
            tag_id, self.args.startup_home_service)
        rospy.wait_for_service(
            self.args.startup_home_service,
            timeout=self.args.startup_home_wait_seconds)
        response = rospy.ServiceProxy(self.args.startup_home_service, Trigger)()
        if not response.success:
            raise RuntimeError(
                'Startup homing service failed after ID%d: %s'
                % (tag_id, response.message))
        rospy.sleep(self.args.startup_home_settle_seconds)

    def tag_tf_is_available(self, tag_id):
        if self.tf_listener is None:
            self.tf_listener = tf.TransformListener()
        tag_frame = 'tag_%d' % int(tag_id)
        try:
            now = rospy.Time(0)
            self.tf_listener.waitForTransform(
                self.args.base_frame, tag_frame, now, rospy.Duration(0.1))
            self.tf_listener.lookupTransform(self.args.base_frame, tag_frame, now)
            return True
        except (tf.Exception, tf.LookupException, tf.ConnectivityException,
                tf.ExtrapolationException):
            return False

    def wait_for_tag_tf_before_pick(self, tag_id):
        if self.args.align_only or self.args.dry_run:
            return True
        if self.args.tag_tf_wait_seconds <= 0.0:
            return True
        stable = 0
        deadline = rospy.Time.now() + rospy.Duration(self.args.tag_tf_wait_seconds)
        rate = rospy.Rate(self.args.control_hz)
        rospy.loginfo(
            'Waiting up to %.1fs for tag_%d TF to stabilize before pick.',
            self.args.tag_tf_wait_seconds, tag_id)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            if self.tag_tf_is_available(tag_id):
                stable += 1
                if stable >= self.args.tag_tf_stable_frames:
                    rospy.loginfo(
                        'tag_%d TF is stable for %d consecutive reads.',
                        tag_id, stable)
                    return True
            else:
                stable = 0
            rate.sleep()
        self.stop_chassis()
        rospy.logwarn(
            'Skipping ID%d because tag_%d TF did not stabilize within %.1fs.',
            tag_id, tag_id, self.args.tag_tf_wait_seconds)
        return False

    def wait_between_tags(self, tag_id, index, total):
        self.stop_chassis()
        print('')
        print('ID%d 已完成对准/处理（%d/%d）。' % (tag_id, index, total))
        print('请确认现场状态，把下一个 tag 准备好后按 Enter 继续；输入 q 再回车退出。')
        if hasattr(sys.stdout, 'flush'):
            sys.stdout.flush()
        line = sys.stdin.readline()
        if line.strip().lower() in ('q', 'quit', 'exit'):
            raise RuntimeError('User aborted after ID%d.' % tag_id)

    def run(self):
        try:
            remaining_tags = list(self.args.sequence)
            total = len(remaining_tags)
            processed = 0
            while remaining_tags:
                tag_id = self.select_next_tag(remaining_tags)
                processed += 1
                rospy.loginfo('Aligning ID%d into target ROI.', tag_id)
                self.align_tag(tag_id)
                needs_pick_tf = not self.args.align_only and not self.args.dry_run
                if needs_pick_tf and not self.wait_for_tag_tf_before_pick(tag_id):
                    remaining_tags.remove(tag_id)
                    if self.args.wait_key_between_tags and processed < total:
                        self.wait_between_tags(tag_id, processed, total)
                    continue
                pick_completed = self.run_pick(tag_id)
                if pick_completed is False:
                    remaining_tags.remove(tag_id)
                    if self.args.wait_key_between_tags and processed < total:
                        self.wait_between_tags(tag_id, processed, total)
                    continue
                self.run_startup_home(tag_id)
                remaining_tags.remove(tag_id)
                if self.args.wait_key_between_tags and processed < total:
                    self.wait_between_tags(tag_id, processed, total)
        finally:
            self.stop_chassis()


def main(argv=None):
    argv = sys.argv if argv is None else argv
    rospy.init_node('tag_chassis_align_pick_sequence', anonymous=False)
    args = parse_args(argv)
    node = ChassisAlignPickSequence(args)
    node.run()


if __name__ == '__main__':
    main()
