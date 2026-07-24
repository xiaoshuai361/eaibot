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
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import String


DEFAULT_SEQUENCE = '1,2,3,4'
DEFAULT_PRESET_FILE = '/home/eaibot/handeye-calib/config/tag_pick_place_presets.json'
DEFAULT_PICK_SCRIPT = '/home/eaibot/handeye-calib/src/mirobot_pick_test_tag.py'
DEFAULT_TARGET_ROI_RATIO = '0.06,0.00,0.24,1.00'

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
        '--velocity-scale', str(args.pick_velocity_scale),
        '--acceleration-scale', str(args.pick_acceleration_scale),
        '--motion-settle-seconds', str(args.pick_motion_settle_seconds),
        '--home-after-idle',
    ]
    if args.disable_replanning:
        command.append('--disable-replanning')
    return command


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
    parser.add_argument('--target-right-motion', choices=['forward', 'backward'],
                        default='forward')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--align-only', action='store_true')
    parser.add_argument('--python2', default=sys.executable)
    parser.add_argument('--pick-script', default=DEFAULT_PICK_SCRIPT)
    parser.add_argument('--preset-file', default=DEFAULT_PRESET_FILE)
    parser.add_argument('--pick-velocity-scale', type=float, default=0.1)
    parser.add_argument('--pick-acceleration-scale', type=float, default=0.1)
    parser.add_argument('--pick-motion-settle-seconds', type=float, default=0.25)
    parser.add_argument('--disable-replanning', action='store_true')
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

    def resolve_order(self):
        if self.args.order == 'sequence':
            return list(self.args.sequence)
        message = self.wait_for_detections(self.args.max_align_seconds)
        order = left_to_right_order(
            message, self.args.sequence, self.args.min_confidence)
        if not order:
            raise RuntimeError('No requested tag IDs are visible for left-to-right ordering.')
        rospy.loginfo('Left-to-right tag order: %s', order)
        return order

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
        subprocess.check_call(command)

    def run(self):
        try:
            order = self.resolve_order()
            for tag_id in order:
                rospy.loginfo('Aligning ID%d into target ROI.', tag_id)
                self.align_tag(tag_id)
                self.run_pick(tag_id)
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
