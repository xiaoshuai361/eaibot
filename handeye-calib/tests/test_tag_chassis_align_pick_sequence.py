import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "src" / "tag_chassis_align_pick_sequence.py"


def load_symbols(*names):
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.Assign, ast.ClassDef))
    ]
    class FakeDuration:
        def __init__(self, seconds):
            self.seconds = float(seconds)

    class FakeTime:
        def __init__(self, seconds=0.0):
            self.seconds = float(seconds)

        @staticmethod
        def now():
            FakeRospy._now += 0.1
            return FakeTime(FakeRospy._now)

        def __add__(self, duration):
            return FakeTime(self.seconds + duration.seconds)

        def __lt__(self, other):
            return self.seconds < other.seconds

    class FakeRate:
        def __init__(self, hz):
            self.hz = hz

        def sleep(self):
            FakeRospy._now += 0.1

    class FakeRospy:
        _now = 0.0
        Time = FakeTime
        Duration = FakeDuration
        Rate = FakeRate

        @staticmethod
        def loginfo(*args, **kwargs):
            pass

        @staticmethod
        def logwarn(*args, **kwargs):
            pass

        @staticmethod
        def logwarn_throttle(*args, **kwargs):
            pass

        @staticmethod
        def is_shutdown():
            return False

        @staticmethod
        def myargv(argv):
            return argv

    namespace = {
        "argparse": __import__("argparse"),
        "json": json,
        "math": __import__("math"),
        "os": __import__("os"),
        "subprocess": __import__("subprocess"),
        "sys": __import__("sys"),
        "time": __import__("time"),
        "rospy": FakeRospy,
        "Trigger": object,
        "STRING_TYPES": (str,),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return [namespace[name] for name in names]


def test_parse_roi_ratio_and_convert_to_pixels():
    parse_roi_ratio, roi_ratio_to_pixels = load_symbols(
        "parse_roi_ratio", "roi_ratio_to_pixels")

    ratio = parse_roi_ratio("0.06,0.00,0.24,1.00")
    assert ratio == pytest.approx([0.06, 0.0, 0.24, 1.0])
    assert roi_ratio_to_pixels(ratio, image_width=640, image_height=480) == pytest.approx(
        [38.4, 0.0, 153.6, 480.0])

    with pytest.raises(RuntimeError, match="target ROI"):
        parse_roi_ratio("0.2,0,0.1,1")


def test_select_detection_for_tag_uses_id_and_confidence():
    select_detection_for_tag, = load_symbols("select_detection_for_tag")
    message = {
        "image_width": 640,
        "image_height": 480,
        "detections": [
            {"tag_id": 2, "confidence": 0.2, "box": [200, 10, 240, 50]},
            {"tag_id": 4, "confidence": 0.9, "box": [80, 10, 120, 50]},
        ],
    }

    selected = select_detection_for_tag(message, tag_id=4, min_confidence=0.5)

    assert selected["tag_id"] == 4
    assert selected["box"] == [80.0, 10.0, 120.0, 50.0]
    assert select_detection_for_tag(message, tag_id=2, min_confidence=0.5) is None


def test_compute_drive_command_moves_forward_when_target_is_right_of_roi():
    compute_drive_command, = load_symbols("compute_drive_command")
    roi = [40.0, 0.0, 160.0, 480.0]

    right = compute_drive_command(
        detection={"box": [220.0, 10.0, 260.0, 50.0]},
        roi_pixels=roi,
        drive_speed=0.02,
        tolerance_px=12.0,
        target_right_forward=True)
    left = compute_drive_command(
        detection={"box": [0.0, 10.0, 20.0, 50.0]},
        roi_pixels=roi,
        drive_speed=0.02,
        tolerance_px=12.0,
        target_right_forward=True)
    inside = compute_drive_command(
        detection={"box": [80.0, 10.0, 120.0, 50.0]},
        roi_pixels=roi,
        drive_speed=0.02,
        tolerance_px=12.0,
        target_right_forward=True)

    assert right.linear_x == pytest.approx(0.02)
    assert right.aligned is False
    assert left.linear_x == pytest.approx(-0.02)
    assert inside.linear_x == pytest.approx(0.0)
    assert inside.aligned is True


def test_alignment_stability_counts_only_new_yolo_inferences():
    update_stability, = load_symbols("update_alignment_stability")
    first = {
        "refresh_yolo": True,
        "inference_seq": 10,
        "inference_stamp": {"secs": 100, "nsecs": 0},
    }
    cached = {
        "refresh_yolo": False,
        "inference_seq": 10,
        "inference_stamp": {"secs": 100, "nsecs": 0},
    }
    second = {
        "refresh_yolo": True,
        "inference_seq": 11,
        "inference_stamp": {"secs": 101, "nsecs": 0},
    }

    stable, key, accepted = update_stability(0, None, first, aligned=True)
    assert (stable, accepted) == (1, True)
    stable, key, accepted = update_stability(stable, key, cached, aligned=True)
    assert (stable, accepted) == (1, False)
    stable, key, accepted = update_stability(stable, key, second, aligned=True)
    assert (stable, accepted) == (2, True)


def test_left_to_right_order_sorts_available_tag_ids_by_x_center():
    left_to_right_order, = load_symbols("left_to_right_order")
    message = {
        "detections": [
            {"tag_id": 4, "confidence": 0.8, "box": [300, 0, 340, 40]},
            {"tag_id": 1, "confidence": 0.8, "box": [10, 0, 50, 40]},
            {"tag_id": 2, "confidence": 0.8, "box": [160, 0, 200, 40]},
        ],
    }

    assert left_to_right_order(message, allowed_sequence=[1, 2, 3, 4],
                               min_confidence=0.5) == [1, 2, 4]


def test_left_to_right_selects_next_visible_tag_without_waiting_for_all_requested_tags():
    ChassisAlignPickSequence, = load_symbols("ChassisAlignPickSequence")

    partial = {
        "detections": [
            {"tag_id": 1, "confidence": 0.8, "box": [10, 0, 50, 40]},
            {"tag_id": 2, "confidence": 0.8, "box": [80, 0, 120, 40]},
            {"tag_id": 3, "confidence": 0.8, "box": [160, 0, 200, 40]},
        ],
    }

    class FakeSequence(ChassisAlignPickSequence):
        def __init__(self):
            self.args = SimpleNamespace(
                order="left_to_right",
                sequence=[1, 2, 3, 4],
                min_confidence=0.5,
                max_align_seconds=1.0,
                control_hz=5.0,
            )
            self.messages = [partial]

        def wait_for_detections(self, timeout):
            return self.messages.pop(0)

    assert FakeSequence().select_next_tag([1, 2, 3, 4]) == 1


def test_pick_command_leaves_startup_homing_to_chassis_sequence():
    build_pick_command, = load_symbols("build_pick_command")
    args = SimpleNamespace(
        python2="/usr/bin/python2",
        pick_script="/home/eaibot/handeye-calib/src/mirobot_pick_test_tag.py",
        preset_file="/home/eaibot/handeye-calib/config/tag_pick_place_presets.json",
        base_frame="base",
        tag_tf_wait_seconds=10.0,
        pick_velocity_scale=0.1,
        pick_acceleration_scale=0.2,
        pick_motion_settle_seconds=0.4,
        disable_replanning=True,
    )

    command = build_pick_command(args, tag_id=4)

    assert command[:3] == [
        "/usr/bin/python2",
        "/home/eaibot/handeye-calib/src/mirobot_pick_test_tag.py",
        "--mode",
    ]
    assert "--home-after-idle" not in command
    assert "--disable-replanning" in command
    assert command[command.index("--sequence") + 1] == "4"
    assert command[command.index("--base-frame") + 1] == "base"
    assert command[command.index("--tf-timeout") + 1] == "10.0"
    assert command[command.index("--velocity-scale") + 1] == "0.1"
    assert command[command.index("--acceleration-scale") + 1] == "0.2"


def test_pick_failure_is_missing_tf_detects_child_tf_error():
    pick_failure_is_missing_tf, = load_symbols("pick_failure_is_missing_tf")

    assert pick_failure_is_missing_tf(
        "RuntimeError: TF for tag_2 was not found.", 2) is True
    assert pick_failure_is_missing_tf(
        "RuntimeError: MoveIt failed during taught_grasp.", 2) is False


def test_parse_args_accepts_wait_key_and_tag_tf_wait_option():
    parse_args, = load_symbols("parse_args")

    args = parse_args([
        "tag_chassis_align_pick_sequence.py",
        "--sequence", "1,2",
        "--wait-key-between-tags",
        "--tag-tf-wait-seconds", "10",
        "--base-frame", "base",
        "--startup-home-service", "/mirobot_startup_home",
        "--skip-startup-home",
        "--max-detection-age-seconds", "1.5",
        "--chassis-settle-seconds", "0.8",
    ])

    assert args.sequence == [1, 2]
    assert args.wait_key_between_tags is True
    assert args.tag_tf_wait_seconds == 10.0
    assert args.base_frame == "base"
    assert args.startup_home_service == "/mirobot_startup_home"
    assert args.skip_startup_home is True
    assert args.max_detection_age_seconds == pytest.approx(1.5)
    assert args.chassis_settle_seconds == pytest.approx(0.8)


def test_parse_args_defaults_match_competition_short_command():
    parse_args, = load_symbols("parse_args")

    args = parse_args(["tag_chassis_align_pick_sequence.py"])

    assert args.sequence == [1, 2, 3, 4]
    assert args.order == "left_to_right"
    assert args.preset_file == "/home/eaibot/handeye-calib/config/tag_pick_place_presets.json"
    assert args.target_roi_ratio == pytest.approx([0.06, 0.0, 0.24, 1.0])
    assert args.drive_speed == pytest.approx(0.012)
    assert args.align_tolerance_px == pytest.approx(12.0)
    assert args.stable_frames == 4
    assert args.max_detection_age_seconds == pytest.approx(4.0)
    assert args.chassis_settle_seconds == pytest.approx(0.8)
    assert args.tag_tf_wait_seconds == pytest.approx(10.0)
    assert args.startup_home_settle_seconds == pytest.approx(3.0)
    assert args.pick_velocity_scale == pytest.approx(0.2)
    assert args.pick_acceleration_scale == pytest.approx(0.2)


def test_tf_gate_ignores_cached_stamp_and_accepts_next_new_tf():
    ChassisAlignPickSequence, = load_symbols("ChassisAlignPickSequence")
    events = []

    class FakeSequence(ChassisAlignPickSequence):
        def __init__(self):
            self.args = SimpleNamespace(
                align_only=False,
                dry_run=False,
                tag_tf_wait_seconds=1.0,
                control_hz=5.0,
            )
            self.samples = [
                100,
                100,
                101,
            ]

        def read_tag_tf_stamp(self, tag_id):
            events.append(("read", tag_id))
            return self.samples.pop(0)

        def stop_chassis(self):
            events.append(("stop",))

    assert FakeSequence().wait_for_tag_tf_before_pick(3) is True
    assert events == [("read", 3), ("read", 3), ("read", 3)]


def test_tf_gate_times_out_when_only_cached_tf_is_available():
    ChassisAlignPickSequence, = load_symbols("ChassisAlignPickSequence")
    events = []

    class FakeSequence(ChassisAlignPickSequence):
        def __init__(self):
            self.args = SimpleNamespace(
                align_only=False,
                dry_run=False,
                tag_tf_wait_seconds=0.3,
                control_hz=5.0,
            )

        def read_tag_tf_stamp(self, tag_id):
            return 100

        def stop_chassis(self):
            events.append(("stop",))

    assert FakeSequence().wait_for_tag_tf_before_pick(3) is False
    assert events == [("stop",)]


def test_align_tag_gives_confirmation_a_full_alignment_timeout():
    ChassisAlignPickSequence, AlignmentResult = load_symbols(
        "ChassisAlignPickSequence", "AlignmentResult")
    events = []

    class FakeDuration:
        def __init__(self, seconds):
            self.seconds = float(seconds)

    class FakeTime:
        def __init__(self, seconds=0.0):
            self.seconds = float(seconds)

        @staticmethod
        def now():
            FakeRospy._now += 0.1
            return FakeTime(FakeRospy._now)

        def __add__(self, duration):
            return FakeTime(self.seconds + duration.seconds)

        def __lt__(self, other):
            return self.seconds < other.seconds

    class FakeRate:
        def __init__(self, hz):
            self.hz = hz

        def sleep(self):
            FakeRospy._now += 0.1

    class FakeRospy:
        _now = 0.0
        Time = FakeTime
        Duration = FakeDuration
        Rate = FakeRate

        @staticmethod
        def is_shutdown():
            return False

        @staticmethod
        def sleep(seconds):
            FakeRospy._now += float(seconds)

        @staticmethod
        def loginfo(*items):
            events.append("info")

        @staticmethod
        def logwarn(*items):
            events.append("warn")

        @staticmethod
        def logwarn_throttle(*items):
            events.append("warn_throttle")

    class FakeSequence(ChassisAlignPickSequence):
        def __init__(self):
            self.args = SimpleNamespace(
                max_align_seconds=0.3,
                control_hz=5.0,
                target_right_motion="forward",
                max_detection_age_seconds=4.0,
                min_confidence=0.1,
                target_roi_ratio=[0.0, 0.0, 1.0, 1.0],
                drive_speed=0.012,
                align_tolerance_px=12.0,
                stable_frames=1,
                chassis_settle_seconds=0.8,
            )
            self.latest_detections = {"image_width": 640, "image_height": 480}

        def publish_velocity(self, linear_x):
            events.append(("velocity", linear_x))

        def stop_chassis(self):
            events.append("stop")

    ChassisAlignPickSequence.align_tag.__globals__.update({
        "rospy": FakeRospy,
        "inference_age_seconds": lambda message, now: 0.0,
        "ros_time_to_seconds": lambda stamp: stamp.seconds,
        "select_detection_for_tag": lambda message, tag_id, min_confidence: {"box": [50, 0, 80, 20]},
        "roi_ratio_to_pixels": lambda ratio, width, height: [0.0, 0.0, 640.0, 480.0],
        "compute_drive_command": lambda *items: AlignmentResult(0.0, True, 65.0, 12.0, 628.0),
        "update_alignment_stability": lambda stable, key, message, aligned: (
            stable + 1, object(), True),
    })

    FakeSequence().align_tag(4)

    assert events.count("stop") >= 2
    assert "info" in events


def test_run_skips_alignment_timeout_and_continues_with_next_tag():
    ChassisAlignPickSequence, AlignmentTimeout = load_symbols(
        "ChassisAlignPickSequence", "AlignmentTimeout")
    calls = []

    class FakeSequence(ChassisAlignPickSequence):
        def __init__(self):
            self.args = SimpleNamespace(
                sequence=[1, 2],
                align_only=True,
                dry_run=False,
                wait_key_between_tags=False,
            )

        def select_next_tag(self, remaining_tags):
            return remaining_tags[0]

        def align_tag(self, tag_id):
            calls.append(("align", tag_id))
            if tag_id == 1:
                raise AlignmentTimeout("ID1 timed out")

        def run_pick(self, tag_id):
            calls.append(("pick", tag_id))

        def run_startup_home(self, tag_id):
            calls.append(("home", tag_id))

        def stop_chassis(self):
            calls.append(("stop",))

    FakeSequence().run()

    assert calls == [
        ("align", 1),
        ("stop",),
        ("align", 2),
        ("pick", 2),
        ("home", 2),
        ("stop",),
    ]


def test_run_calls_controller_startup_home_after_each_successful_pick():
    ChassisAlignPickSequence, = load_symbols("ChassisAlignPickSequence")
    calls = []

    class FakeSequence(ChassisAlignPickSequence):
        def __init__(self):
            self.args = SimpleNamespace(
                sequence=[1, 2],
                align_only=False,
                dry_run=False,
                wait_key_between_tags=False,
            )

        def select_next_tag(self, remaining_tags):
            return remaining_tags[0]

        def align_tag(self, tag_id):
            calls.append(("align", tag_id))

        def wait_for_tag_tf_before_pick(self, tag_id):
            calls.append(("tf", tag_id))
            return True

        def run_pick(self, tag_id):
            calls.append(("pick", tag_id))

        def run_startup_home(self, tag_id):
            calls.append(("startup_home", tag_id))

        def stop_chassis(self):
            calls.append(("stop",))

    FakeSequence().run()

    assert calls == [
        ("align", 1),
        ("tf", 1),
        ("pick", 1),
        ("startup_home", 1),
        ("align", 2),
        ("tf", 2),
        ("pick", 2),
        ("startup_home", 2),
        ("stop",),
    ]


def test_run_waits_for_key_between_tags_when_enabled():
    ChassisAlignPickSequence, = load_symbols("ChassisAlignPickSequence")
    calls = []

    class FakeSequence(ChassisAlignPickSequence):
        def __init__(self):
            self.args = SimpleNamespace(
                sequence=[1, 2, 3],
                align_only=False,
                dry_run=False,
                wait_key_between_tags=True,
                skip_startup_home=True,
            )

        def select_next_tag(self, remaining_tags):
            return remaining_tags[0]

        def align_tag(self, tag_id):
            calls.append(("align", tag_id))

        def run_pick(self, tag_id):
            calls.append(("pick", tag_id))

        def wait_for_tag_tf_before_pick(self, tag_id):
            return True

        def wait_between_tags(self, tag_id, index, total):
            calls.append(("wait", tag_id, index, total))

        def stop_chassis(self):
            calls.append(("stop",))

    FakeSequence().run()

    assert calls == [
        ("align", 1),
        ("pick", 1),
        ("wait", 1, 1, 3),
        ("align", 2),
        ("pick", 2),
        ("wait", 2, 2, 3),
        ("align", 3),
        ("pick", 3),
        ("stop",),
    ]


def test_run_does_not_wait_between_tags_by_default():
    ChassisAlignPickSequence, = load_symbols("ChassisAlignPickSequence")
    calls = []

    class FakeSequence(ChassisAlignPickSequence):
        def __init__(self):
            self.args = SimpleNamespace(
                sequence=[1, 2],
                align_only=False,
                dry_run=False,
                wait_key_between_tags=False,
                skip_startup_home=True,
            )

        def select_next_tag(self, remaining_tags):
            return remaining_tags[0]

        def align_tag(self, tag_id):
            calls.append(("align", tag_id))

        def run_pick(self, tag_id):
            calls.append(("pick", tag_id))

        def wait_for_tag_tf_before_pick(self, tag_id):
            return True

        def wait_between_tags(self, tag_id, index, total):
            calls.append(("wait", tag_id, index, total))

        def stop_chassis(self):
            calls.append(("stop",))

    FakeSequence().run()

    assert calls == [
        ("align", 1),
        ("pick", 1),
        ("align", 2),
        ("pick", 2),
        ("stop",),
    ]


def test_run_skips_pick_when_tag_tf_does_not_stabilize():
    ChassisAlignPickSequence, = load_symbols("ChassisAlignPickSequence")
    calls = []

    class FakeSequence(ChassisAlignPickSequence):
        def __init__(self):
            self.args = SimpleNamespace(
                sequence=[1, 2],
                align_only=False,
                dry_run=False,
                wait_key_between_tags=False,
                skip_startup_home=True,
            )

        def select_next_tag(self, remaining_tags):
            return remaining_tags[0]

        def align_tag(self, tag_id):
            calls.append(("align", tag_id))

        def wait_for_tag_tf_before_pick(self, tag_id):
            calls.append(("tf", tag_id))
            return tag_id == 2

        def run_pick(self, tag_id):
            calls.append(("pick", tag_id))

        def stop_chassis(self):
            calls.append(("stop",))

    FakeSequence().run()

    assert calls == [
        ("align", 1),
        ("tf", 1),
        ("align", 2),
        ("tf", 2),
        ("pick", 2),
        ("stop",),
    ]


def test_run_continues_when_child_pick_loses_tag_tf_before_motion():
    ChassisAlignPickSequence, = load_symbols("ChassisAlignPickSequence")
    calls = []

    class FakeSequence(ChassisAlignPickSequence):
        def __init__(self):
            self.args = SimpleNamespace(
                sequence=[1, 2],
                align_only=False,
                dry_run=False,
                wait_key_between_tags=False,
                skip_startup_home=False,
            )

        def select_next_tag(self, remaining_tags):
            return remaining_tags[0]

        def align_tag(self, tag_id):
            calls.append(("align", tag_id))

        def wait_for_tag_tf_before_pick(self, tag_id):
            calls.append(("tf", tag_id))
            return True

        def run_pick(self, tag_id):
            calls.append(("pick", tag_id))
            return tag_id != 1

        def run_startup_home(self, tag_id):
            calls.append(("startup_home", tag_id))

        def stop_chassis(self):
            calls.append(("stop",))

    FakeSequence().run()

    assert calls == [
        ("align", 1),
        ("tf", 1),
        ("pick", 1),
        ("align", 2),
        ("tf", 2),
        ("pick", 2),
        ("startup_home", 2),
        ("stop",),
    ]


def test_run_does_not_require_tag_tf_for_align_only():
    ChassisAlignPickSequence, = load_symbols("ChassisAlignPickSequence")
    calls = []

    class FakeSequence(ChassisAlignPickSequence):
        def __init__(self):
            self.args = SimpleNamespace(
                sequence=[1],
                align_only=True,
                dry_run=False,
                wait_key_between_tags=False,
            )

        def select_next_tag(self, remaining_tags):
            return remaining_tags[0]

        def align_tag(self, tag_id):
            calls.append(("align", tag_id))

        def wait_for_tag_tf_before_pick(self, tag_id):
            calls.append(("tf", tag_id))
            return False

        def run_pick(self, tag_id):
            calls.append(("pick", tag_id))

        def stop_chassis(self):
            calls.append(("stop",))

    FakeSequence().run()

    assert calls == [
        ("align", 1),
        ("pick", 1),
        ("stop",),
    ]


def test_run_stops_after_requested_success_count():
    ChassisAlignPickSequence, = load_symbols("ChassisAlignPickSequence")
    calls = []

    class FakeSequence(ChassisAlignPickSequence):
        def __init__(self):
            self.args = SimpleNamespace(
                sequence=[1, 2, 3, 4], max_targets=2,
                fail_on_skip=True, align_only=True, dry_run=False,
                wait_key_between_tags=False,
            )

        def select_next_tag(self, remaining_tags):
            return remaining_tags[0]

        def align_tag(self, tag_id):
            calls.append(("align", tag_id))

        def run_pick(self, tag_id):
            calls.append(("pick", tag_id))

        def run_startup_home(self, tag_id):
            calls.append(("home", tag_id))

        def stop_chassis(self):
            calls.append(("stop",))

    FakeSequence().run()

    assert calls == [
        ("align", 1), ("pick", 1), ("home", 1),
        ("align", 2), ("pick", 2), ("home", 2),
        ("stop",),
    ]


def test_strict_run_fails_on_alignment_skip():
    ChassisAlignPickSequence, AlignmentTimeout = load_symbols(
        "ChassisAlignPickSequence", "AlignmentTimeout")

    class FakeSequence(ChassisAlignPickSequence):
        def __init__(self):
            self.args = SimpleNamespace(
                sequence=[1, 2], max_targets=1, fail_on_skip=True,
                align_only=True, dry_run=False, wait_key_between_tags=False,
            )

        def select_next_tag(self, remaining_tags):
            return remaining_tags[0]

        def align_tag(self, _tag_id):
            raise AlignmentTimeout("alignment failed")

        def stop_chassis(self):
            pass

    with pytest.raises(RuntimeError, match="alignment failed"):
        FakeSequence().run()
