from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[4]
CONTROLLER = ROOT / "mirobot_ws/src/mirobot_urdf_2/src/mirobot_arm_controller.cpp"
LAUNCH = ROOT / "mirobot_ws/src/mirobot_moveit_config/launch/mirobot.launch"
JOINT_LIMITS = ROOT / "mirobot_ws/src/mirobot_moveit_config/config/joint_limits.yaml"
MOTION_MATH = ROOT / "mirobot_ws/src/mirobot_urdf_2/src/mirobot_motion_math.h"


def test_real_robot_launch_defaults_to_single_arm_serial_owner():
    source = LAUNCH.read_text()

    assert '<arg name="single_arm_serial" default="true" />' in source
    assert '<node name="mirobot_Pub_node"' in source
    assert 'unless="$(arg single_arm_serial)"' in source
    assert '<node name="mirobot_arm_controller_node"' in source
    assert '<param name="publish_joint_states" value="$(arg single_arm_serial)" />' in source


def test_controller_publishes_only_measured_joint_states_not_commanded_fake_state():
    source = CONTROLLER.read_text()

    assert '#include <sensor_msgs/JointState.h>' in source
    assert '#include "MirobotType.h"' in source
    assert "publishMeasuredJointState" in source
    assert "queryCurrentPoseUnlocked" in source
    assert "waitForFirmwareTarget" in source
    assert "publishCommanded" not in source
    assert "initializeZeroCommanded" not in source
    assert "commanded_joint_state" not in source


def test_controller_pauses_measured_query_during_motion_and_keeps_logs_quiet():
    source = CONTROLLER.read_text()

    assert "g_executing_trajectory" in source
    assert "if (!g_publish_joint_states || isExecutingTrajectory())" in source
    assert "waitForFirmwareTarget(target_positions, *joint_pub)" in source
    assert "ROS_INFO_STREAM(\"Sending arm command" not in source
    assert "ROS_DEBUG_STREAM(\"Arm GCode:" in source


def test_controller_skips_stale_start_point_and_waits_after_each_sent_point():
    source = CONTROLLER.read_text()

    assert "index == 0 && n_tra_points > 1" in source
    assert "Skipping first trajectory point" in source
    assert "wait_time = point.time_from_start - goalPtr->trajectory.points[index - 1].time_from_start" in source
    assert "index + 1 < n_tra_points" not in source


def test_controller_never_republishes_cached_measurement_with_a_fresh_stamp():
    source = CONTROLLER.read_text()

    assert "publishLastMeasuredJointStateIfFresh" not in source
    assert "g_last_measured_pose" not in source
    assert "measured_state_hold_seconds" not in source
    assert "commanded_joint_state" not in source
    assert "publishCommanded" not in source


def test_controller_backs_off_serial_pose_queries_after_a_miss():
    source = CONTROLLER.read_text()
    launch = LAUNCH.read_text()

    assert "g_pose_query_failure_cooldown_seconds" in source
    assert "g_next_pose_query_time" in source
    assert "markPoseQueryMiss()" in source
    assert "if (ros::Time::now() < g_next_pose_query_time)" in source
    assert '<arg name="pose_query_failure_cooldown_seconds" default="0.6" />' in launch
    assert '<param name="pose_query_failure_cooldown_seconds" value="$(arg pose_query_failure_cooldown_seconds)" />' in launch


def test_controller_requires_firmware_idle_and_target_tolerance_before_success():
    source = CONTROLLER.read_text()
    launch = LAUNCH.read_text()

    assert "bool waitForFirmwareTarget" in source
    assert "pose.state == Alarm" in source
    assert "pose.state == Idle" in source
    assert "maxJointTargetErrorRadians" in source
    assert "g_trajectory_goal_tolerance_rad" in source
    assert "if (!waitForFirmwareTarget(target_positions, *joint_pub))" in source
    assert "moveit_server->setAborted()" in source
    assert '<arg name="trajectory_completion_timeout_seconds" default="15.0" />' in launch
    assert '<arg name="trajectory_goal_tolerance_rad" default="0.05" />' in launch


def test_driver_feedrate_is_parameterized_and_moveit_limits_are_conservative():
    source = CONTROLLER.read_text()
    launch = LAUNCH.read_text()
    limits = JOINT_LIMITS.read_text()

    assert "g_arm_feedrate" in source
    assert '" F" + feedrate' in source
    assert "F3000" not in source
    assert '<arg name="arm_feedrate" default="1200" />' in launch
    assert limits.count("max_velocity: 0.35") == 6
    assert limits.count("has_acceleration_limits: true") == 6
    assert limits.count("max_acceleration: 0.5") == 6


def test_joint6_trajectory_is_not_rewritten_and_excess_travel_is_rejected():
    source = CONTROLLER.read_text()
    math_source = MOTION_MATH.read_text()
    launch = LAUNCH.read_text()

    assert '#include "mirobot_motion_math.h"' in source
    assert "nearestEquivalentAngleWithinLimits" not in source
    assert "adjusted_positions" not in source
    assert "accumulatedJointTravel" in source
    assert "g_joint6_max_trajectory_travel_rad" in source
    assert "accumulatedJointTravel" in math_source
    assert '<arg name="joint6_max_trajectory_travel_rad" default="3.0" />' in launch
    assert "JointConstraint" not in source


def test_joint6_travel_math_measures_the_original_path(tmp_path):
    source = tmp_path / "motion_math_test.cpp"
    binary = tmp_path / "motion_math_test"
    source.write_text(
        '#include "mirobot_motion_math.h"\n'
        '#include <vector>\n'
        'int main() {\n'
        '  std::vector<double> path;\n'
        '  path.push_back(0.0);\n'
        '  path.push_back(0.4);\n'
        '  path.push_back(-0.1);\n'
        '  const double result = accumulatedJointTravel(path, 0.0);\n'
        '  return std::fabs(result - 0.9) < 1e-6 ? 0 : 1;\n'
        '}\n')

    subprocess.check_call([
        "g++", "-std=c++11", "-I", str(MOTION_MATH.parent),
        str(source), "-o", str(binary),
    ])
    subprocess.check_call([str(binary)])


def test_controller_exposes_startup_homing_service_that_sends_grbl_home():
    source = CONTROLLER.read_text()

    assert "#include <std_srvs/Trigger.h>" in source
    assert "trigger_startup_home" in source
    assert 'nh.advertiseService("mirobot_startup_home", trigger_startup_home)' in source
    assert '_serial.write("$H\\n")' in source
