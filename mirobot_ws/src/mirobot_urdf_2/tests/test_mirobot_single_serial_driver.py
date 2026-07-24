from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
CONTROLLER = ROOT / "mirobot_ws/src/mirobot_urdf_2/src/mirobot_arm_controller.cpp"
LAUNCH = ROOT / "mirobot_ws/src/mirobot_moveit_config/launch/mirobot.launch"


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
    assert "waitForMeasuredJointState" in source
    assert "publishCommanded" not in source
    assert "initializeZeroCommanded" not in source
    assert "commanded_joint_state" not in source


def test_controller_pauses_measured_query_during_motion_and_keeps_logs_quiet():
    source = CONTROLLER.read_text()

    assert "g_executing_trajectory" in source
    assert "if (!g_publish_joint_states || isExecutingTrajectory())" in source
    assert "waitForMeasuredJointState(*joint_pub)" in source
    assert "ROS_INFO_STREAM(\"Sending arm command" not in source
    assert "ROS_DEBUG_STREAM(\"Arm GCode:" in source


def test_controller_skips_stale_start_point_and_waits_after_each_sent_point():
    source = CONTROLLER.read_text()

    assert "index == 0 && n_tra_points > 1" in source
    assert "Skipping first trajectory point" in source
    assert "wait_time = point.time_from_start - goalPtr->trajectory.points[index - 1].time_from_start" in source
    assert "index + 1 < n_tra_points" not in source


def test_controller_republishes_recent_real_measurement_when_serial_query_misses():
    source = CONTROLLER.read_text()
    launch = LAUNCH.read_text()

    assert "g_last_measured_pose" in source
    assert "g_have_last_measured_pose" in source
    assert "g_measured_state_hold_seconds" in source
    assert "rememberMeasuredPose(pose)" in source
    assert "publishLastMeasuredJointStateIfFresh(*joint_pub)" in source
    assert '<arg name="measured_state_hold_seconds" default="1.5" />' in launch
    assert '<param name="measured_state_hold_seconds" value="$(arg measured_state_hold_seconds)" />' in launch

    assert "commanded_joint_state" not in source
    assert "publishCommanded" not in source


def test_controller_backs_off_serial_pose_queries_after_a_miss():
    source = CONTROLLER.read_text()
    launch = LAUNCH.read_text()

    assert "g_pose_query_failure_cooldown_seconds" in source
    assert "g_next_pose_query_time" in source
    assert "markPoseQueryMiss()" in source
    assert "if (ros::Time::now() < g_next_pose_query_time)" in source
    assert "publishLastMeasuredJointStateIfFresh(*joint_pub)" in source
    assert '<arg name="pose_query_failure_cooldown_seconds" default="0.6" />' in launch
    assert '<param name="pose_query_failure_cooldown_seconds" value="$(arg pose_query_failure_cooldown_seconds)" />' in launch


def test_controller_retries_post_motion_measured_state_before_reporting_success():
    source = CONTROLLER.read_text()
    launch = LAUNCH.read_text()

    assert "g_post_motion_state_attempts" in source
    assert "g_post_motion_state_retry_delay_seconds" in source
    assert "attempt < g_post_motion_state_attempts" in source
    assert "queryCurrentPoseUnlocked(&pose, g_post_motion_state_timeout_seconds)" in source
    assert "ros::Duration(g_post_motion_state_retry_delay_seconds).sleep()" in source
    assert "Trajectory finished, but measured joint state was not updated from serial after" in source
    assert '<arg name="post_motion_state_attempts" default="3" />' in launch
    assert '<arg name="post_motion_state_retry_delay_seconds" default="0.20" />' in launch
    assert '<param name="post_motion_state_attempts" value="$(arg post_motion_state_attempts)" />' in launch
    assert '<param name="post_motion_state_retry_delay_seconds" value="$(arg post_motion_state_retry_delay_seconds)" />' in launch
