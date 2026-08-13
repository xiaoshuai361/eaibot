from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
CONTROLLER = ROOT / "mirobot_ws/src/mirobot_urdf_2/src/mirobot_arm_controller.cpp"
LAUNCH = ROOT / "mirobot_ws/src/mirobot_moveit_config/launch/mirobot.launch"
JOINT_LIMITS = ROOT / "mirobot_ws/src/mirobot_moveit_config/config/joint_limits.yaml"


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
    assert "waitForFirmwareTarget(" in source
    assert "ROS_INFO_STREAM(\"Sending arm command" not in source
    assert "ROS_DEBUG_STREAM(\"Arm GCode:" in source


def test_controller_sparsifies_dense_trajectories_and_repeats_final_target():
    source = CONTROLLER.read_text()

    assert "kTrajectoryWaypointStride = 4" in source
    assert "kFinalTargetRepeats = 3" in source
    assert "kSparseWaypointSleepSeconds = 0.15" in source
    assert "bool shouldSendSparseWaypoint" in source
    assert "index == 1 ||" in source
    assert "((index - 1) % kTrajectoryWaypointStride) == 0" in source
    assert "sendArmCommand(target_positions, feedrate)" in source
    assert "repeat < final_target_repeats" in source
    assert "kContactProbeWaypointSleepSeconds = 0.02" in source
    assert "kContactProbeCompletionPollSeconds = 0.03" in source
    assert "wait_time = point.time_from_start" not in source


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
    assert "struct JointTargetError" in source
    assert "maxJointTargetErrorRadians" in source
    assert "final joint%d error" in source
    assert "measured=%.3f rad, target=%.3f rad" in source
    assert "g_trajectory_goal_tolerance_rad" in source
    assert "if (!waitForFirmwareTarget(" in source
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


def test_joint6_trajectory_has_no_extra_travel_rejection():
    source = CONTROLLER.read_text()
    launch = LAUNCH.read_text()

    assert "mirobot_motion_math.h" not in source
    assert "shortestAngularDistance" in source
    assert "nearestEquivalentAngleWithinLimits" not in source
    assert "adjusted_positions" not in source
    assert "accumulatedJointTravel" not in source
    assert "g_joint6_max_trajectory_travel_rad" not in source
    assert "joint6_max_trajectory_travel_rad" not in launch
    assert "Rejecting trajectory before execution: joint6 travel" not in source
    assert "JointConstraint" not in source


def test_controller_exposes_startup_homing_service_that_sends_grbl_home():
    source = CONTROLLER.read_text()

    assert "#include <std_srvs/Trigger.h>" in source
    assert "trigger_startup_home" in source
    assert 'nh.advertiseService("mirobot_startup_home", trigger_startup_home)' in source
    assert '_serial.write("$H\\n")' in source


def test_controller_owns_contact_signal_on_the_shared_pump_serial():
    source = CONTROLLER.read_text()
    launch = LAUNCH.read_text()

    assert "#include <std_srvs/SetBool.h>" in source
    assert 'kContactTriggeredFrame("3\\r\\n")' in source
    assert "g_pump_rx_buffer" in source
    assert "consumePumpSerialFrames" in source
    assert "frame == kContactTriggeredFrame" in source
    assert "g_pump_serial_mutex" in source
    assert '"mirobot_contact_probe_enable"' in source
    assert '"mirobot_contact_state"' in source
    assert "Rejecting pump ON while the contact probe is armed." in source
    assert '<arg name="contact_probe_feedrate" default="1200" />' in launch
    assert '<param name="contact_probe_feedrate" value="$(arg contact_probe_feedrate)" />' in launch
    assert "g_contact_probe_armed ? g_contact_probe_feedrate : g_arm_feedrate" in source
    assert "executeContactProbeTrajectory" in source
    assert "readContactTriggered(&triggered)" in source
    assert "for (size_t index = 1; index < point_count; ++index)" in source
    assert "waitForFirmwareTarget(" in source
    assert "Contact probe stopped at the current 2 mm waypoint." in source
