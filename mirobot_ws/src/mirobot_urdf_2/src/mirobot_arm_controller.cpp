#include <ros/ros.h>
#include <actionlib/server/simple_action_server.h>
#include <control_msgs/FollowJointTrajectoryAction.h>
#include <std_msgs/Float32MultiArray.h>
#include <iostream>
#include <serial/serial.h>
#include <std_msgs/String.h>
#include <std_msgs/Empty.h>
#include <std_msgs/UInt16.h>
#include <std_msgs/Float32.h>
#include <std_srvs/Trigger.h>
#include <moveit_msgs/RobotTrajectory.h>
#include <mirobot_urdf_2/mirobotPump.h>
#include <sensor_msgs/JointState.h>
#include <trajectory_msgs/JointTrajectoryPoint.h>
#include "MirobotType.h"
#include <boost/bind.hpp>
#include <boost/thread/mutex.hpp>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

using namespace std;

typedef actionlib::SimpleActionServer<control_msgs::FollowJointTrajectoryAction> Server;

serial::Serial _serial;
serial::Serial g_pump_serial;

namespace
{
	std::string g_serial_port("/dev/serial/by-path/pci-0000:00:14.0-usb-0:6.1:1.0-port0");
	std::string g_pump_serial_port("/dev/serial/by-path/pci-0000:00:14.0-usb-0:6.2:1.0-port0");
	int g_pump_response_wait_ms = 200;
	bool g_publish_joint_states = true;
	bool g_auto_home_on_start = true;
	bool g_move_to_zero_pose_on_start = false;
	double g_joint_state_publish_hz = 5.0;
	double g_pose_query_timeout_seconds = 0.40;
	double g_pose_query_poll_seconds = 0.02;
	double g_post_motion_state_timeout_seconds = 1.0;
	int g_post_motion_state_attempts = 3;
	double g_post_motion_state_retry_delay_seconds = 0.20;
	double g_measured_state_hold_seconds = 1.5;
	double g_pose_query_failure_cooldown_seconds = 0.6;
	const std::string kPumpOnCommand("1");
	const std::string kPumpOffCommand("2");
	boost::mutex g_arm_serial_mutex;
	boost::mutex g_execution_state_mutex;
	boost::mutex g_last_measured_pose_mutex;
	bool g_executing_trajectory = false;
	bool g_have_last_measured_pose = false;
	Pose g_last_measured_pose;
	ros::Time g_last_measured_pose_stamp;
	ros::Time g_next_pose_query_time;

	void setExecutingTrajectory(bool executing)
	{
		boost::mutex::scoped_lock lock(g_execution_state_mutex);
		g_executing_trajectory = executing;
	}

	bool isExecutingTrajectory()
	{
		boost::mutex::scoped_lock lock(g_execution_state_mutex);
		return g_executing_trajectory;
	}

	class TrajectoryExecutionGuard
	{
	public:
		TrajectoryExecutionGuard()
		{
			setExecutingTrajectory(true);
		}

		~TrajectoryExecutionGuard()
		{
			setExecutingTrajectory(false);
		}
	};

	bool ensureSerialOpen()
	{
		if (_serial.isOpen())
		{
			return true;
		}

		try
		{
			_serial.setPort(g_serial_port);
			_serial.setBaudrate(115200);
			serial::Timeout to = serial::Timeout::simpleTimeout(1000);
			_serial.setTimeout(to);
			_serial.open();
			_serial.write("M50\r\n");
			ROS_INFO_STREAM("Port has been open successfully: " << g_serial_port);
			ros::Duration(1).sleep();
			ROS_INFO_STREAM("Attach and wait for commands");
			return true;
		}
		catch (serial::IOException &e)
		{
			ROS_ERROR_STREAM("Unable to open port: " << g_serial_port);
			return false;
		}
	}

	void closeSerialIfOpen()
	{
		if (_serial.isOpen())
		{
			_serial.close();
		}
	}

	bool ensurePumpSerialOpen()
	{
		if (g_pump_serial.isOpen())
		{
			return true;
		}

		try
		{
			g_pump_serial.setPort(g_pump_serial_port);
			g_pump_serial.setBaudrate(115200);
			g_pump_serial.setBytesize(serial::eightbits);
			g_pump_serial.setParity(serial::parity_none);
			g_pump_serial.setStopbits(serial::stopbits_one);
			g_pump_serial.setFlowcontrol(serial::flowcontrol_none);
			serial::Timeout to = serial::Timeout::simpleTimeout(500);
			g_pump_serial.setTimeout(to);
			g_pump_serial.open();
			ROS_INFO_STREAM("Pump controller port has been open successfully: " << g_pump_serial_port);
			return true;
		}
		catch (serial::IOException &e)
		{
			ROS_ERROR_STREAM("Unable to open pump controller port: " << g_pump_serial_port);
			return false;
		}
	}

	void clearPumpSerialInput()
	{
		if (!g_pump_serial.isOpen())
		{
			return;
		}

		while (g_pump_serial.available())
		{
			g_pump_serial.read(g_pump_serial.available());
		}
	}

	bool sendPumpCommand(const std::string &command, std::string *response)
	{
		if (!ensurePumpSerialOpen())
		{
			return false;
		}

		clearPumpSerialInput();
		g_pump_serial.write(command);
		ros::Duration(static_cast<double>(g_pump_response_wait_ms) / 1000.0).sleep();

		if (!g_pump_serial.available())
		{
			ROS_ERROR_STREAM("Pump controller did not reply after command: " << command);
			return false;
		}

		*response = g_pump_serial.read(g_pump_serial.available());
		ROS_INFO_STREAM("Pump controller replied: " << *response);
		return true;
	}

	void closePumpSerialIfOpen()
	{
		if (g_pump_serial.isOpen())
		{
			g_pump_serial.close();
		}
	}

	bool parseValues(const std::string &text, size_t start_index, double *values, size_t count)
	{
		size_t cursor = start_index;
		for (size_t index = 0; index < count; ++index)
		{
			size_t next = text.find(',', cursor);
			if (next == std::string::npos || next <= cursor)
			{
				return false;
			}
			values[index] = atof(text.substr(cursor, next - cursor).c_str());
			cursor = next + 1;
		}
		return true;
	}

	bool parseArmPoseResponse(const std::string &response, Pose *pose)
	{
		if (pose == NULL)
		{
			return false;
		}

		size_t start = response.rfind('<');
		size_t end = response.rfind('>');
		if (start == std::string::npos || end == std::string::npos || end <= start)
		{
			return false;
		}

		std::string pose_info = response.substr(start, end - start + 1);
		size_t state_end = pose_info.find(',');
		if (state_end == std::string::npos || state_end <= 1)
		{
			return false;
		}

		std::string state = pose_info.substr(1, state_end - 1);
		if (state == "Idle")
		{
			pose->state = Idle;
		}
		else if (state == "Home")
		{
			pose->state = Home;
		}
		else if (state == "Alarm")
		{
			pose->state = Alarm;
		}
		else
		{
			pose->state = Unknow;
		}

		size_t angle_start = pose_info.find("Angle(ABCDXYZ):");
		size_t cart_start = pose_info.find("Cartesian coordinate(XYZ RxRyRz):");
		if (angle_start == std::string::npos || cart_start == std::string::npos || cart_start <= angle_start)
		{
			return false;
		}

		double joint_values[7];
		double cart_values[6];
		if (!parseValues(pose_info, angle_start + 15, joint_values, 7) ||
			!parseValues(pose_info, cart_start + 31, cart_values, 6))
		{
			return false;
		}

		pose->jointAngle[3] = joint_values[0];
		pose->jointAngle[4] = joint_values[1];
		pose->jointAngle[5] = joint_values[2];
		pose->jointAngle[6] = joint_values[3];
		pose->jointAngle[0] = joint_values[4];
		pose->jointAngle[1] = joint_values[5];
		pose->jointAngle[2] = joint_values[6];

		pose->x = cart_values[0];
		pose->y = cart_values[1];
		pose->z = cart_values[2];
		pose->a = cart_values[3];
		pose->b = cart_values[4];
		pose->c = cart_values[5];
		return true;
	}

	bool queryCurrentPoseUnlocked(Pose *pose, double timeout_seconds)
	{
		while (_serial.available())
		{
			_serial.read(_serial.available());
		}

		_serial.write("?\n");
		const ros::Time deadline = ros::Time::now() + ros::Duration(timeout_seconds);
		std::string response;
		while (ros::ok() && ros::Time::now() < deadline)
		{
			if (_serial.available())
			{
				response += _serial.read(_serial.available());
				if (response.find('>') != std::string::npos)
				{
					break;
				}
			}
			ros::Duration(g_pose_query_poll_seconds).sleep();
		}

		if (response.empty())
		{
			return false;
		}
		return parseArmPoseResponse(response, pose);
	}

	void fillJointStateFromPose(const Pose &pose, sensor_msgs::JointState *joint_state)
	{
		joint_state->header.stamp = ros::Time::now();
		joint_state->name.resize(6);
		joint_state->position.resize(6);
		joint_state->name[0] = "joint1";
		joint_state->position[0] = pose.jointAngle[0] * pi / 180;
		joint_state->name[1] = "joint2";
		joint_state->position[1] = pose.jointAngle[1] * pi / 180;
		joint_state->name[2] = "joint3";
		joint_state->position[2] = pose.jointAngle[2] * pi / 180;
		joint_state->name[3] = "joint4";
		joint_state->position[3] = pose.jointAngle[3] * pi / 180;
		joint_state->name[4] = "joint5";
		joint_state->position[4] = pose.jointAngle[4] * pi / 180;
		joint_state->name[5] = "joint6";
		joint_state->position[5] = pose.jointAngle[5] * pi / 180;
	}

	void rememberMeasuredPose(const Pose &pose)
	{
		boost::mutex::scoped_lock lock(g_last_measured_pose_mutex);
		g_last_measured_pose = pose;
		g_last_measured_pose_stamp = ros::Time::now();
		g_have_last_measured_pose = true;
	}

	void publishMeasuredJointState(const Pose &pose, const ros::Publisher &joint_pub)
	{
		rememberMeasuredPose(pose);

		sensor_msgs::JointState joint_state;
		fillJointStateFromPose(pose, &joint_state);
		joint_pub.publish(joint_state);
	}

	bool publishLastMeasuredJointStateIfFresh(const ros::Publisher &joint_pub)
	{
		if (g_measured_state_hold_seconds <= 0.0)
		{
			return false;
		}

		Pose pose;
		{
			boost::mutex::scoped_lock lock(g_last_measured_pose_mutex);
			if (!g_have_last_measured_pose)
			{
				return false;
			}
			if ((ros::Time::now() - g_last_measured_pose_stamp).toSec() > g_measured_state_hold_seconds)
			{
				return false;
			}
			pose = g_last_measured_pose;
		}

		sensor_msgs::JointState joint_state;
		fillJointStateFromPose(pose, &joint_state);
		joint_pub.publish(joint_state);
		return true;
	}

	void markPoseQueryMiss()
	{
		if (g_pose_query_failure_cooldown_seconds <= 0.0)
		{
			return;
		}
		g_next_pose_query_time = ros::Time::now() + ros::Duration(g_pose_query_failure_cooldown_seconds);
	}

	void markPoseQuerySuccess()
	{
		g_next_pose_query_time = ros::Time(0);
	}

	void forgetLastMeasuredPose()
	{
		boost::mutex::scoped_lock lock(g_last_measured_pose_mutex);
		g_have_last_measured_pose = false;
	}

	bool waitForMeasuredJointState(const ros::Publisher &joint_pub)
	{
		if (!g_publish_joint_states)
		{
			return true;
		}

		Pose pose;
		for (int attempt = 0; attempt < g_post_motion_state_attempts; ++attempt)
		{
			if (queryCurrentPoseUnlocked(&pose, g_post_motion_state_timeout_seconds))
			{
				markPoseQuerySuccess();
				publishMeasuredJointState(pose, joint_pub);
				return true;
			}

			if (attempt + 1 < g_post_motion_state_attempts &&
				g_post_motion_state_retry_delay_seconds > 0.0)
			{
				ros::Duration(g_post_motion_state_retry_delay_seconds).sleep();
			}
		}

		markPoseQueryMiss();
		ROS_WARN("Trajectory finished, but measured joint state was not updated from serial after %d attempt(s).",
				 g_post_motion_state_attempts);
		return false;
	}

	void measuredJointStateTimer(const ros::TimerEvent &, ros::Publisher *joint_pub)
	{
		if (!g_publish_joint_states || isExecutingTrajectory())
		{
			return;
		}

		boost::mutex::scoped_lock lock(g_arm_serial_mutex);
		if (!ensureSerialOpen())
		{
			return;
		}
		if (ros::Time::now() < g_next_pose_query_time)
		{
			publishLastMeasuredJointStateIfFresh(*joint_pub);
			return;
		}

		Pose pose;
		if (!queryCurrentPoseUnlocked(&pose, g_pose_query_timeout_seconds))
		{
			markPoseQueryMiss();
			if (publishLastMeasuredJointStateIfFresh(*joint_pub))
			{
				return;
			}
			ROS_WARN_THROTTLE(5.0, "Failed to query measured arm joint state from serial.");
			return;
		}
		markPoseQuerySuccess();
		publishMeasuredJointState(pose, *joint_pub);
	}

	void runStartupMotion()
	{
		if (!g_publish_joint_states)
		{
			return;
		}

		boost::mutex::scoped_lock lock(g_arm_serial_mutex);
		if (!ensureSerialOpen())
		{
			return;
		}

		if (g_auto_home_on_start)
		{
			ROS_INFO_STREAM("Auto homing on startup is enabled.");
			_serial.write("$H\n");
			forgetLastMeasuredPose();
		}
		else if (g_move_to_zero_pose_on_start)
		{
			ROS_INFO_STREAM("Auto homing on startup is disabled. Moving directly to zero pose.");
			_serial.write("M50 G0 X0 Y0 Z0 A0B0C0 F3000\r\n");
		}
		else
		{
			ROS_INFO_STREAM("Startup motion is disabled. Keeping current arm pose.");
		}
	}
}

void execute_callback(const control_msgs::FollowJointTrajectoryGoalConstPtr &goalPtr,
					  Server *moveit_server,
					  ros::Publisher *joint_pub)
{
	if (goalPtr->trajectory.points.empty())
	{
		moveit_server->setAborted();
		return;
	}

	TrajectoryExecutionGuard execution_guard;
	boost::mutex::scoped_lock lock(g_arm_serial_mutex);
	if (!ensureSerialOpen())
	{
		moveit_server->setAborted();
		return;
	}

	std::string Gcode = "";
	char angle0[10];
	char angle1[10];
	char angle2[10];
	char angle3[10];
	char angle4[10];
	char angle5[10];

	const size_t n_tra_points = goalPtr->trajectory.points.size();
	for (size_t index = 0; index < n_tra_points; ++index)
	{
		if (index == 0 && n_tra_points > 1)
		{
			ROS_DEBUG("Skipping first trajectory point because it is MoveIt's planned start state.");
			continue;
		}

		if (moveit_server->isPreemptRequested() || !ros::ok())
		{
			ROS_WARN("Trajectory execution was preempted or ROS is shutting down.");
			moveit_server->setPreempted();
			return;
		}

		const trajectory_msgs::JointTrajectoryPoint &point = goalPtr->trajectory.points[index];
		if (point.positions.size() < 6)
		{
			ROS_ERROR("Trajectory point has fewer than 6 joint positions; aborting.");
			moveit_server->setAborted();
			return;
		}

		sprintf(angle0, "%.2f", point.positions[0] * 57.296);
		sprintf(angle1, "%.2f", point.positions[1] * 57.296);
		sprintf(angle2, "%.2f", point.positions[2] * 57.296);
		sprintf(angle3, "%.2f", point.positions[3] * 57.296);
		sprintf(angle4, "%.2f", point.positions[4] * 57.296);
		sprintf(angle5, "%.2f", point.positions[5] * 57.296);
		Gcode = (std::string) "M50 G0 X" + angle0 + " Y" + angle1 + " Z" + angle2 + " A" + angle3 + "B" + angle4 + "C" + angle5 + " F3000" + "\r\n";
		ROS_DEBUG_STREAM("Arm GCode: " << Gcode);
		_serial.write(Gcode.c_str());

		ros::Duration wait_time;
		if (index == 0)
		{
			wait_time = point.time_from_start;
		}
		else
		{
			wait_time = point.time_from_start - goalPtr->trajectory.points[index - 1].time_from_start;
		}

		if (wait_time.toSec() > 0.0)
		{
			wait_time.sleep();
		}
	}

	waitForMeasuredJointState(*joint_pub);
	if (moveit_server->isPreemptRequested())
	{
		ROS_WARN("Trajectory execution finished after a preempt request.");
		moveit_server->setPreempted();
	}
	else
	{
		moveit_server->setSucceeded();
	}
}

bool toggle_pump(mirobot_urdf_2::mirobotPump::Request &req,
				 mirobot_urdf_2::mirobotPump::Response &res)
{
	std::string pump_response;
	const std::string &pump_command = req.Status ? kPumpOnCommand : kPumpOffCommand;

	if (!sendPumpCommand(pump_command, &pump_response))
	{
		res.Sucess = false;
		return true;
	}

	res.Sucess = true;
	return true;
}

bool trigger_startup_home(std_srvs::Trigger::Request &req,
						  std_srvs::Trigger::Response &res)
{
	(void)req;
	if (isExecutingTrajectory())
	{
		res.success = false;
		res.message = "Arm is executing a trajectory; startup homing was not sent.";
		return true;
	}

	boost::mutex::scoped_lock lock(g_arm_serial_mutex);
	if (!ensureSerialOpen())
	{
		res.success = false;
		res.message = "Could not open arm serial port.";
		return true;
	}

	ROS_INFO_STREAM("Startup homing service requested.");
	_serial.write("$H\n");
	forgetLastMeasuredPose();
	res.success = true;
	res.message = "Sent startup homing command $H.";
	return true;
}

int main(int argc, char *argv[])
{
	ros::init(argc, argv, "moveit_action_server");
	ros::NodeHandle nh;
	ros::NodeHandle private_nh("~");
	private_nh.param("serial_port", g_serial_port, std::string("/dev/serial/by-path/pci-0000:00:14.0-usb-0:6.1:1.0-port0"));
	private_nh.param("pump_serial_port", g_pump_serial_port, std::string("/dev/serial/by-path/pci-0000:00:14.0-usb-0:6.2:1.0-port0"));
	private_nh.param("pump_response_wait_ms", g_pump_response_wait_ms, 200);
	private_nh.param("publish_joint_states", g_publish_joint_states, true);
	private_nh.param("auto_home_on_start", g_auto_home_on_start, true);
	private_nh.param("move_to_zero_pose_on_start", g_move_to_zero_pose_on_start, false);
	private_nh.param("joint_state_publish_hz", g_joint_state_publish_hz, 5.0);
	private_nh.param("pose_query_timeout_seconds", g_pose_query_timeout_seconds, 0.40);
	private_nh.param("post_motion_state_timeout_seconds", g_post_motion_state_timeout_seconds, 1.0);
	private_nh.param("post_motion_state_attempts", g_post_motion_state_attempts, 3);
	private_nh.param("post_motion_state_retry_delay_seconds", g_post_motion_state_retry_delay_seconds, 0.20);
	private_nh.param("measured_state_hold_seconds", g_measured_state_hold_seconds, 1.5);
	private_nh.param("pose_query_failure_cooldown_seconds", g_pose_query_failure_cooldown_seconds, 0.6);

	if (g_joint_state_publish_hz <= 0.0)
	{
		g_joint_state_publish_hz = 5.0;
	}
	if (g_pose_query_timeout_seconds < 0.05)
	{
		g_pose_query_timeout_seconds = 0.05;
	}
	if (g_post_motion_state_timeout_seconds < 0.05)
	{
		g_post_motion_state_timeout_seconds = 0.05;
	}
	if (g_post_motion_state_attempts < 1)
	{
		g_post_motion_state_attempts = 1;
	}
	if (g_post_motion_state_retry_delay_seconds < 0.0)
	{
		g_post_motion_state_retry_delay_seconds = 0.0;
	}
	if (g_measured_state_hold_seconds < 0.0)
	{
		g_measured_state_hold_seconds = 0.0;
	}
	if (g_pose_query_failure_cooldown_seconds < 0.0)
	{
		g_pose_query_failure_cooldown_seconds = 0.0;
	}

	ros::Publisher joint_pub = nh.advertise<sensor_msgs::JointState>("/joint_states", 1);
	ros::Timer joint_state_timer;
	if (g_publish_joint_states)
	{
		joint_state_timer = nh.createTimer(
			ros::Duration(1.0 / g_joint_state_publish_hz),
			boost::bind(&measuredJointStateTimer, _1, &joint_pub));
	}
	runStartupMotion();

	Server moveit_server(nh, "mirobot_arm_controller/follow_joint_trajectory",
						 boost::bind(&execute_callback, _1, &moveit_server, &joint_pub), false);
	moveit_server.start();
	ros::ServiceServer service = nh.advertiseService("switch_pump_status", toggle_pump);
	ros::ServiceServer startup_home_service = nh.advertiseService("mirobot_startup_home", trigger_startup_home);

	ros::AsyncSpinner spinner(2);
	spinner.start();
	ros::waitForShutdown();
	closeSerialIfOpen();
	closePumpSerialIfOpen();
	return 0;
}
