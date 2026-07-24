#include <string>
#include <ros/ros.h>
#include <serial/serial.h>
#include <std_msgs/String.h>
#include <std_msgs/Empty.h>
#include <std_msgs/UInt16.h>
#include <std_msgs/Float32.h>
#include <sensor_msgs/JointState.h>
#include "MirobotType.h"

serial::Serial _serial; // serial object

int GetPose(Pose *pose)
{
    _serial.write("?\n");
    ros::Duration(0.1).sleep();
    std::stringstream ss;
    if (_serial.available())
    {
        // ROS_INFO("Reading from serial port:\n");
        std_msgs::String result;
        result.data = _serial.read(_serial.available());
        // result.data[1];
        ROS_DEBUG_THROTTLE(5.0, "Read %zu bytes from arm serial", result.data.size());
        if (!result.data.empty() && result.data[result.data.size() - 1] == '\n') // check the last char ,if it is \n,the command is complate
        {

            std_msgs::String info;
            ss << result;
            info.data = ss.str();
            ss.str(""); // use this method to clear ss!

            // read_pub.publish(info);
            std::string pose_info = info.data;
            size_t start = pose_info.rfind('<');
            size_t end = pose_info.rfind('>');
            if (start == std::string::npos || end == std::string::npos || end <= start)
            {
                return GetPoseFail;
            }

            pose_info = pose_info.substr(start, end - start + 1);

            size_t state_end = pose_info.find(',');
            if (state_end == std::string::npos || state_end <= 1)
            {
                return GetPoseFail;
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
                return GetPoseFail;
            }

            auto parse_values = [](const std::string &text, size_t start_index, double *values, size_t count) -> bool
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
            };

            double joint_values[7];
            double cart_values[6];
            if (!parse_values(pose_info, angle_start + 15, joint_values, 7) ||
                !parse_values(pose_info, cart_start + 31, cart_values, 6))
            {
                return GetPoseFail;
            }

            pose->jointAngle[3] = joint_values[0]; // joint A
            pose->jointAngle[4] = joint_values[1]; // joint B
            pose->jointAngle[5] = joint_values[2]; // joint C
            pose->jointAngle[6] = joint_values[3]; // joint D
            pose->jointAngle[0] = joint_values[4]; // joint X
            pose->jointAngle[1] = joint_values[5]; // joint Y
            pose->jointAngle[2] = joint_values[6]; // joint Z

            pose->x = cart_values[0]; // X
            pose->y = cart_values[1]; // Y
            pose->z = cart_values[2]; // Z
            pose->a = cart_values[3]; // A
            pose->b = cart_values[4]; // B
            pose->c = cart_values[5]; // C

            return GetPoseSuccess;
        }
        else
        {
            ss << result;
            if (ss.str().size() > 500) // if ss is too long and still not have \n ,we should abandon it
            {
                ss.str("");
            }
        }
    }
    return GetPoseFail;
}

Pose res, pose;
int c;
int main(int argc, char **argv)
{
    ros::init(argc, argv, "mirobot_pub_node"); // 初始化，节点名称 "Mirobot_write_node"
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
    ros::Publisher joint_pub = nh.advertise<sensor_msgs::JointState>("/joint_states", 1);
    ros::Rate loop_rate(15); // 指定了频率为15Hz

    std::string serial_port = "/dev/serial/by-path/pci-0000:00:14.0-usb-0:6.1:1.0-port0";
    bool auto_home_on_start = true;
    bool move_to_zero_pose_on_start = false;
    private_nh.param("serial_port", serial_port, std::string("/dev/serial/by-path/pci-0000:00:14.0-usb-0:6.1:1.0-port0"));
    private_nh.param("auto_home_on_start", auto_home_on_start, true);
    private_nh.param("move_to_zero_pose_on_start", move_to_zero_pose_on_start, false);

    try // 尝试连接机械臂的串口
    {
        _serial.setPort(serial_port);
        _serial.setBaudrate(115200);
        serial::Timeout to = serial::Timeout::simpleTimeout(1000);
        _serial.setTimeout(to);
        _serial.open();
        _serial.write("M50\r\n");
        ROS_INFO_STREAM("Port has been open successfully: " << serial_port);
    }
    catch (serial::IOException &e)
    {
        ROS_ERROR_STREAM("Unable to open port: " << serial_port);
        return -1;
    }

    if (_serial.isOpen())
    {
        ros::Duration(1).sleep();
        ROS_INFO_STREAM("Attach and wait for commands");
    }

    if (auto_home_on_start)
    {
        ROS_INFO_STREAM("Auto homing on startup is enabled.");
        _serial.write("$H\n");
    }
    else if (move_to_zero_pose_on_start)
    {
        ROS_INFO_STREAM("Auto homing on startup is disabled. Moving directly to zero pose.");
        _serial.write("M50 G0 X0 Y0 Z0 A0B0C0 F3000\r\n");
    }
    else
    {
        ROS_INFO_STREAM("Startup motion is disabled. Keeping current arm pose.");
    }

    while (ros::ok())
    {
        ros::spinOnce();
        c = GetPose(&pose);
        if (c == GetPoseSuccess)
        {
            sensor_msgs::JointState joint_state;
            joint_state.header.stamp = ros::Time::now();
            joint_state.name.resize(6);
            joint_state.position.resize(6);
            joint_state.name[0] = "joint1";
            joint_state.position[0] = pose.jointAngle[0] * pi / 180;
            joint_state.name[1] = "joint2";
            joint_state.position[1] = pose.jointAngle[1] * pi / 180;
            joint_state.name[2] = "joint3";
            joint_state.position[2] = pose.jointAngle[2] * pi / 180;
            joint_state.name[3] = "joint4";
            joint_state.position[3] = pose.jointAngle[3] * pi / 180;
            joint_state.name[4] = "joint5";
            joint_state.position[4] = pose.jointAngle[4] * pi / 180;
            joint_state.name[5] = "joint6";
            joint_state.position[5] = pose.jointAngle[5] * pi / 180;
            joint_pub.publish(joint_state);
        }
        loop_rate.sleep();
    }

    return 0;
}
