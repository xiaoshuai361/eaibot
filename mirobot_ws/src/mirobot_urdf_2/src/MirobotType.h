#ifndef MIROBOTYPE_H
#define MIROBOTYPE_H

#define pi 3.1415926
enum result
{
    success = 1,
    fail = -1
};

enum connect_state
{
    MirobotConnect_Success = 1,
    MirobotConnect_Error = -1
};

typedef struct tagPose
{
    int state;
    float x;
    float y;
    float z;
    float a;
    float b;
    float c;
    float jointAngle[7];
} Pose;

typedef struct tagCartCmd
{
    float x;
    float y;
    float z;
    float a;
    float b;
    float c;
    int speed;
} CartCmd;

enum
{
    Idle,
    Alarm,
    Home,
    Unknow = -1

};

enum
{
    GetPoseSuccess = 1,
    GetPoseFail = -1

};

enum
{
    GetOkSuccess = 1,
    GetOkFail = -1

};

#endif
