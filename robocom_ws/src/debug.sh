#!/bin/bash

CAMERA_PYTHON="/home/eaibot/miniconda3/bin/python"

echo "=== 1. 摄像头设备节点 ==="
ls -l /dev/video* 2>/dev/null || echo "未发现 /dev/video*"
echo ""

echo "=== 2. 摄像头索引扫描 ==="
echo "运行 /home/eaibot/find_camera.py，按 q 关闭当前画面并切到下一个摄像头。"
"${CAMERA_PYTHON}" /home/eaibot/find_camera.py
echo ""

echo "=== 3. rostopic scan ==="
rostopic list | grep scan || echo "未发现包含 scan 的话题"
echo ""

echo "=== 4. rostopic cmd_vel ==="
rostopic list | grep cmd_vel || echo "未发现包含 cmd_vel 的话题"
echo ""

echo "=== 5. 激光频率 ==="
timeout 3 rostopic hz /scan
echo ""

echo "=== 6. 节点参数 ==="
rosparam list | grep linear || echo "未发现 linear 参数"
rosparam list | grep angular || echo "未发现 angular 参数"
echo ""

echo "=== 7. 节点列表 ==="
rosnode list