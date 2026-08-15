# 有 Tag 纯 YOLO 抓取测试

本流程只用于独立测试整块有 Tag 物资的 YOLO 抓取，尚未接入正式比赛入口。
原 AprilTag 抓取和 `main.py --tag-pick` 均未改动。

## 1. 数据和对应关系

- 模型：`tag_yolo_v8_yolov5n_640_best.onnx`
- 类别：`0=ID1，1=ID2，2=ID3，3=ID4`
- 视觉预抓文件：`tag_yolo_pick_presets.json`
- 无 Tag 动作文件：`block_mono_pick_place_presets.json`
- 动作对应：`ID1→power，ID2→fire，ID3→gas，ID4→support`

`tag_yolo_pick_presets.json` 只在机器人上示教生成，不要用 Windows 文件覆盖。
它只保存四类的预抓偏移、吸盘姿态和前探方向；放置点、carry 和 idle
始终从无 Tag 动作文件读取。

## 2. 运行前终端

终端 1：启动底盘和机械臂底层。

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/robocom_ws/devel/setup.bash
roslaunch xpkg_bringup bringup_basic_ctrl.launch
```

终端 2：启动 Astra 相机。

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
roslaunch astra_camera astrapro.launch
```

终端 3：启动 MoveIt 和 RViz。

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
roslaunch mirobot_moveit_config mirobot.launch start_rviz:=true
```

终端 4：发布已有手眼标定 TF。

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
roslaunch easy_handeye publish.launch eye_on_hand:=false tracking_base_frame:=camera_link
```

命令终端统一执行：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
conda activate ww
cd /home/eaibot/handeye-calib/src
```

## 3. 预览和距离检查

先确认窗口显示 `ID1~ID4`，且检测框覆盖整个物块：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
conda activate ww
cd /home/eaibot/handeye-calib/src
python3 block_pick_main.py \
  --live-preview \
  --preview-hz 1.0 \
  --confidence 0.3 \
  --config /home/eaibot/handeye-calib/src/config/tag_yolo_grasp.yaml
```

把 ID1 放在实测约 400mm 处，只检查定位，不移动机械臂：

```bash
python3 block_pick_main.py \
  --target 1 \
  --dry-run \
  --show-rgb \
  --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/tag_yolo_grasp.yaml
```

ID2～ID4 将 `--target` 改为 `2～4`。若任一类别估距误差超过 20mm，先停止
机械臂测试，单独重新标定该模型的距离参数。

## 4. 分别示教四类预抓点

以 ID1 为例：检测后机械臂自动到物块前约 85mm，再在 RViz 中调整到近距离、
正对且未接触的预抓点 P，按 Enter 保存。

```bash
python3 block_pick_main.py \
  --target 1 \
  --teach-block-pregrasp \
  --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/tag_yolo_grasp.yaml \
  --preset-file /home/eaibot/handeye-calib/config/tag_yolo_pick_presets.json
```

依次把 `--target` 改为 `2、3、4`。这里只示教预抓点，不重新示教无 Tag 的
放置点、carry 或 idle。

## 5. 先停在 P 检查

此命令不开泵，也不需要动作 preset：

```bash
python3 block_pick_main.py \
  --target 1 \
  --stop-at-taught-pre-grasp \
  --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/tag_yolo_grasp.yaml \
  --preset-file /home/eaibot/handeye-calib/config/tag_yolo_pick_presets.json
```

## 6. 单类完整抓取

```bash
python3 block_pick_main.py \
  --target 1 \
  --run-taught-block \
  --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/tag_yolo_grasp.yaml \
  --preset-file /home/eaibot/handeye-calib/config/tag_yolo_pick_presets.json \
  --motion-preset-file /home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json
```

流程为：到 P 后方 30mm，开启限位，按 5mm 到 P，再按 2mm 最多前探
65mm；触发后开泵，直退到 P 后方 30mm，再走无 Tag carry、同 ID 放置点、
关泵、上退 50mm并回无 Tag idle。

## 7. 四类一键连续抓取

```bash
python3 block_pick_main.py \
  --run-chassis-sequence \
  --sequence 1,2,3,4 \
  --max-targets 4 \
  --result-file /tmp/tag_yolo_pick_result.json \
  --show-rgb \
  --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/tag_yolo_grasp.yaml \
  --preset-file /home/eaibot/handeye-calib/config/tag_yolo_pick_presets.json \
  --motion-preset-file /home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json
```

连续抓取按画面中最左侧剩余 ID 选择目标，中途不等待 Enter。运行结束后查看：

```bash
cat /tmp/tag_yolo_pick_result.json
```

四类全部成功时结果为：

```json
{ "completed_ids": [1, 2, 3, 4] }
```

测试本流程时不要同时运行旧 AprilTag 检测/抓取程序，也不要从正式比赛
`main.py --tag-pick` 启动；当前版本故意保持两套入口隔离。
