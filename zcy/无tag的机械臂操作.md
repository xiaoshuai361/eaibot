# 无 Tag 机械臂操作

## 1. 当前固定逻辑

物资编号：

|  ID | 类别      | 物资         |
| --: | --------- | ------------ |
|   1 | `power`   | 应急电源     |
|   2 | `fire`    | 灭火装置     |
|   3 | `gas`     | 气体净化装置 |
|   4 | `support` | 结构支撑装置 |

无 Tag 抓取使用 Astra 矫正 RGB、YOLO、纯 RGB 估距和手眼 TF，不使用深度图。
四类物资分别保存预抓点和入仓放置点。

楼宇投递使用 `/dev/video2` 的 `320×240` 图像：

- 四类楼宇分别使用自己的框宽距离模型。
- 框中心进入红框 `x=54~173` 时停车，不旋转、不做底盘距离闭环。
- 四类楼宇共用 ID1（power）示教的机械臂预投递点 P。
- 视觉只按“楼宇估距 - 450mm”修正 P 的前后位置，不修改左右、高度和姿态。
- 到修正后 P 的后方30mm开启限位，5mm步长到P，再以2mm步长最多前探65mm。
- 释放后直退30mm，再回 idle。

真机数据只在机器人上生成，Windows 仓库文件不得覆盖：

```text
/home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json
/home/eaibot/handeye-calib/config/delivery_presets.json
/home/eaibot/handeye-calib/config/untagged_delivery_presets.json
/home/eaibot/handeye-calib/config/building_delivery_calibration.json
```

## 2. 无 Tag 抓取所需终端

### 终端1：底盘

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/robocom_ws/devel/setup.bash
roslaunch xpkg_bringup bringup_basic_ctrl.launch
```

### 终端2：Astra RGB

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
roslaunch astra_camera astrapro.launch
```

### 终端3：MoveIt/RViz

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
roslaunch mirobot_moveit_config mirobot.launch start_rviz:=true
```

正式运行不需要 RViz 时改为 `start_rviz:=false`。

### 终端4：手眼 TF

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
roslaunch easy_handeye publish.launch eye_on_hand:=false tracking_base_frame:=camera_link
```

### 命令终端环境

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
conda activate ww
cd /home/eaibot/handeye-calib/src
stty -ixon
```

检测预览（可选）：

```bash
python3 block_pick_main.py \
  --live-preview --preview-hz 1.0 --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml
```

按 `q`/`Esc` 或 `Ctrl+C` 退出。若旧进程仍占用
`/tmp/mirobot_arm_motion.lock`，必须根据报错 PID 结束旧进程，不能只删除锁文件。

## 3. 无 Tag 物资距离标定

只在需要重做抓取距离模型时运行：

```bash
python3 /home/eaibot/handeye-calib/src/block_distance_collect.py \
  --targets power,fire,gas,support \
  --distances 340,360,370,380,390,400,410,420,430,440,460 \
  --frames 10 \
  --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml \
  --output-dir /home/eaibot/handeye-calib/config/block_distance_samples_occlusion640_400
```

镜头到物块正面的距离必须准确，物块正面尽量与相机平行。每个距离依次采集四类；
重复运行会跳过已有 CSV，不要随意加 `--overwrite`。

## 4. 示教无 Tag 抓取和入仓放置

四类物资尺寸和位置不同，必须分别示教。下面示教 ID4；依次改为 `1~4`：

```bash
python3 block_pick_main.py \
  --target 4 \
  --teach-block-pick-place \
  --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml \
  --preset-file /home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json
```

流程：稳定检测 → 自动到物块前约85mm → RViz调好预抓点P → 从P移动到该类
无 Tag 入仓放置点 → 保存。程序保存6关节角、Link6姿态和前探方向，不强制四元数。

只重采预抓点：

```bash
python3 block_pick_main.py \
  --target 3 --teach-block-pregrasp --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml \
  --preset-file /home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json
```

只重采放置点：

```bash
python3 block_pick_main.py \
  --target 4 --teach-block-place --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml \
  --preset-file /home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json
```

无 Tag 入仓放置点与有 Tag 分开。投递时只有四个仓内抓取点共用
`delivery_presets.json`；楼宇投递中转点和共享 P 单独保存在
`untagged_delivery_presets.json`。

### 示教楼宇投递中转点

启动 MoveIt/RViz 后运行：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
cd /home/eaibot/handeye-calib/src

python2 /home/eaibot/handeye-calib/src/mirobot_delivery.py \
  --mode teach_transit \
  --delivery-file /home/eaibot/handeye-calib/config/untagged_delivery_presets.json \
  --overwrite
```

按提示在 RViz 中把机械臂移动到安全的携物中转姿态，再回终端按 Enter；当前
6个关节角会保存为 `transit_joint_values`。该点只供无 Tag 楼宇投递使用，必须
确保四个仓内抓取点和楼宇共享 P 都能安全到达，不会修改有 Tag 中转点。

## 5. 验证和连续抓取

单类验证（例：ID4）：

```bash
python3 block_pick_main.py \
  --target 4 --run-taught-block --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml \
  --preset-file /home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json
```

一键连续抓取四类，中途不需要按 Enter：

```bash
python3 block_pick_main.py \
  --run-chassis-sequence --sequence 1,2,3,4 --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml \
  --preset-file /home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json
```

抓取接触时序固定为：P后方30mm → 开限位 → 5mm步长到P → 2mm步长最多
前探65mm → 开泵 → 直退30mm → 入仓放置 → idle。

## 6. 标定楼宇距离

该程序独占 `/dev/video2`，运行前关闭 `zcy_last.main` 和其他楼宇预览。它不需要
MoveIt、Astra或手眼TF。建议先备份真机 JSON：

```bash
cp /home/eaibot/handeye-calib/config/untagged_delivery_presets.json \
  /home/eaibot/handeye-calib/config/untagged_delivery_presets.json.bak
cp /home/eaibot/handeye-calib/config/building_delivery_calibration.json \
  /home/eaibot/handeye-calib/config/building_delivery_calibration.json.bak
```

标定命令：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/robocom_ws/devel/setup.bash
conda activate ww
cd /home/eaibot/robocom_ws/src

python3 -m zcy_last.building_delivery_calibrate \
  --distances 350,400,410,420,430,440,450,460,470,480,490,500,550,600 \
  --frames 5 \
  --reference-distance-mm 450
```

每个距离依次采集四类楼宇。将绿色框中心移入红框后按 Enter；`s` 跳过，`q`
退出。只要求框左右边界完整，上下裁切不影响。结果写入：

```text
/home/eaibot/handeye-calib/config/building_delivery_calibration.json
```

## 7. 示教楼宇共享 P

车辆正对 power 楼宇，准确停在镜头距楼面 `450mm`。启动 Astra、MoveIt/RViz
和手眼 TF，然后只示教 ID1 一次：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
cd /home/eaibot/handeye-calib/src

python2 /home/eaibot/handeye-calib/src/mirobot_delivery.py \
  --mode teach_contact_release \
  --sequence 1 \
  --delivery-file /home/eaibot/handeye-calib/config/untagged_delivery_presets.json \
  --overwrite
```

程序不会自动移动机械臂。在 RViz 中将吸盘调到靠近、正对但不接触楼面的 P，
再回终端按 Enter。ID2～4正式投递时均读取这个 ID1 P，无需重复示教。

## 8. 空载验证楼宇接触投递

先启动 MoveIt。车辆放在镜头距楼面约 `450mm`，执行：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
cd /home/eaibot/handeye-calib/src

python2 /home/eaibot/handeye-calib/src/mirobot_delivery.py \
  --mode run_delivery \
  --sequence 1 \
  --delivery-file /home/eaibot/handeye-calib/config/untagged_delivery_presets.json \
  --cargo-pick-file /home/eaibot/handeye-calib/config/delivery_presets.json \
  --tag-preset-file /home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json \
  --contact-release \
  --force-release-on-contact-miss
```

该独立命令不运行楼宇 YOLO，距离修正默认为0，适合验证450mm示教姿态。测试其他
仓内物资只改 `--sequence 2/3/4`，投递 P 仍共用 ID1。

正确顺序：仓内抓取 → 中转点 → P后方30mm → 5mm到P → 2mm前探最多65mm
→ 关泵等待0.7秒 → 直退30mm → idle。

走满65mm未触发限位时会报警，但仍按要求强制释放并返回成功。限位服务、串口或
轨迹执行报错时保持泵开启并返回失败，必须人工处理。

## 9. 正式循迹抓取与投递

完成四类物资抓取示教、四类楼宇距离标定和一次共享 P 示教后运行：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/robocom_ws/devel/setup.bash
conda activate ww
cd /home/eaibot/robocom_ws/src

python3 -m zcy_last.main \
  --no-tag-pick \
  --untagged-pick --untagged-pick-count 4 \
  --untagged-delivery
```

正式投递会根据四类各自的楼宇框宽模型估距，并相对450mm修正共享 P。若检测框
左右边界被裁切，或估距超出已标定的350～600mm，则不启动机械臂，该ID记为失败。
