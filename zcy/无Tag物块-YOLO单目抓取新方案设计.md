# 无 Tag 物块 YOLO 单目抓取新方案设计

## 1. 已确认决策

- 旧深度相机方案废弃：不再使用深度图读取 Z。
- 新方案使用 RGB 单目定位：YOLO 框中心确定方向，YOLO 框宽 `w` 估算距离 `Z`。
- 图像来源使用 ROS：
  - `/camera/rgb/image_raw`
  - `/camera/rgb/camera_info`
- 模型使用 ONNX：

```bash
/home/eaibot/handeye-calib/src/model/yolov5/Block_v5n_yolov5n_640_best.onnx
```

- 之前 ArUco / easy_handeye 手眼标定结果可以继续使用，用来做：

```text
相机坐标(Xc,Yc,Zc) → 机械臂base坐标(Xb,Yb,Zb)
```

- 新增需要解决的是：无 Tag 物块没有 AprilTag 位姿，所以必须通过 YOLO 框宽估算距离 `Z`。

## 2. 总体流程

```text
ROS RGB图像
→ ONNX YOLO识别无Tag物块
→ 多帧筛选稳定框
→ 得到框中心(u,v)和框宽w
→ 用框宽w估算距离Z
→ RGB相机内参算相机坐标(Xc,Yc,Zc)
→ 复用已有ArUco/easy_handeye外参转机械臂坐标(Xb,Yb,Zb)
→ 计算预抓取点
→ 机械臂正面接近
→ 开泵吸取
→ 后退
```

有 Tag 和无 Tag 的区别：

```text
有Tag：
AprilTag检测 → 直接得到相机里的3D位置/姿态 → 手眼转换 → 机械臂抓

无Tag：
YOLO检测框 → 用框宽w估距离Z → 相机内参算3D位置 → 手眼转换 → 机械臂抓
```

## 3. 旧代码归档

把旧深度方案相关代码放到：

```bash
/home/zcy/eaibot/handeye-calib/src/old/
```

归档文件：

```text
block_detector_protocol.py
block_grasp_sequence.py
block_grasp_vision.py
block_pick_main.py
block_yolo_preview.py
mirobot_pick_test.py
```

其中 `mirobot_pick_test.py` 先完整备份，再重写/清理无 Tag 部分，避免原始功能丢失。

## 4. 新代码结构

建议只保留三个核心文件：

```text
block_pick_main.py
block_mono_vision.py
mirobot_pick_test.py
```

### block_pick_main.py

Python3 主入口，在 `ww` 环境运行。

负责：

- 加载 ONNX 模型；
- 接收 Python2 子进程发来的 RGB 图片路径；
- 执行 YOLO 推理；
- 返回检测框、类别、置信度；
- 支持 dry-run、预抓取、完整抓取三种模式。

### block_mono_vision.py

纯视觉和数学工具。

负责：

- ONNX YOLO 前处理和后处理；
- 计算 `u, v, w, h`；
- 多帧稳定判断；
- 用 `w` 估算 `Z`；
- 用内参计算 `(Xc, Yc, Zc)`；
- 画 RGB 调试图；
- 预留不同的 Z 估算方法。

### mirobot_pick_test.py

Python2 ROS / MoveIt 机械臂执行文件。

负责：

- 订阅 ROS RGB 图像和 CameraInfo；
- 调用 Python3 YOLO；
- 从 TF 读取已有 easy_handeye 手眼标定结果；
- 把相机坐标转换为 base 坐标；
- dry-run 打印坐标；
- 执行预抓取或完整吸取。

## 5. Z 估算方法预留

配置项：

```yaml
distance_method: theory
```

后续支持：

```yaml
distance_method: theory
distance_method: calibrated
distance_method: fixed_plane
```

### theory：理论公式

用于前期 dry-run 测试：

```text
Z = fx * 30 / w
```

其中：

- `fx` 来自 `/camera/rgb/camera_info`
- `30` 是物块真实宽度 30 mm
- `w` 是 YOLO 框宽像素

该方法可以先用来查看坐标，但不建议直接真抓。

### calibrated：实测拟合

正式抓取推荐使用：

```text
Z = a / w + b
```

其中：

- `w` 是 YOLO 框宽像素；
- `Z` 是物块到相机的距离，单位 mm；
- `a,b` 由实测数据拟合得到。

配置预留：

```yaml
distance_models:
  power: {a: null, b: null}
  fire: {a: null, b: null}
  gas: {a: null, b: null}
  support: {a: null, b: null}
```

如果真实吸取时选择 `calibrated`，但目标类别没有 `a,b`，程序必须拒绝执行。

### fixed_plane：固定物资架平面

如果后续发现框宽估 Z 不稳定，可切换到固定平面方案：

```text
YOLO只负责图像方向/XY，Z由固定物资架平面或示教高度给出
```

该方法先预留接口，不作为第一版主方案。

## 6. 配置文件

新增：

```bash
/home/zcy/eaibot/handeye-calib/src/config/block_mono_grasp.yaml
```

建议内容：

```yaml
model_path: /home/eaibot/handeye-calib/src/model/yolov5/Block_v5n_yolov5n_640_best.onnx

target_classes:
  power:
    class_id: 0
    class_name: Emergency power supply device
  fire:
    class_id: 1
    class_name: Fire extinguishing device
  gas:
    class_id: 2
    class_name: Gas purification device
  support:
    class_id: 3
    class_name: Structural support device

target_size_mm: 30.0
distance_method: theory

frames_required: 10
confidence_min: 0.70
box_width_min_px: 30
box_aspect_ratio_min: 0.75
box_aspect_ratio_max: 1.30
center_std_max_px: 2.0
width_cv_max: 0.03

rgb_topic: /camera/rgb/image_raw
camera_info_topic: /camera/rgb/camera_info
camera_frame: camera_rgb_optical_frame
base_frame: base

distance_models:
  power: {a: null, b: null}
  fire: {a: null, b: null}
  gas: {a: null, b: null}
  support: {a: null, b: null}

pregrasp_distance_mm: 50.0
approach_distance_mm: 60.0
suction_compression_mm: 3.0
velocity_scale: 0.05
acceleration_scale: 0.05
max_retries: 2
```

## 7. 调试命令

### 只看 YOLO 和单目坐标，不动机械臂

```bash
python3 block_pick_main.py \
  --target fire \
  --dry-run \
  --show-rgb \
  --confidence 0.55 \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml
```

`--confidence` 用于临时覆盖配置文件里的 `confidence_min`；`--show-rgb` 会显示最后一帧检测图，按 `q` 或 `Esc` 退出。

终端打印：

```text
目标=fire
置信度=...
检测框=(x1,y1,x2,y2)
框中心=(u,v)
框宽px=w
distance_method=theory
单目距离Z_mm=...
相机坐标mm=(Xc,Yc,Zc)
机械臂坐标mm=(Xb,Yb,Zb)
```

### 只移动到预抓取点

```bash
python3 block_pick_main.py \
  --target fire \
  --stop-at-pre-grasp \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml
```

该模式会动机械臂，但不开泵、不接触物块。

### 完整吸取

```bash
python3 block_pick_main.py \
  --target fire \
  --execute \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml
```

完整吸取前必须检查：

- YOLO 检测稳定；
- 框宽足够；
- 能获取 RGB CameraInfo；
- 能获取 easy_handeye TF；
- 机械臂 base 坐标在安全范围内；
- 如果使用 `calibrated`，目标类别必须有有效 `a,b`。

## 8. 终端启动方式

### 定位 dry-run 最少需要

终端 1：Astra RGB 相机

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
roslaunch astra_camera astrapro.launch
```

终端 2：发布手眼标定结果

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
roslaunch easy_handeye publish.launch eye_on_hand:=false tracking_base_frame:=camera_link
```

终端 3：运行无 Tag 程序

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
conda activate ww
export PYTHONDONTWRITEBYTECODE=1
cd /home/eaibot/handeye-calib/src
python3 block_pick_main.py --target fire --dry-run --show-rgb
```

如果要机械臂运动，还需要终端 4：MoveIt 后端

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
roslaunch mirobot_moveit_config mirobot.launch start_rviz:=false
```

## 9. 标定数据采集

后面拟合 `a/w+b` 用：

```bash
python3 block_pick_main.py \
  --target fire \
  --calib-record \
  --known-z-mm 500 \
  --frames 30
```

它不动机械臂，只输出：

```text
known_z_mm,target,conf,x1,y1,x2,y2,u,v,w,h
```

在不同真实距离重复采集，例如：

```text
350, 400, 450, 500, 550, 600, 650, 700 mm
```

之后拟合每类的 `a,b`，写回配置。

## 10. 实现顺序

1. 归档旧深度方案代码到 `old/`。
2. 写 ONNX YOLO 推理和后处理。
3. 写 ROS RGB 单帧采集。
4. 写多帧稳定检测。
5. 写 `theory` 单目距离估算。
6. 复用 easy_handeye TF，把相机坐标转 base 坐标。
7. dry-run 显示 RGB 框并打印全部坐标。
8. 写标定记录功能，为拟合 `a,b` 留接口。
9. 写预抓取点运动。
10. 写完整吸取动作。

## 11. 安全边界

- dry-run 可以使用 `theory` 公式。
- 完整吸取默认不应该使用未验证的理论距离。
- 真实吸取推荐使用 `calibrated`，并要求目标类别有实测 `a,b`。
- 单目 Z 不稳定时，不靠固定补偿硬怼，应切换到 `fixed_plane` 或重新做距离标定。
- 任何超出工作空间的坐标不得下发给机械臂。
