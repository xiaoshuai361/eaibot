# 无 Tag 机械臂抓取操作

本方案只使用 Astra 的 RGB 图像，不读取深度图。定位流程为：

```text
矫正 RGB -> YOLOv5 ONNX -> 鲜帧多帧过滤 -> 单目估距
-> 手眼 TF 转到 base -> 固定吸盘姿态/接近方向 -> 示教偏移抓取 -> 固定点放置
```

## 1. 真机需要同步的文件

WSL 文件复制到机器人本机相同的 `/home/eaibot` 目录结构：

```text
/home/eaibot/handeye-calib/src/block_pick_main.py
/home/eaibot/handeye-calib/src/block_mono_vision.py
/home/eaibot/handeye-calib/src/mirobot_pick_test.py
/home/eaibot/handeye-calib/src/block_distance_collect.py
/home/eaibot/handeye-calib/src/block_distance_calibrate.py
/home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml
/home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json
```

模型保留在：

```text
/home/eaibot/handeye-calib/src/model/yolov5/Block_v5n_yolov5n_640_best.onnx
```

仓位放置姿态、抓取后的中间过渡点和 idle 都沿用有 Tag preset。无 Tag 只需要重新示教抓取；真机使用前要确认无 Tag preset 已包含从有 Tag preset 复制的 `carry_joint_values`。

## 2. 必须启动的终端

### 终端 1：RGB 相机

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
roslaunch astra_camera astrapro.launch
```

本方案需要 `/camera/rgb/image_rect_color`，不使用任何 depth 话题。

### 终端 2：机械臂和 MoveIt

示教时打开 RViz：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
roslaunch mirobot_moveit_config mirobot.launch start_rviz:=true
```

比赛运行时可关闭 RViz：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash

roslaunch mirobot_moveit_config mirobot.launch start_rviz:=false
```

### 终端 3：发布手眼标定 TF

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
roslaunch easy_handeye publish.launch eye_on_hand:=false tracking_base_frame:=camera_link
```

### 终端 4：无 Tag 实时检测画面

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
conda activate ww
cd /home/eaibot/handeye-calib/src
python3 block_pick_main.py \
  --live-preview \
  --preview-hz 1.0 \
  --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml
```

该终端只负责持续显示，不控制机械臂。没有检测结果时终端保持安静；每个类别在本次运行中第一次出现时打印一行，例如 `检测到：power POW91`，之后不逐帧重复。`POW/FIR/GAS/SUP` 分别表示电力、消防、气体净化、支撑，末尾数字是置信度百分比。按 `q` 或 `Esc` 退出画面。默认 `1Hz` 是为了给抓取终端留出推理资源。

### 终端 5：无 Tag 抓取命令环境

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
conda activate ww
cd /home/eaibot/handeye-calib/src
```

## 3. 启动后检查

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash

rostopic hz /camera/rgb/image_rect_color
rostopic echo -n1 /camera/rgb/camera_info
rosrun tf tf_echo base camera_rgb_optical_frame
```

必须满足：

- 矫正图持续发布，分辨率为 `640x480`。
- `CameraInfo` 的 `K`/`P` 中 `fx`、`fy` 大于 0。
- `base` 到相机光学坐标系的 TF 连通。

## 4. 先做纯视觉测试

检测全部物块，不动机械臂：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
conda activate ww
cd /home/eaibot/handeye-calib/src
python3 block_pick_main.py \
  --dry-run \
  --show-rgb \
  --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml
```

检测指定物块：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
conda activate ww
cd /home/eaibot/handeye-calib/src
python3 block_pick_main.py \
  --target fire \
  --dry-run \
  --show-rgb \
  --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml
```

程序只接受新图像时间戳，并用中位数/MAD 剔除跳变帧。目标框中心还必须进入画面的红色 ROI；定位不稳定或在 ROI 外时不会让机械臂抓取。

## 5. 标定纯 RGB 距离

初次联调可以保留配置中的：

```yaml
distance_method: fixed_plane
fixed_z_mm: 330.0
```

这只适合固定车距。正式标定使用引导采集程序，一次完成四类物块在 `280、300、320、340、360、380、400、420、440、460、480mm` 的采样，每个距离采集 10 帧：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
conda activate ww
cd /home/eaibot/handeye-calib/src

python3 /home/eaibot/handeye-calib/src/block_distance_collect.py \
  --targets power,fire,gas,support \
  --distances 280,300,320,340,360,380,400,420,440,460,480 \
  --frames 10 \
  --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml
```

程序按距离优先处理：先在 `280mm` 依次采集 `power/fire/gas/support`，四个完成后再改到 `300mm`，直到 `480mm`。每一步都会提示：

```text
1. 只摆放当前提示的物块。
2. 测量 RGB 镜头到物块正面的真实距离。
3. 在终端 4 确认物块框中心进入红色 ROI。
4. 保持物块正面与相机近似平行，回采集终端按 Enter。
```

输入 `s` 跳过当前距离，输入 `q` 中止。默认是安全续采模式：再次运行会保留并跳过已有 CSV，只采缺失距离，最后用全部新旧数据重新拟合。已有 `280~380mm` 数据时直接运行上面的命令，不要增加 `--overwrite`，程序会从 `400mm` 继续。只有确认要丢弃计划内旧采样并全部重采时才增加 `--overwrite`。

相机推理速度较低时，10 帧可能需要约 15~30 秒。程序最多等待 50 秒，采满 10 帧会立即继续，不会固定等满 50 秒；同一距离的框宽和框高分别取中位数，不使用容易被单帧异常影响的简单平均值。

采样和拟合结果保存在：

```text
/home/eaibot/handeye-calib/config/block_distance_samples/
```

每类完成后会自动生成 `power_model.yaml`、`fire_model.yaml` 等。将其中的 `width/height` 参数填入 `block_mono_grasp.yaml` 对应类别，四类完成后修改：

```yaml
distance_method: calibrated
```

当前真机已完成 `280~480mm` 的 11 点标定；拟合参数已经写入 WSL 侧配置。同步最新 `block_mono_grasp.yaml` 到真机后，先按下一节执行 `--dry-run --show-rgb` 距离复核，再进行机械臂示教或抓取。

若宽、高估算距离相差超过 `20mm`，程序会认为框不完整或视角不合格并拒绝抓取。

当前 ROI 配置为：

```yaml
grasp_roi_ratio: [0.06, 0.00, 0.24, 1.00]
```

在 `640x480` 图像中约对应 `x=38~154` 的红色竖框。距离采样、dry-run、示教和正式抓取共用这一范围。

示教和抓取与有 Tag 链路一样，确认 `5` 个新鲜稳定观测后立即继续；距离标定仍使用每点 `10` 帧。

## 6. 单类抓取示教

四类的放置点、中间过渡点和 idle 沿用有 Tag 数据，只需逐个示教抓取点。示教程序不会自动靠近物块，也不会自动改变 joint5；只摆当前类别，框中心放进红色 ROI，然后完全通过 RViz `Plan/Execute` 移动到吸盘接触姿态，不能手掰。

第一次可用 `power` 示教，并锁定四类共用的吸盘姿态和接近方向：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
conda activate ww
cd /home/eaibot/handeye-calib/src

python3 block_pick_main.py \
  --target power \
  --teach-block-grasp \
  --reset-pickup-model \
  --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml \
  --preset-file /home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json
```

之后分别示教 `fire、gas、support`。下面以 `fire` 为例，其他类别只替换 `--target`：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
conda activate ww
cd /home/eaibot/handeye-calib/src

python3 block_pick_main.py \
  --target fire \
  --teach-block-grasp \
  --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml \
  --preset-file /home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json
```

已有类别需要重教时，在命令末尾增加 `--overwrite`。不要再加 `--reset-pickup-model`，否则会重置四类共用的吸盘姿态。

## 7. 单类验证和抓取

以下以 `fire` 为例，其他类别只替换 `--target`。

只计算抓取点，不移动机械臂：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
conda activate ww
cd /home/eaibot/handeye-calib/src

python3 block_pick_main.py \
  --target fire \
  --preview-taught-block \
  --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml \
  --preset-file /home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json
```

只移动到抓取点外 `100mm`，不接触、不开泵：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
conda activate ww
cd /home/eaibot/handeye-calib/src

python3 block_pick_main.py \
  --target fire \
  --stop-at-taught-pre-grasp \
  --pregrasp-distance-mm 100 \
  --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml \
  --preset-file /home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json
```

确认预抓方向正确后，正式抓取并放置：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
conda activate ww
cd /home/eaibot/handeye-calib/src

python3 block_pick_main.py \
  --target fire \
  --run-taught-block \
  --pregrasp-distance-mm 80 \
  --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml \
  --preset-file /home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json
```

每类均按“示教 → 预览 → 100mm 预抓点 → 正式抓放”的顺序验证。预抓方向不正确时不要执行正式抓取。
