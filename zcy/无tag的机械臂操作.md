# 无 Tag 机械臂抓取操作

本方案只使用 Astra 的 RGB 图像，不读取深度图。定位流程为：

```text
矫正 RGB -> YOLOv5 ONNX -> 鲜帧多帧过滤 -> 单目估距
-> 手眼 TF 转到 base -> 固定吸盘姿态/接近方向 -> 示教偏移抓取 -> 固定点放置
```

当前检测模型为 `block_occlusion_yolov5n_640_best.onnx`，固定输入 `640x640`；配置文件中的 `model_path` 和 `input_size` 必须同时保持一致。

四类物块沿用有 Tag 放置点的编号对应关系：

| 编号 | 英文名称  | 中文         |
| ---: | --------- | ------------ |
|  `1` | `power`   | 应急电源     |
|  `2` | `fire`    | 灭火装置     |
|  `3` | `gas`     | 气体净化装置 |
|  `4` | `support` | 结构支撑装置 |

所有 `--target` 命令都可以直接写数字，例如 `--target 3` 等价于 `--target gas`；原英文写法仍然兼容。

## 2. 必须启动的终端

### 终端 0：底盘

````bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/robocom_ws/devel/setup.bash
roslaunch xpkg_bringup bringup_basic_ctrl.launch
### 终端 1：RGB 相机

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
roslaunch astra_camera astrapro.launch
````

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
stty -ixon
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
stty -ixon
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
  --target 2 \
  --dry-run \
  --show-rgb \
  --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml
```

程序只接受新图像时间戳，并用中位数/MAD 剔除跳变帧。目标框中心还必须进入画面的红色 ROI；定位不稳定或在 ROI 外时不会让机械臂抓取。

## 5. 标定纯 RGB 距离

当前实际抓取距离约为 `400mm`，新标定围绕工作区加密采样：

- 采集点：`340、360、370、380、390、400、410、420、430、440、460mm`
- `360~440mm` 每 `10mm` 一点，覆盖实际抓取波动；两端各留约 `60mm` 余量。
- 每类、每个距离采集 `10` 帧，共 `110` 帧/类。
- 距离是 RGB 镜头光心到物块正面的垂直 Z 距离；物块正面应与相机尽量平行。

重新采集期间保持配置为 `distance_method: theory`。这会阻止旧模型参数驱动机械臂，但不影响距离采集。运行：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
conda activate ww
cd /home/eaibot/handeye-calib/src

python3 /home/eaibot/handeye-calib/src/block_distance_collect.py \
  --targets power,fire,gas,support \
  --distances 340,360,370,380,390,400,410,420,430,440,460 \
  --frames 10 \
  --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml \
  --output-dir /home/eaibot/handeye-calib/config/block_distance_samples_occlusion640_400
```

每次只摆一个类别，框中心放进红色 ROI。程序按同一距离依次采集四类，再提示调整下一个距离。不要加 `--overwrite`；重复启动会跳过新目录里已经完成的 CSV。

旧数据完整保留在：

```text
/home/eaibot/handeye-calib/config/block_distance_samples/
```

本轮新数据和拟合结果单独写入：

```text
/home/eaibot/handeye-calib/config/block_distance_samples_occlusion640_400/
```

每类完成后会自动生成 `power_model.yaml`、`fire_model.yaml` 等。将其中的 `width/height` 参数填入 `block_mono_grasp.yaml` 对应类别，四类全部更新后再修改：

```yaml
distance_method: calibrated
```

本轮参数已在 `2026-08-08` 写入配置并启用 `calibrated`。其中 gas 的宽/高拟合 RMSE 分别为 `11.86mm/13.72mm`，最大残差达到 `27.56mm`，必须通过下面的独立距离复核才能用于运动。

不要用拟合点本身验证。分别把物块放在 `350、405、450mm`，按下一节执行 `--dry-run --show-rgb`，比较终端打印距离与实测距离。三个验证点都正常后，才能开始机械臂示教或抓取。

当前 `max_axis_distance_disagreement_mm: 0.0`，表示不再因宽、高两轴估距差异阻断抓取，最终距离取两者中位数。

当前 ROI 配置为：

```yaml
grasp_roi_ratio: [0.06, 0.00, 0.24, 1.00]
```

在 `640x480` 图像中约对应 `x=38~154` 的红色竖框。距离采样、dry-run、示教和正式抓取共用这一范围。

示教和抓取与有 Tag 链路一样，确认 `5` 个新鲜稳定观测后立即继续；距离标定仍使用每点 `10` 帧。

## 6. 放置点和抓取点示教

### 6.1 连续示教四个放置点

不写 `--target`，按 `1、2、3、4` 连续示教。程序每完成一个就立即保存；中途退出不会丢失已经完成的结果，也不会删除尚未示教类别的旧值。

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
conda activate ww
cd /home/eaibot/handeye-calib/src
stty -ixon

python3 block_pick_main.py \
  --teach-block-place \
  --sequence 1,2,3,4 \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml \
  --preset-file /home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json
```

每一步都在 RViz 中把末端移动到对应载物仓的释放姿态，执行 `Plan/Execute`，确认到位后回终端按 Enter。顺序为 `1=power、2=fire、3=gas、4=support`。

### 6.2 单独示教一个放置点

下面示例只示教 `3=gas`；把数字改为 `1~4` 即可。已有值在本次确认记录后自动替换，不需要 `--overwrite`。

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
conda activate ww
cd /home/eaibot/handeye-calib/src
stty -ixon

python3 block_pick_main.py \
  --target 3 \
  --teach-block-place \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml \
  --preset-file /home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json
```

### 6.3 单类抓取点示教

无 Tag 的放置点初始值可沿用有 Tag 数据，但两份 JSON 不会自动同步。抓取点必须逐类示教。正式抓取强制要求 preset 中存在 `carry_joint_values`，缺失时会在机械臂运动和开泵前报错，不再静默跳过过渡点。示教程序先保持当前末端姿态，自动移动到检测目标表面前约 `110mm`；然后通过 RViz `Plan/Execute` 移动到吸盘接触姿态，不能手掰。

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
conda activate ww
cd /home/eaibot/handeye-calib/src

python3 block_pick_main.py \
  --target 1 \
  --teach-block-grasp \
  --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml \
  --preset-file /home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json
```

已有类别可以直接重复运行同一条示教命令，不需要 `--overwrite`。程序只会在新示教完整成功并确认记录后替换该类别旧抓取偏移；移动失败或中途退出时旧值不变。除非确实要重新定义四类共用的吸盘姿态，否则不要再加 `--reset-pickup-model`。

## 7. 单类验证和抓取

1 = power 应急电源
2 = fire 灭火装置
3 = gas 气体净化装置
4 = support 结构支撑装置
确认预抓方向正确后，正式抓取并放置：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
conda activate ww
cd /home/eaibot/handeye-calib/src

python3 block_pick_main.py \
  --target 1 \
  --run-taught-block \
  --pregrasp-distance-mm 70 \
  --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml \
  --preset-file /home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json
```

每类均按“示教 → 预览 → 80mm 预抓点 → 正式抓放”的顺序验证。预抓方向不正确时不要执行正式抓取。

## 8. 底盘缓慢前进并逐个抓取

该入口移植有 Tag 的自动流程：从当前可见的剩余目标中选择画面最靠左的一个，底盘低速移动到红框，立即停车，等待底盘稳定并确认 `4` 个新鲜检测帧，然后执行“预抓 → 抓取 → 中转点 → 放置 → idle → `$H` 回零”。处理完后再选择下一个目标。检测不到、检测框异常或定位失败时底盘会停车，不会盲目前进。

运行前必须已经完成四类抓取示教，并确认无 Tag preset 同时具有 `pickup_model`、四类抓取/放置数据、`carry_joint_values` 和 `idle_joint_values`。程序会在底盘首次移动前统一检查这些字段。

逐个对准并抓取 `1~4`，每个完成后按 Enter 再继续：

按 Enter 后如果剩余物块尚未出现在画面中，程序会保持底盘停止并持续等待，不会因等待超时退出；将物块放入画面即可继续，需要终止时按 Ctrl+C。

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/robocom_ws/devel/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
conda activate ww
cd /home/eaibot/handeye-calib/src

python3 block_pick_main.py \
  --run-chassis-sequence \
  --sequence 1,2,3,4 \
  --wait-key-between-targets \
  --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml \
  --preset-file /home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json
```
