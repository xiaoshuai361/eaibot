# 无 Tag 机械臂抓取操作

本方案只使用 Astra 的 RGB 图像，不读取深度图。定位流程为：

```text
矫正 RGB -> YOLOv5 ONNX -> 鲜帧多帧过滤 -> 单目估距
-> 手眼 TF 转到 base -> 固定吸盘姿态/接近方向 -> 示教偏移抓取 -> 固定点放置
```

当前检测模型为 `block_occlusion_yolov5n_640_best.onnx`，固定输入 `640x640`；配置文件中的 `model_path` 和 `input_size` 必须同时保持一致。

退出 `block_pick_main.py` 时优先在预览窗口按 `q`/`Esc`，或在终端按
`Ctrl+C`。程序也会把 `Ctrl+Z` 当作受控退出并清理 Python2 ROS 子进程，避免
示教子进程留在后台继续占用 `/tmp/mirobot_arm_motion.lock`。升级前遗留的后台
进程仍需先按 PID 手动终止一次；不要只删除锁文件。

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

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/robocom_ws/devel/setup.bash
roslaunch xpkg_bringup bringup_basic_ctrl.launch
```

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

## 6. 抓取和放置示教

无 Tag 四类物块尺寸和抓取位置差异较大，因此必须分别示教。每次命令只处理一个类别，并在同一次流程中依次采集该类别的“预抓点 P → 无 Tag 入仓放置点”。示教程序先自动移动到检测目标表面前约 `85mm`；如果该辅助移动执行失败，会像有 Tag 流程一样报警但继续示教。随后在 RViz 中 `Plan/Execute` 到靠近、正对但未接触物块的预抓点 P；不要手掰机械臂。

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
conda activate ww
cd /home/eaibot/handeye-calib/src

python3 block_pick_main.py \
  --target 4 \
  --teach-block-pick-place \
  --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml \
  --preset-file /home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json
```

`--target` 分别使用 `1、2、3、4`，四条命令都要执行。每条命令的交互顺序固定为：

```text
稳定检测指定物块
→ 自动到检测表面前85mm
→ RViz微调并确认该类别预抓点P
→ 保持机械臂停在P
→ 从P开始在RViz移动到该类别无Tag载物仓放置点
→ 确认后同时保存该类别的完整预抓模型和放置位姿
```

每个类别分别保存 `pregrasp_offset_xyz_base`、`pickup_model` 和
`place_ee_in_base`。`pickup_model` 包含该类别实际调整后的 Link6 姿态和限位
前进轴，不强制使用 `(0,0,0,1)`。只有预抓点和放置点都成功采集后才替换该
类别旧数据；中途退出不会写入半套数据。

也可以分开重采。只重采该类别预抓点并保留原放置点：

```bash
python3 block_pick_main.py \
  --target 3 \
  --teach-block-pregrasp \
  --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml \
  --preset-file /home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json
```

只重采该类别放置点并保留原预抓数据：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
conda activate ww
cd /home/eaibot/handeye-calib/src
python3 block_pick_main.py \
  --target 4 \
  --teach-block-place \
  --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml \
  --preset-file /home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json
```

只采放置点时，程序会先根据当前检测和该类别已保存的数据，普通规划到 P 后方
`30mm`，再直线到 P；此示教移动不开泵。到 P 后再由你在 RViz 中调整到无 Tag
放置点。该入口要求此类别已经有完整预抓数据。

无 Tag 入仓放置位姿与有 Tag 完全分开，保存在无 Tag preset 的对应类别中；
重新示教无 Tag 不会修改有 Tag preset。后续投递从载物仓取物的关节角仍与有
Tag 共用 `delivery_presets.json`，不需要重新示教投递仓内抓取点。

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
  --target 4 \
  --run-taught-block \
  --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml \
  --preset-file /home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json
```

四类示教完成后逐类验证。有 Tag 和无 Tag 正式抓取安全时序统一为“普通规划到示教预抓点 P 后方 30mm → 在这里开启限位 → 以 5mm 步长保持姿态受保护地直线到 P，途中触发立即停止 → 未触发再从 P 以 2mm 步长最多前探 65mm → 触发后开泵 → 沿原路径直线退到 P 后方 30mm → carry/放置 → idle”。两者的检测方法、预抓参数和入仓放置位姿分别读取各自数据；只共用限位安全时序。退回成功前不会执行斜向搬运规划。预抓方向不正确时不要执行正式抓取。

## 8. 底盘缓慢前进并逐个抓取

该入口移植有 Tag 的自动流程：从当前可见的剩余目标中选择画面最靠左的一个，底盘低速移动到红框，立即停车，等待底盘稳定并确认 `4` 个新鲜检测帧，然后执行“预抓 → 抓取 → 中转点 → 放置 → idle → `$H` 回零”。处理完后再选择下一个目标。检测不到、检测框异常或定位失败时底盘会停车，不会盲目前进。

运行前必须分别完成四类“预抓点 + 放置点”示教，并确认无 Tag preset 的四个
类别各自具有 `pickup_model`、`pregrasp_offset_xyz_base` 和
`place_ee_in_base`，顶层仍具有 `carry_joint_values` 和 `idle_joint_values`。

一键连续对准并抓取 `1~4`，中途不等待 Enter。若剩余物块尚未出现在画面中，程序会保持底盘停止并持续等待；目标进入画面后自动继续，需要终止时按 Ctrl+C。

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
  --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml \
  --preset-file /home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json
```

## 9. 楼宇磁吸投递标定

这一节标定的是“从载物仓取出物资并贴到楼宇”，与第 5 节物块抓取距离标定、
第 6 节抓取后放入载物仓的示教不是一回事。楼宇投递需要完成两类数据：

```text
楼宇视觉距离标定：building_delivery_calibration.json
四类机械臂投递点P：untagged_delivery_presets.json
```

### 9.1 先备份真机数据

```bash
cp /home/eaibot/handeye-calib/config/untagged_delivery_presets.json \
  /home/eaibot/handeye-calib/config/untagged_delivery_presets.json.bak

cp /home/eaibot/handeye-calib/config/building_delivery_calibration.json \
  /home/eaibot/handeye-calib/config/building_delivery_calibration.json.bak
```

第二个文件首次标定时可能还不存在，此时跳过第二条备份命令。Windows 仓库中的
文件不能覆盖真机生成的 JSON。

当前框宽单模型格式为 `"version": 3`。如果原 JSON 是 `version 1/2`，备份后
把它移开，再重新完成四类标定。已采集的 CSV 可以继续用：

```bash
grep '"version"' /home/eaibot/handeye-calib/config/building_delivery_calibration.json
# 只有不是 version 3 时执行：
mv /home/eaibot/handeye-calib/config/building_delivery_calibration.json \
  /home/eaibot/handeye-calib/config/building_delivery_calibration_old.json
```

### 9.2 标定四类楼宇的真实距离

楼宇与物资 ID 对应关系：

原理与第 5 节无 Tag 物块的纯 RGB 距离标定相同。针对每类楼宇，在多个已知真实
距离采集 YOLO 框，只用左右边界的宽度拟合：

```text
Z = a_width / (右边界x - 左边界x) + b_width
```

机械臂 P 的示教参考距离为 `450mm`，默认标定范围从 `350mm` 开始：

```text
350、400、410、420、430、440、450、460、470、480、490、500、550、600mm
```

`400~500mm` 每 `10mm` 加密，每类、每个距离采集 `5` 帧，共 `70` 帧/类、
四类合计 `280` 帧。脚本按“距离优先”工作：每到一个距离，依次采集电力故障、
火灾、有毒气体、坍塌四类楼宇，再进入下一个距离。整套标定只运行一次命令。
真实距离是 `/dev/video2` RGB 镜头平面到楼宇正面的垂直距离。每次测量时车辆和
摄像机保持正对楼面。YOLO 框左、右边界必须完整；上、下边界被画面
裁切可以继续标定，因为距离计算完全不使用框高或面积。
标定窗口会画出全高度红框，原始 `320×240` 画面对应 `x=54~173`（归一化
`0.17~0.54`）。必须把绿色楼宇检测框的中心点移入红框，程序才会计入有效样本；
正式循迹投递也使用完全相同的红框作为停车条件。

该标定是独立的纯视觉程序，只需要下面这一个终端。**不需要**提前启动底盘、
MoveIt、RViz、Astra、手眼 TF、机械臂节点或比赛主程序。运行前必须关闭
`zcy_last.main`、楼宇预览以及其他占用 `/dev/video2` 的程序；不确定时可先检查：

```bash
fuser /dev/video2
```

没有任何输出表示摄像头空闲；若输出 PID，应先正常退出对应程序，不要直接删除
文件或强行清理机械臂锁。确认 `/dev/video2` 空闲后运行：

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

程序启动后立即打开检测窗口，持续显示 YOLO 绿框和标定红框。保持当前
距离不变，在终端或检测窗口按 Enter，程序自动记录当前“距离 + 楼宇类别”
的 `5` 帧；然后换同一距离的下一类。四类完成后再调整到下一距离。`s` 跳过，
`q` 退出；已有组合会自动跳过。只有左右边界被裁切时才不能采样。

全部四类模型都成功拟合后才一次性原子更新 JSON。原始 CSV 和最终模型分别为：

```text
/home/eaibot/handeye-calib/config/building_delivery_distance_samples_building_new_320x240/
/home/eaibot/handeye-calib/config/building_delivery_calibration.json
```

如果实际画面不是 `320×240`、检测不到指定类别或模型文件发生变化，程序会拒绝
保存。旧的 `version: 1/2` 文件不能继续使用，必须备份后重新完成四类
多距离标定；不要手工修改 JSON 绕过检查。

### 9.3 示教四类机械臂预投递点 P

楼宇距离标定完成后，示教复用无 Tag 抓取的完整定位链路：Astra 矫正 RGB、
`CameraInfo`、四类楼宇各自的框宽距离模型以及已经发布的手眼 TF。车辆应停在
约 `450mm` 的参考工作距离并正对楼面；程序会根据当前检测结果计算真实距离，
不是把 `450mm` 写死成机械臂目标。

先关闭距离标定、`zcy_last.main` 及其他机械臂动作程序。然后分别启动以下终端。

终端1，启动 Astra：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
roslaunch astra_camera astrapro.launch
```

终端2，启动 MoveIt 和 RViz：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
roslaunch mirobot_moveit_config mirobot.launch start_rviz:=true
```

终端3，发布已有手眼标定 TF：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
roslaunch easy_handeye publish.launch \
  eye_on_hand:=false \
  tracking_base_frame:=camera_link
```

终端4，持续显示楼宇检测框。该预览订阅 Astra 的 ROS 图像话题，可以和示教
进程同时工作：

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
  --model /home/eaibot/handeye-calib/src/model/yolov5/building_new_yolov5n_320_best.onnx \
  --config /home/eaibot/handeye-calib/src/config/building_delivery_teach.yaml
```

| ID | 楼宇类别 |
| --: | ------------ |
| 1 | 电力故障楼宇 |
| 2 | 火灾楼宇 |
| 3 | 有毒气体楼宇 |
| 4 | 坍塌楼宇 |

终端5，每类单独示教。下面以火灾楼宇 ID2 为例；将 `--target` 改为 `1~4`
分别执行四次：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
conda activate ww
cd /home/eaibot/handeye-calib/src

python3 block_pick_main.py \
  --target 2 \
  --teach-building-contact-release \
  --confidence 0.5 \
  --model /home/eaibot/handeye-calib/src/model/yolov5/building_new_yolov5n_320_best.onnx \
  --config /home/eaibot/handeye-calib/src/config/building_delivery_teach.yaml \
  --preset-file /home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json \
  --delivery-file /home/eaibot/handeye-calib/config/untagged_delivery_presets.json \
  --overwrite
```

单类实际流程如下：

```text
1. 自动移动到无 Tag idle，先让开摄像头视野
2. 连续取得5帧稳定楼宇检测
3. 将640宽图像中的楼宇框宽换算到标定时的320宽坐标
4. 用该类框宽距离模型求Z，再用CameraInfo和手眼TF求楼面点的base坐标
5. 自动移动到楼面前约40mm的辅助点
6. 在RViz中从该点微调到靠近、正对但不接触楼面的P，按Enter保存
```

程序保存 P 的六关节角，并由手眼 TF 的相机正前方向自动生成安全退回方向；不再
要求手动示教后方参考点。检测框左右边界必须完整，上下边被裁切允许继续。四类
楼宇的 P 独立保存，因此四类都要各示教一次。已有该类数据时必须显式传
`--overwrite` 才会替换。示教完成后可在预览窗口按 `q` 或 `Esc` 退出。

### 9.4 单类空载验证接触投递动作

先确认共享仓内抓取点、无 Tag 中转点以及对应 ID 的 P 都已经示教。验证时可以不
在载物仓放物资，让吸泵空吸，只检查运动、限位和退回顺序：

```bash
python2 /home/eaibot/handeye-calib/src/mirobot_delivery.py \
  --mode run_delivery \
  --sequence 2 \
  --delivery-file /home/eaibot/handeye-calib/config/untagged_delivery_presets.json \
  --cargo-pick-file /home/eaibot/handeye-calib/config/delivery_presets.json \
  --tag-preset-file /home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json \
  --contact-release \
  --force-release-on-contact-miss \
  --contact-staging-gap 0.030 \
  --contact-staging-step 0.005 \
  --contact-probe-step 0.002 \
  --contact-probe-max-travel 0.065
```

正确时序为：

```text
取对应仓内点 -> 共享中转点 -> 普通规划到P后方30mm
-> 开启限位 -> 5mm步长到P -> 2mm步长最多前探65mm
-> 关泵并等待0.7秒 -> 沿原方向直退30mm -> idle
```

正常走满 `65mm` 仍未触发限位时，会打印醒目警告并按已确认策略强制关泵，进程
仍返回成功。限位服务、串口或轨迹执行报错不属于“正常未触发”：此时不会强制
释放，吸泵保持开启并返回失败，需要人工处理。

## 10. 正式循迹自动投递

完成四类楼宇距离标定和四类 P 示教后，正式任务使用：

```bash
cd /home/eaibot/robocom_ws/src
python3 -m zcy_last.main \
  --no-tag-pick \
  --untagged-pick --untagged-pick-count 4 --untagged-delivery
```

底盘继续按原循迹前进，不做楼宇单独旋转或距离闭环。楼宇 YOLO 框中心进入红色
竖框 `x=54~173` 时，原 `YOLO_STOP` 逻辑立即停车。停车框的左右宽度模型只用于
估算镜头到楼面的真实距离，不参与停车判断。

机械臂以 `450mm` 时示教的 P 为基准，按“停车估距 - 450mm”沿已示教前探轴修正
P。例如停车估距为 `550mm`，P 沿墙面方向前移 `100mm`；随后仍执行“P 后方
30mm -> 5mm步长到P -> 2mm步长最多前探65mm”。左右边界被裁切，或框宽估距超出
已采集的 `350~600mm` 时不启动机械臂，该 ID 记为投递失败。上下边界裁切不影响估距。
