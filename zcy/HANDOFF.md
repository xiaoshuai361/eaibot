# 交接文档：Mirobot 机械臂抓取与 WSL 协作

写给完全没有上下文的新会话。

## 1. 当前协作环境

当前不是在 WSL 里直接跑真机。

- WSL 镜像目录：`/home/zcy/eaibot`
- 真机运行目录：`/home/eaibot`

工作方式：

1. 在 WSL 里读代码、改代码、做静态检查。
2. 改完后必须把对应文件同步回真机 `/home/eaibot/...`。
3. 机械臂、吸泵、Astra 相机、AprilTag、MoveIt 真机动作都在真机上验证，不要在 WSL 里假装已经验证真机。

重点参考文档：

- `/home/zcy/eaibot/zcy/WSL协作说明.md`
- `/home/zcy/eaibot/zcy/机械臂操作.txt`
- `/home/zcy/eaibot/zcy/记忆.md`

ROS 环境是 Melodic，抓取脚本必须用 Python 2：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
```

## 2. 我们在做什么任务

目标是让 Mirobot 机械臂更稳定地完成比赛物资抓取。

最近的核心需求有两个：

1. 启动机械臂 MoveIt 后端时可以不打开 RViz，降低比赛机负载。
2. 抓取时把吸盘从“默认朝下”调整成用户照片里确认过的“水平朝前”姿态，再用于后续正面/侧向接近物体。

补充：当前比赛物资 AprilTag 已切到 `tag16h5`，黑色码块外边长为 `1.45cm`，也就是 `size=0.0145` 米。

用户已经确认的目标腕部姿态来自真机：

```text
name:     [joint1, joint2, joint3, joint4, joint5, joint6]
position: [0.0, 0.0, 0.0002792526888193459, 0.0, -1.5709534265016345, 0.0]
```

最关键的是：

```text
joint5 = -1.5709534265016345
```

这个姿态就是用户想要的“吸盘水平朝前”样子。不要再误判为主要调 `joint6`；`joint6` 更多是绕吸盘自身轴旋转，不能把吸盘从朝下变成朝前。主要关节是 `joint5`。

## 3. 已经完成了什么

### 3.1 RViz 启动开关

已在 WSL 修改：

- `/home/zcy/eaibot/mirobot_ws/src/mirobot_moveit_config/launch/mirobot.launch`
- `/home/zcy/eaibot/mirobot_ws/src/mirobot_moveit_config/launch/mirobot_moveit.launch`

新增参数：

```bash
start_rviz:=false
```

用法：

```bash
roslaunch mirobot_moveit_config mirobot.launch start_rviz:=false
```

默认不加参数仍会打开 RViz：

```bash
roslaunch mirobot_moveit_config mirobot.launch
```

实现方式：

- `mirobot.launch` 定义并转发 `start_rviz`
- `mirobot_moveit.launch` 用 `if="$(arg start_rviz)"` 包住 `moveit_rviz.launch`

已做过 WSL 静态检查：

- XML 解析通过
- 参数转发断言通过

注意：改 launch 不需要重新 `catkin_make`。

### 3.2 抓取脚本新增 wrist_forward

已在 WSL 修改：

- `/home/zcy/eaibot/handeye-calib/src/mirobot_pick_test.py`
- `/home/zcy/eaibot/zcy/机械臂操作.txt`

新增常量：

```python
WRIST_FORWARD_JOINT5 = -1.5709534265016345
```

新增模式：

```bash
--mode wrist_forward
```

单独把吸盘转成水平朝前姿态：

```bash
python2 /home/eaibot/handeye-calib/src/mirobot_pick_test.py --mode wrist_forward --velocity-scale 0.05 --acceleration-scale 0.05 --planning-time 8.0
```

新增抓取参数：

```bash
--wrist-forward
--wrist-forward-joint5
```

抓取前先转腕部，再继续抓取：

```bash
python2 /home/eaibot/handeye-calib/src/mirobot_pick_test.py --mode grasp --supply basic --skip-home --wrist-forward --planning-time 8.0 --velocity-scale 0.1 --acceleration-scale 0.1 --grasp-x -0.045 --y-offset 0.025 --z-offset 0.08 --approach-axis z --approach-gap 0.02
```

脚本行为：

- `--mode wrist_forward`：只设置 `joint5`，不打开吸泵。
- `--wrist-forward`：在 `grasp / pick_place / pick_lift_place` 计算抓取目标前，先执行 `go_wrist_forward()`。
- `--dry-run --wrist-forward` 同时使用时不会实际转腕部，会打印 warning；因为 dry-run 不执行真机动作。

已做过 WSL 检查：

```text
parse wrist_forward OK
go_wrist_forward joint target OK
python3 -m py_compile /home/zcy/eaibot/handeye-calib/src/mirobot_pick_test.py 通过
```

### 3.3 AprilTag 检测切换到 tag16h5

已在 WSL 修改：

- `/home/zcy/eaibot/mirobot_ws/src/apriltag_ros/apriltag_ros/config/settings.yaml`
- `/home/zcy/eaibot/mirobot_ws/src/apriltag_ros/apriltag_ros/config/tags.yaml`

当前配置：

```yaml
tag_family: 'tag16h5'
```

`tags.yaml` 中只允许真实比赛 ID 1-4。不要把误识别出来的 `15、26、12、18` 映射成 `tag_1` 到 `tag_4`，否则会把错误识别合法化。看到这些 ID 时，应处理图像质量、距离、光照、分辨率或重新打印更清晰的 tag16h5 1-4。

尺寸均为：

```yaml
size: 0.0145
```

`0.0145m` 来自用户实测：tag16h5 黑色码块外边长 `1.45cm`。这个尺寸不是整张纸，也不是白边，是黑色大正方形的外边长。

为降低小码误识别影响，`tag_0` 和旧的 `tag_10` bundle 已取消配置，`tag_bundles: []`。这样 `/tag_detections` 和 TF 只会继续处理配置内的 1-4。注意：`apriltag_ros` 内部会先对原始检测结果做 duplicate pruning，再按 `tags.yaml` 过滤，所以终端里仍可能短暂看到 “Pruning tag ID 12/22/25...” 这类原始误检警告；关键看 `/tag_detections` 是否只输出 1-4。

`settings.yaml` 里的 `max_hamming_dist` 当前使用 `1`。现场测试发现改成 `2` 后近距离糊图会带来明显误识别，因此比赛优先用 `1` 降低 false positive。代价是 1.45cm 小码在近距离失焦时可能漏检，解决方向应是让相机在更清晰距离识别，然后再调机械臂抓取偏移/手眼标定。

`continuous_detection.launch` 默认发布并打开 `/tag_detections_image` 预览窗口，方便看检测框：

```bash
roslaunch apriltag_ros continuous_detection.launch
```

也可以显式打开：

```bash
roslaunch apriltag_ros continuous_detection.launch publish_tag_detections_image:=true show_image:=true
```

当前 `astrapro.launch` 默认 RGB 输入仍是 `640x480@30`、`mjpeg`，AprilTag 识别吃的是 `/camera/rgb/image_raw`。已给它加了可选参数：

```bash
roslaunch astra_camera astrapro.launch rgb_width:=1280 rgb_height:=720 rgb_video_mode:=mjpeg rgb_frame_rate:=30
```

提高分辨率可能改善 1.45cm 小码识别，但会更吃 CPU，并且可能和原相机内参标定分辨率不一致。正式用前必须在真机上确认 `/camera/rgb/image_raw` 实际 width/height、`/tag_detections` 是否稳定，以及抓取位姿是否偏。

### 3.4 front 模式曾经改过，但当前不要优先用它

脚本里现在仍有 `--approach-axis front`，并且曾经修过一个逻辑问题：

- 旧问题：`front` 模式下 `grasp` 是按 AprilTag 局部正面算的，但 `pre_grasp` 偏移用了末端旋转后的姿态，导致不是“码前方再向前伸”。
- 已改成：`pre_grasp` 和 `grasp` 都沿 AprilTag 正面法线计算位置，末端姿态只负责吸盘朝向。

但是实际调试中 `front-tool-pitch-deg 90 / -90 / yaw 180` 等都没得到用户想要的姿态。用户最后明确说想要的是照片中那种固定腕部形态，并确认 `joint5=-1.5709534265016345` 的姿态就是想要的。

所以当前下一步不要继续盲调 `front-tool-*`。优先用 `wrist_forward` 这条线。

## 4. 当前卡在哪

当前卡点不是“找不到 joint5 姿态”了，这个已经找到了。

当前真正未完成的是：

1. WSL 改动是否已经全部同步到真机 `/home/eaibot`，需要确认。
2. 真机上需要重新跑新增的 `--mode wrist_forward`，确认脚本内置动作和之前临时 Python 片段效果一致。
3. 需要验证“先 wrist_forward，再按 AprilTag 抓取”的完整链路是否能稳定吸住物体。

也就是说，代码层面的入口已经加好，真机验证还没完成。

## 5. 下一步计划

### 第一步：同步文件到真机

必须同步这些文件：

```text
/home/zcy/eaibot/handeye-calib/src/mirobot_pick_test.py
-> /home/eaibot/handeye-calib/src/mirobot_pick_test.py

/home/zcy/eaibot/zcy/机械臂操作.txt
-> /home/eaibot/zcy/机械臂操作.txt

/home/zcy/eaibot/mirobot_ws/src/mirobot_moveit_config/launch/mirobot.launch
-> /home/eaibot/mirobot_ws/src/mirobot_moveit_config/launch/mirobot.launch

/home/zcy/eaibot/mirobot_ws/src/mirobot_moveit_config/launch/mirobot_moveit.launch
-> /home/eaibot/mirobot_ws/src/mirobot_moveit_config/launch/mirobot_moveit.launch
```

改的是 Python 和 launch，不需要重新构建包。同步后重新启动相关 launch 即可。

### 第二步：启动机械臂后端，可不开 RViz

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
roslaunch mirobot_moveit_config mirobot.launch start_rviz:=false
```

如果需要看姿态，再开 RViz 或直接不传 `start_rviz:=false`。

### 第三步：单独验证 wrist_forward

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash

python2 /home/eaibot/handeye-calib/src/mirobot_pick_test.py --mode wrist_forward --velocity-scale 0.05 --acceleration-scale 0.05 --planning-time 8.0
```

期望效果：吸盘水平朝前，类似用户发的第二张确认图。

如果动作失败，先看：

```bash
rostopic echo -n 1 /joint_states
rosnode list
```

确认 `/move_group`、`mirobot` 相关节点存在，且 `/joint_states` 正常更新。

### 第四步：基于已有 z 模式抓取，加 wrist_forward

目前最稳的抓取路线仍建议先用 `z` 模式，只是在抓取前先把吸盘调成水平朝前：

```bash
python2 /home/eaibot/handeye-calib/src/mirobot_pick_test.py --mode grasp --supply basic --skip-home --wrist-forward --planning-time 8.0 --velocity-scale 0.1 --acceleration-scale 0.1 --grasp-x -0.045 --y-offset 0.025 --z-offset 0.08 --approach-axis z --approach-gap 0.02
```

如果只想看坐标，不动机械臂：

```bash
python2 /home/eaibot/handeye-calib/src/mirobot_pick_test.py --mode grasp --supply basic --skip-home --dry-run --planning-time 2.0 --disable-replanning --velocity-scale 0.1 --acceleration-scale 0.1 --grasp-x -0.045 --y-offset 0.025 --z-offset 0.08 --approach-axis z --approach-gap 0.02 --debug-hold-seconds 30
```

注意：`--dry-run --wrist-forward` 不会真的转腕部，所以如果要观察转腕后的当前末端姿态，先单独跑 `--mode wrist_forward`，再跑 dry-run。

### 第五步：如果要进一步自动化

后续可以考虑新增更明确的模式，例如：

```bash
--approach-axis wrist_z
```

或者把 `--wrist-forward` 和 `z` 模式封装成一个专门比赛演示命令。但目前先不要扩大改动，先验证现有小改动。

## 6. 踩过的坑，绝对不要再踩

### 6.1 不要把 WSL 当真机

WSL 里没有真实串口、机械臂、相机、吸泵链路。WSL 只能做：

- 读代码
- 改代码
- 语法检查
- launch/XML 静态检查
- 离线逻辑断言

真机动作必须回到 `/home/eaibot` 上执行。

### 6.2 WSL 修改后必须同步

只改 `/home/zcy/eaibot/...` 不会影响真机运行。真机跑的是 `/home/eaibot/...`。

### 6.3 不要为了 WSL 适配乱改 `/home/eaibot` 硬编码

很多路径是给真机跑的，看到 `/home/eaibot` 不一定是错。除非明确要做双环境兼容，否则优先保留真机路径语义。

### 6.4 抓取脚本必须 Python 2

`mirobot_pick_test.py` 必须用：

```bash
python2 /home/eaibot/handeye-calib/src/mirobot_pick_test.py ...
```

不要用 Python 3 跑真机脚本。

### 6.5 Python 2 stdin 片段不要写中文

之前临时运行：

```bash
python2 - <<'PY'
...
PY
```

如果里面有中文注释，会报：

```text
SyntaxError: Non-ASCII character '\xe8' ... but no encoding declared
```

临时 Python 2 片段要么纯英文，要么第一行加编码声明。但现在已经不需要临时片段了，直接用 `--mode wrist_forward`。

### 6.6 `rosdep view is empty` 不是抓取失败主因

真机日志里经常出现：

```text
the rosdep view is empty: call 'sudo rosdep init' and 'rosdep update'
```

这不是 MoveIt 规划失败的直接原因。不要被它带偏。

### 6.7 不要继续盲调 `front-tool-pitch-deg`

用户想要的不是“跟随 tag 姿态的完整末端姿态”，而是照片里固定的“吸盘水平朝前”。已经确认核心是：

```text
joint5 = -1.5709534265016345
```

`front-tool-pitch-deg 90 / -90 / yaw 180` 这条路线已经试过不理想。后续优先走 `wrist_forward`。

### 6.8 不要误以为最后一个关节 joint6 能解决朝向

`joint6` 主要绕末端轴旋转，不能把吸盘从朝下变成朝前。吸盘从朝下到水平朝前主要靠 `joint5`。

### 6.9 `--dry-run` 不会执行腕部动作

`--dry-run` 只算位姿，不动机械臂。即使加了 `--wrist-forward`，也不会真的转腕部。要验证腕部姿态，单独跑：

```bash
python2 /home/eaibot/handeye-calib/src/mirobot_pick_test.py --mode wrist_forward
```

### 6.10 front 模式的 `z-offset` 符号容易反

在 `front` 模式下，`z-offset` 是按 AprilTag 局部坐标换算，不等同于 base 坐标 Z。之前 `--z-offset 0.0` 导致目标太低，`--z-offset -0.08` 才把目标抬高到约 9 到 13cm。现在如果不是明确要调 front 模式，不要继续在这里浪费时间。

### 6.11 改 launch 不需要构建

只改 `.launch` 文件不需要 `catkin_make`。重新 roslaunch 即可。

## 7. 当前关键文件速览

### `/home/zcy/eaibot/handeye-calib/src/mirobot_pick_test.py`

当前新增点：

- `WRIST_FORWARD_JOINT5 = -1.5709534265016345`
- `--mode wrist_forward`
- `--wrist-forward`
- `--wrist-forward-joint5`
- `go_wrist_forward(arm, joint5_target)`

### `/home/zcy/eaibot/mirobot_ws/src/mirobot_moveit_config/launch/mirobot.launch`

当前新增点：

- `start_rviz` 参数
- 转发给 `mirobot_moveit.launch`

### `/home/zcy/eaibot/mirobot_ws/src/mirobot_moveit_config/launch/mirobot_moveit.launch`

当前新增点：

- `start_rviz` 参数
- `moveit_rviz.launch` 受 `if="$(arg start_rviz)"` 控制

### `/home/zcy/eaibot/zcy/机械臂操作.txt`

已更新常用命令和参数说明。后续如果再改脚本入口，必须同步更新这个文档。

## 8. 建议给下个会话的第一句话

如果新会话继续接手，可以先说：

```text
请先阅读 /home/zcy/HANDOFF.md 和 /home/zcy/eaibot/zcy/WSL协作说明.md。当前重点是把 WSL 中 mirobot_pick_test.py 的 wrist_forward 改动同步到真机，然后验证 --mode wrist_forward 和 --wrist-forward 抓取链路。
```
