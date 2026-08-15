# Mirobot 抓取与比赛联调 AI 交接

> 更新时间：2026-08-13
> 面向：没有历史上下文、需要继续处理机械臂抓取或 `zcy_last` 总调度的下一位 AI。
> 原则：本文只保留当前有效实现、边界和待验收事项，不记录历次试错。

`block_pick_main.py` 已将 `SIGTSTP`（终端 `Ctrl+Z`）纳入受控退出：父进程会在
`finally` 中停止独立会话里的 Python2 ROS 子进程，避免机械臂动作进程继续持有
运动锁。历史遗留进程必须按锁提示中的 PID 终止，不能靠删除锁文件解锁。

## 1. 接手后先做什么

1. 阅读本文。
2. 根据任务再读：
   - 机械臂人工操作：`zcy/机械臂操作.md`
   - 无 Tag 人工操作：`zcy/无tag的机械臂操作.md`
   - 总调度有 Tag 接口：`zcy/循迹调用有Tag抓取说明.md`
   - 总调度无 Tag 接口：`zcy/循迹调用无Tag抓取.md`
   - 最新循迹状态机交接：`robocom_ws/src/zcy_last/循迹与联调AI交接.md`
3. 执行 `git status --short`。当前工作树有大量未提交的修改、删除和新增文件，全部按用户已有工作处理。
4. 不要 `reset`、`checkout`、批量覆盖或擅自恢复已删除文档。
5. 先确认用户正在验证哪一条链路以及真机已同步哪些文件，再只处理一个故障层。

## 2. 当前目标和核心设计

项目在比赛车上完成三类动作：

1. B 点有 Tag：底盘对准 Tag，限位接触抓取，放入 ID 1~4 对应载物仓。
2. A 点无 Tag：YOLO 单目粗定位，限位接触抓取，四类分别使用自己的预抓模型和独立示教的入仓放置点。

有 Tag、无 Tag 的正式四目标连续入口都必须一键自动完成，中途不等待 Enter。无 Tag 旧参数 `--wait-key-between-targets` 仅为兼容历史命令而保留，当前不转发也不起作用。
3. 投递：从载物仓取物，经示教中转点送到指定投递点，不使用视觉。

核心设计不能改变：

```text
视觉只负责把吸盘送到物块正前方
-> 机械限位器决定真实接触位置
-> 限位触发后下位机发送精确消息 3\r\n
-> C++ 停止剩余探测路点
-> 上位机确认触发后才允许开泵
```

有 Tag 和无 Tag 除检测定位方式外，共用相同的预抓点语义和限位探测安全时序：

```text
有 Tag：示教一个近距离正对但未接触的预抓点 P
无 Tag：四类分别示教近距离正对但未接触的预抓点 P、Link6姿态和限位前进轴
两者正式动作：先普通规划到 P 后方30mm；在这里开启限位，再受保护地直线到 P，未触发才继续前探
两者限位最大前伸：65mm
限位路径步长：后方安全点到 P 为5mm，P 后继续前探为2mm
限位触发并开泵后：无论在到 P 途中还是 P 前方触发，都沿原路径直退到预抓点后方30mm
```

预抓点后方 `30mm` 用于吸附成功后的搬运间隙。限位未触发时也先直退到该点，不把物块当作已吸住，也不进入 carry。

## 3. 不可破坏的安全约束

- 吸泵串口只能由 C++ 控制器打开；Python 不得再次打开同一设备。
- 限位触发前绝不开泵。
- 吸住后先沿原接近轴直退，完成前不得斜向 carry。
- 直退失败时关泵并终止，不能从接触点直接规划搬运。
- 只把完整精确行 `3\r\n` 识别为限位触发，不能把 `13`、泵回复或旧缓存当成触发。
- 开启新探测时清除旧触发；探测期间拒绝其他客户端提前开泵。
- 固件未进入 `Idle`、串口查询失败或最终关节误差超限时不得返回 `SUCCEEDED`。
- 不给 Joint6 添加硬路径约束。只做等价角最短展开，避免整圈自转和缠管。
- 同时只能有一个机械臂动作客户端和一个 `/cmd_vel` 所有者。
- 示教时使用 RViz `Plan/Execute`，不要手掰机械臂。
- 参考驱动包只能用于 diff，不能放入当前 Catkin `src`，否则会发生同名包冲突。

## 4. 环境和真机边界

```text
本地开发环境：Windows，工作区 F:\桌面\平安城市\国赛\code，Python 3.9
真机工程：/home/eaibot
ROS：Melodic，机械臂脚本使用 Python 2
ONNX/总任务：conda ww，Python 3
Git 分支：main
```

Windows 本地工作区用于读代码、修改、静态检查和 Python 3.9 单元测试。相机、TF、底盘、串口、机械臂、泵、限位器和 MoveIt 动作必须在真机验收。

同步规则：

- Python/YAML 修改：同步对应文件并重启脚本，一般不需编译。
- C++、launch 或 ROS 包修改：同步后在真机执行 `catkin_make`，再重启机械臂终端。
- `zcy_last` 内部接口有跨文件配合和启动自检，部署时应完整同步 `/home/eaibot/robocom_ws/src/zcy_last/`，不要只替换单个模块。

## 5. 限位器和 C++ 后端

主要文件：

```text
mirobot_ws/src/mirobot_urdf_2/src/mirobot_arm_controller.cpp
mirobot_ws/src/mirobot_moveit_config/launch/mirobot.launch
```

服务：

```text
/mirobot_contact_probe_enable   std_srvs/SetBool
/mirobot_contact_state          std_srvs/Trigger
/mirobot_startup_home           std_srvs/Trigger
/switch_pump_status             原吸泵服务
```

实现方式：Python 一次发送包含多个细分路点的 MoveIt action；C++ 在 action 内逐路点检查限位。不要改回每 `2mm` 重新规划一次。

当前 `arm_feedrate=1200`，`contact_probe_feedrate=1200`。如果再次出现 “MoveIt SUCCEEDED 但真机未动或未到位”，先记录串口原始响应、固件状态和 action 时间线，不要直接改抓取偏移。

## 6. 有 Tag 当前实现

数据链路：

```text
Astra 640x480 矫正 RGB
-> YOLO 检出 Tag 区域并补白
-> apriltag_ros 检测 ID 1~4
-> base -> tag_N TF
-> 采集3个不同时间戳的新鲜 TF，中位数/MAD 过滤
-> 四个 ID 共用吸盘姿态、接近轴和接触偏移 + 当前 Tag 平移
-> 普通规划到示教预抓点后方30mm
-> 在预抓点后方30mm开启限位，受保护地直线伸到示教预抓点，再继续限位探测抓取
-> 直退到预抓点后方30mm
-> 固定载物仓放置点
```

主要文件：

```text
handeye-calib/src/tag_yolo_quiet_zone_relay.py
handeye-calib/src/tag_yolo_onnx_worker.py
handeye-calib/src/mirobot_pick_test_tag.py
handeye-calib/src/tag_chassis_align_pick_sequence.py
handeye-calib/config/tag_pick_place_presets.json
mirobot_ws/src/apriltag_ros/apriltag_ros/config/tags.yaml
```

关键默认值：

```text
TF 样本：3
TF 最大年龄：2s
MAD 阈值：5mm
总调度 TF 等待：18s
底盘顺序：当前可见剩余 Tag 从左到右
停车确认：1个新的检测帧
```

单 Tag 限位完整行程未触发时返回码为 `4`。B 点总调度使用 `--allow-partial`：目标未出现、对准超时、Tag TF 超时或限位未接触时跳过该 ID，按实际成功库存继续比赛。

真机权威 `tag_pick_place_presets.json` 必须是版本 3。本地 Windows 同名文件可能过期，绝不能覆盖真机。

有 Tag 抓取固定复用真机 preset 中 ID2 已示教的 `grasp_offset_xyz_base`；ID1~4
共用该接触偏移、`pickup_model` 吸盘姿态和前进轴，只各自保留独立载物仓放置点。
不要新增共享偏移副本或兼容标记。`--approach-gap` 固定表示示教预抓点后方的
安全过渡距离，默认 `30mm`。正式动作固定为“规划到后方安全点 -> 开启限位 ->
受保护地直线到示教预抓点 -> 未触发才从预抓点继续前探”。

## 7. 无 Tag 当前实现

进程分工：

```text
block_pick_main.py：Python 3，ONNX Runtime 和父进程监督
mirobot_pick_test.py：Python 2，ROS/TF/MoveIt/泵/限位
```

主要文件：

```text
handeye-calib/src/block_pick_main.py
handeye-calib/src/block_mono_vision.py
handeye-calib/src/mirobot_pick_test.py
handeye-calib/src/config/block_mono_grasp.yaml
handeye-calib/config/block_mono_pick_place_presets.json
```

模型和类别：

```text
/home/eaibot/handeye-calib/src/model/yolov5/block_occlusion_yolov5n_640_best.onnx
1=power，2=fire，3=gas，4=support
```

当前 YAML 关键值：

```text
矫正 RGB：640x480；ONNX 输入：640x640
稳定定位帧：5
抓取 ROI：x=0.06~0.24
底盘停车确认：4个新鲜帧
无 Tag 示教辅助点：检测表面前85mm；自动移动失败时转为 RViz 手动移动，不中止示教
预抓点后方安全过渡/最大探测：30mm/65mm
限位步长：安全点到 P 为5mm；P 后接触前探为2mm
成功后直退目标：预抓点后方30mm
MoveIt 速度/加速度：0.05/0.05
```

无 Tag 四类必须分别示教。每个 `targets/<类别>` 独立保存
`pregrasp_offset_xyz_base`、`pickup_model` 和 `place_ee_in_base`；不得回退到
顶层共享模型或 Tag 放置点。单类示教入口在同一次运行中先采集未接触预抓点 P，
再让用户从 P 开始通过 RViz 采集该类别的无 Tag 入仓放置点，两个点都成功后才
原子替换该类别旧数据。也支持只重采预抓点或只重采放置点；只采放置点时先按
当前定位自动回到该类别已保存的 P，再由用户从 P 调到放置点。正式动作仍与有
Tag 一致：规划到 P 后方30mm，在这里开启限位，受保护地直线到 P，未触发才
继续前探，触发后直退到 P 后方30mm。

有 Tag 放置示教仍使用 `tag_pick_place_presets.json` 顶层的
`place_teach_start_ee_in_base`，这一链路不改。无 Tag 不读取该起点或其中的
Tag 放置点，而是在每个类别的组合示教中，从刚采集的本类别预抓点 P 开始移动
并保存自己的完整放置位姿。两套放置四元数都不得强制替换。

## 8. `zcy_last` 总调度当前实现

正式唯一入口：

```bash
cd /home/eaibot/robocom_ws/src
python3 -m zcy_last.main [抓取和投递开关]
```

旧 `line_cy_task.py` 只是回退版本，新版不得导入或与它批量同步。

赛前推荐先运行：

```bash
bash /home/eaibot/robocom_ws/src/zcy_last/比赛一键准备.sh
```

它会启动或复用底盘，临时启动 Astra 以建立相机坐标系，再启动 MoveIt 和手眼 TF，检查成功后关闭临时 Astra。底盘、MoveIt 和手眼 TF 转为外部常驻进程，由正式任务复用。

主要总调度文件：

```text
robocom_ws/src/zcy_last/main.py
robocom_ws/src/zcy_last/prepare.py
robocom_ws/src/zcy_last/config.py
robocom_ws/src/zcy_last/control/grasp.py
robocom_ws/src/zcy_last/control/processes.py
robocom_ws/src/zcy_last/task/competition.py
```

终端二的 `python3 -m zcy_last.main` 在最外层使用 `time.monotonic()` 计时，运行期间不输出计时。路线进入 `DONE` 并完成资源清理后只打印一次“全部任务完成，总耗时：X分XX秒”；在 `DONE` 前按 `Ctrl+C` 或收到 ROS 退出则只打印一次“任务已中断，已运行：X分XX秒”。

### B 点有 Tag

B 点在正式循迹前抓取。成功后不能直接回 `FOLLOW`，因为车已经越过第一个入口横条：

```text
B_PICK_PREPARE -> B_PICKING
-> 记录 tag_inventory
-> TRAFFIC_WAIT，确认绿灯
-> 使用 TAG_PICK_FIRST_ENTRY_TIME 直行
-> 使用 TAG_PICK_FIRST_TURN_TIME 第一次右转
-> 出口识别与摆正
-> 完成第1个路口
```

### A 点无 Tag

A 点在第 3 个路口完成后进入搜索握手：

开启 A 点抓取时，第 3 个右拐在入口摆正后独立使用 `A_PICK_THIRD_RIGHT_ENTRY_TIME` 前进、`A_PICK_THIRD_RIGHT_TURN_TIME` 右转；这两个参数只影响该路口，其他路口仍使用全局 `TURN_ENTRY_TIME/TURN_TIME`。

入口横条使用 `STOP_STABLE_FRAMES=1`，单帧命中即进入 `APPROACH`；每次出口完成后通过 `EXIT_ENTRY_IGNORE_TIME=3.0s` 暂停接受入口横条，防止刚经过的出口被再次计作入口。

```text
A_PICK_PREPARE：停车启动 Astra/机械臂/无Tag模型
-> 子进程先刷新识别窗口再写 ready；零检测也显示 RGB 和搜索 ROI
-> A_PICK_SEARCH：先按 FOLLOW_SPEED 直行 UNTAGGED_SEARCH_FORWARD_TIME，窗口只预览不触发
-> 主状态机写 enable，降到 UNTAGGED_SEARCH_SPEED 低速搜索右侧区域
-> 要求数量的不同目标在右侧连续3帧稳定，子进程写 triggered 文件
-> 循迹先发布零速度，再写 release 文件
-> /cmd_vel 所有权交给抓取子进程
-> A_PICKING：慢速把目标移到左侧抓取 ROI 并抓取
-> 记录 untagged_inventory
-> 等待绿灯，按 A 点独立时序通过第 4 个路口
```

发现阶段直接在全画面统计不同目标，没有单独搜索框；窗口显示的就是 `block_mono_grasp.yaml/grasp_roi_ratio` 慢速抓取对齐框。第 4 个路口横条不终止搜索；没有检测够目标时继续固定直行搜索。

### 结果、库存和失败策略

抓取子进程写：

```json
{"completed_ids": [1, 3]}
```

B 点协调器在退出码 `0`、ID 合法且不重复、数量不超过请求数量时判定成功，允许空库存；A 点仍要求数量严格等于请求数量。库存必须来自该 JSON，不能根据检测画面或计划数量猜测。

- B 点目标缺失、对准超时、TF 超时、限位未接触或数量不足：跳过并按实际库存继续。
- A/B 点单个目标失败：停车并跳过当前目标，继续其余目标并按实际库存比赛；只有父进程/结果文件/托管依赖整批故障进入 `PICK_FAILED`。A 点逐个对齐窗口显示所有剩余目标，不再只显示当前一个。
- 投递失败：记录对应 failed ID，报警，不重试该 ID，随后继续循迹。
- 投递成功：从对应库存删除该 ID。

详细接口分别见两份循迹调用文档。

## 9. 投递和权威数据

有 Tag 街区保持原固定点投递。无 Tag 楼宇使用固定红框停车：框中心进入原始320×240画面的 `x=54~173`（归一化 `0.17~0.54`）就进入 `YOLO_STOP`，底盘不旋转，也不做距离闭环；标定与正式运行共用并显示该红框。停车后只使用 `/dev/video2` 楼宇框的左右宽度模型估算真实毫米距离，上下裁切允许，左右边界必须完整；相对 `450mm` 示教参考距离的原始差值会先饱和限制在 `-5mm~+60mm`，再交给机械臂沿前探轴修正 P。该流程不使用相机内参或手眼标定。两套库存只共用 `delivery_presets.json` 中 ID 1~4 的仓内抓取点；中转点、释放点和 idle 按来源独立：

```text
共用仓内抓取点：/home/eaibot/handeye-calib/config/delivery_presets.json
有 Tag 中转/固定释放点：/home/eaibot/handeye-calib/config/delivery_presets.json
无 Tag 楼宇中转/共享P（ID1）：/home/eaibot/handeye-calib/config/untagged_delivery_presets.json
楼宇视觉标定：/home/eaibot/handeye-calib/config/building_delivery_calibration.json
```

无 Tag 投递顺序：楼宇同类框锁定并对准 -> 从机械臂当前姿态直接到仓内抓取点上方约 5cm（投递前不回零） -> 仓内抓取点 -> 开泵 -> 直线上升约 5cm -> 楼宇专用中转点 -> 共享 P 后方30mm -> 开限位 -> 5mm步长到P -> 2mm步长前探最多65mm -> 关泵等待0.7秒 -> 直退30mm -> idle。走满65mm未触发仍按用户决定强制释放并记成功；限位服务、串口或轨迹错误必须保持泵开启并返回失败。正式循迹投递直接使用任务摄像头的楼宇框宽估距，以“实测距离减450mm”沿相机前后方向修正P；框中心只负责进入红框后停车。四类楼宇共用在450mm参考距离下示教的ID1 power楼宇P和前探方向，只需示教一次；不再自动移动idle或辅助点，也不使用楼宇框计算示教高度和姿态。有 Tag 固定投递使用同一投递程序，因此投递前同样不回零。

绝不能覆盖以下真机采集数据：

```text
/home/eaibot/handeye-calib/config/tag_pick_place_presets.json
/home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json
/home/eaibot/handeye-calib/config/delivery_presets.json
/home/eaibot/handeye-calib/config/untagged_delivery_presets.json
/home/eaibot/handeye-calib/config/building_delivery_calibration.json
/home/eaibot/handeye-calib/config/building_delivery_distance_samples_building_new_320x240/
/home/eaibot/handeye-calib/config/astra_rgb_640x480.yaml
手眼标定结果
```

修改前先在真机备份这些文件。

## 10. 当前验证状态

2026-08-13 在 Windows Python 3.9 通过：

```bash
py -3.9 -m pytest -q \
  handeye-calib/tests \
  robocom_ws/src/zcy_last/tests \
  mirobot_ws/src/mirobot_urdf_2/tests \
  --ignore=handeye-calib/tests/test_mirobot_block_mono.py \
  --deselect=handeye-calib/tests/test_block_pick_main.py::test_serve_requests_returns_selected_detection \
  --deselect=handeye-calib/tests/test_block_pick_main.py::test_serve_requests_returns_all_usable_detections_when_target_is_omitted \
  --deselect=handeye-calib/tests/test_block_pick_main.py::test_serve_requests_reports_business_error_without_crashing
```

结果：`275 passed, 3 deselected`；本轮改动脚本通过 Python 3.9 `py_compile`；`git diff --check` 通过。排除项依赖 Linux/ROS 运行语义或测试中未转义的 Windows 路径，仍需在真机验证。

这只证明离线逻辑和接口回归通过，不代表：

- C++ 已在真机成功编译；
- 最新 Python/YAML/`zcy_last` 已全部同步；
- 限位器、泵、Astra、TF、底盘和机械臂动作已完成真机验收；
- B 点特殊首路口和 A 点搜索握手已经跑完整场。

## 11. 建议的真机验收顺序

1. 备份全部 preset、相机内参和手眼标定结果。
2. 完整同步本轮源码；C++/launch 同步后执行 `catkin_make`。
3. 手按限位，验证探测服务、精确触发和限位前不开泵。
4. 分别单测有 Tag、无 Tag：预抓 -> 限位 -> 开泵 -> 直退到预抓点后方 30mm -> 放置。
5. 单测有 Tag 底盘联动及结果 JSON。
6. 单测无 Tag 右侧搜索、release 握手、左侧慢速对准及结果 JSON。
7. 运行一键准备，确认底盘/MoveIt/手眼常驻且临时 Astra 已关闭。
8. 测 B 点抓取后绿灯和第一次专用右转。
9. 测第 3 个路口后的 A 点停车加载、空检测窗口、ready 后直行2秒、低速搜索和第 4 个入口不停边界。
10. 最后才运行有 Tag + 无 Tag + 两套投递的完整比赛流程。

## 12. 故障分层

```text
无框/无检测                  -> 相机、模型、ROI、进程所有权
有框但无 tag_N TF            -> AprilTag、CameraInfo、tags.yaml、手眼 TF
TF/定位不稳定                -> 新鲜帧、MAD、CameraInfo、标定
MoveIt 找不到路径            -> 姿态、工作区、当前关节状态
MoveIt 成功但真机未到位      -> 串口反馈、Idle、action 成功条件
前伸但未吸                   -> 限位消息、探测方向、65mm 行程、泵
已吸但搬运碰撞               -> 预抓点后方30mm直退、carry、放置路径
抓取成功但无库存             -> result JSON 与协调器读取
A 点搜索阶段底盘冲突         -> ready/trigger/release 与 /cmd_vel 所有权
```

一次只改一个明确根因。不要用新增超时、自动继续、盲目重试或堆叠偏移掩盖故障。

## 13. 一句话交接

当前系统已形成“抓取视觉粗对准 + 两套抓取共用预抓语义 + 楼宇中心区域停车 + 无Tag楼宇多距离估距修正P + 四类共享ID1磁吸P + 限位真实接触 + 30mm直退 + JSON真实库存 + 总调度严格资源交接”的完整离线实现；下一位 AI 的重点是保护用户未提交工作和真机权威数据，完成分层真机验收，而不是重新设计抓取算法。
