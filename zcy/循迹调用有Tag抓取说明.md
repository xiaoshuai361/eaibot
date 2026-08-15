# 有 Tag 抓取与投递：总调度 AI 接口

> 更新时间：2026-08-12
> 本文只面向 `zcy_last` 总调度、循迹状态机和进程编排。示教、AprilTag 调试和人工单步命令见《机械臂操作.md》。

## 1. 正式比赛入口（两个终端）

终端一先用 `launch.py` 启动并常驻底盘、MoveIt 和手眼 TF：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/robocom_ws/devel/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
conda activate ww
cd /home/eaibot/robocom_ws/src
python3 -m zcy_last.launch
```

保持终端一运行。终端二再启动比赛任务：

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/robocom_ws/devel/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
conda activate ww
cd /home/eaibot/robocom_ws/src

python3 -m zcy_last.main \
  --tag-pick --tag-pick-count 4 \
  --tag-delivery \
  --no-untagged-pick
```

只抓不投：

```bash
python3 -m zcy_last.main \
  --tag-pick --tag-pick-count 4 \
  --no-tag-delivery
```

同时启用 B 点有 Tag 和 A 点无 Tag 时，只运行一个总任务：

```bash
python3 -m zcy_last.main \
  --tag-pick --tag-pick-count 4 --tag-delivery \
  --untagged-pick --untagged-pick-count 3 --untagged-delivery
```

抓取数量只能是 `1..4`，表示本次最多尝试成功入仓的数量，不代表固定 ID 顺序。单个目标缺失、对准超时、Tag TF 超时或限位未接触会跳过，并按实际库存继续比赛。不要同时运行两个 `zcy_last.main`，也不要手动并行启动 Tag 抓取、Astra、Tag 检测栈或键盘控制。

## 2. 当前 B 点状态机

有 Tag 抓取在正式循迹前执行：

```text
启动 zcy_last.main
-> 直接使用 launch.py 已经常驻的底盘、MoveIt、机械臂和手眼 TF，不重复等待检查
-> 启动 Astra、YOLO 补白和 AprilTag
-> B_PICK_PREPARE：停车稳定1.5s
-> B_PICKING：Tag 子进程独占 /cmd_vel，逐个对准抓取
-> 成功读取实际库存
-> 关闭 Tag 检测栈和 Astra，恢复街区任务 YOLO
-> 等待绿灯
-> 使用 B 点专用时序直行和第一次右转
-> 识别出口横条并摆正
-> 完成第1个路口，进入正常九路口流程
```

B 点抓取成功后不能直接恢复 `FOLLOW`，因为车辆已经越过第一个入口横条。当前专用参数是：

```text
TAG_PICK_FIRST_ENTRY_TIME=5.5s
TAG_PICK_FIRST_TURN_TIME=4.0s
```

这两个值只控制 B 点后的第一次右转；普通路口仍使用 `TURN_ENTRY_TIME` 和 `TURN_TIME`。

## 3. 协调器实际调用

真实命令由 `zcy_last/control/grasp.py` 构造，等价于：

```bash
python2 /home/eaibot/handeye-calib/src/tag_chassis_align_pick_sequence.py \
  --sequence 1,2,3,4 \
  --order left_to_right \
  --max-targets <要求成功数> \
  --allow-partial \
  --result-file <临时JSON> \
  --preset-file /home/eaibot/handeye-calib/config/tag_pick_place_presets.json \
  --pick-velocity-scale 0.2 \
  --pick-acceleration-scale 0.2 \
  --pick-approach-gap 0.030 \
  --tag-tf-wait-seconds 18.0
```

`PICK_DEBUG_VIEW=True` 时会添加 `--show-debug-window`，正式比赛只显示一个 `tag_pick_detection` 合成窗口：

```text
AprilTag 检测图底图
+ YOLO Tag ID 框
+ 黄色中心点
+ 红色底盘对准 ROI
+ 当前 Tag 数量
```

AprilTag 自带的 `show_image` 窗口默认关闭，YOLO relay 也不在送给 AprilTag 的输入图上画调试框，避免多窗口和重复框。合成图同时发布到：

```text
/tag_chassis_align/debug_image
```

需要单独排查原始 AprilTag 输出时，可临时使用 `rqt_image_view /tag_detections_image`，不改正式比赛启动参数。

联动脚本内部调用 `mirobot_pick_test_tag.py`。总调度不得复制第二套抓取算法，也不要自行拼接“对准 + 单 Tag 抓取”。

四个 Tag 一键连续处理，中途不读取终端输入、不等待 Enter。历史参数 `--wait-key-between-tags` 仅保留命令兼容性，当前不会改变连续执行流程。

## 4. 单个 Tag 动作契约

```text
从当前可见剩余 Tag 中选择最左者
-> 底盘低速移入红色 ROI，以1个新检测帧确认停车
-> 不回零，从当前机械臂姿态继续
-> 等待新的 base -> tag_N TF
-> 收集3个不同时间戳的新鲜 TF 并过滤
-> 普通关节规划到四个 Tag 共用的抓取前中转点
-> 普通规划到示教预抓点后方30mm（`--pick-approach-gap` 可配置）
-> 在该后方安全点开启限位检测
-> 以5mm步长保持姿态受保护地直线伸到示教预抓点；途中触发立即停止
-> 若尚未触发，再从预抓点最多前探65mm，每2mm检查
-> 收到精确限位消息 3\r\n 后停止剩余路点
-> 确认限位后才开泵
-> 沿原路径退到示教预抓点后方30mm
-> carry -> ID对应载物仓 -> idle
```

无论限位在“后方安全点到 P”途中还是 P 前方触发，吸附后都沿原轴直退到 P 后方 `30mm`。限位未触发时也先退到该位置，不得开泵或进入 carry。

四个 Tag 固定共用真机 preset 中 ID2 的 `grasp_offset_xyz_base`，其语义是“近距离、正对、未接触的示教预抓点”。不新增迁移字段。四个 ID 只分别保存自己的载物仓放置点。

总调度不得修改限位时序、直接操作泵串口、给 Joint6 添加硬路径约束或自动重试失败抓取。

## 5. 结果和失败处理

子进程原子写入：

```json
{"completed_ids": [3, 1]}
```

总调度接受部分成功：

- 整批退出码为 `0`；
- ID 均为 `1..4` 且不重复；
- 数量可以是 `0..--tag-pick-count`，只投递实际成功 ID。

结果保存到 `tag_inventory`。不能根据 Tag 检测画面、数字顺序或计划数量猜库存。单 Tag 返回码 `4` 表示限位未接触，会跳过当前 ID。

- 目标未出现、对准超时、TF 超时或限位未接触：跳过当前 ID，不重试，继续其他目标或按实际库存进入比赛。
- 单次机械臂子命令非零退出：父流程先停车，再跳过当前 ID 并继续剩余目标。
- B 点父进程自身崩溃、结果文件损坏或托管依赖退出：进入 `PICK_FAILED`。
- 投递失败：报警、记录 failed ID、继续循迹，不重试该 ID。

## 6. 有 Tag 投递

| 街区识别 | 库存 ID | 物资 |
| --- | ---: | --- |
| 普通人群 | 1 | 基本生活物资 |
| 医疗人群 | 2 | 医疗包 |
| 可回收垃圾 | 3 | 常规消杀剂 |
| 其他垃圾 | 4 | 生物危害专用消杀剂 |

只有 ID 存在于 `tag_inventory` 且未在失败集合中，才调用：

```bash
python2 /home/eaibot/handeye-calib/src/mirobot_delivery.py \
  --mode run_delivery \
  --sequence <单个库存ID> \
  --delivery-file /home/eaibot/handeye-calib/config/delivery_presets.json \
  --cargo-pick-file /home/eaibot/handeye-calib/config/delivery_presets.json \
  --tag-preset-file /home/eaibot/handeye-calib/config/tag_pick_place_presets.json
```

投递不使用视觉。`delivery_presets.json` 中 ID 1~4 的仓内抓取点同时供有 Tag
和无 Tag 投递使用；有 Tag 的中转点和释放点仍从本文件读取。成功后从库存删除
ID。如果后续还启用 A 点无 Tag 抓取，总调度会保留机械臂公共栈。

## 7. 资源和数据边界

正式比赛前先在终端一运行：

```bash
python3 -m zcy_last.launch
```

`launch.py` 会启动或复用底盘，临时启动 Astra，再启动 MoveIt 和手眼 TF；成功后关闭临时 Astra，并让底盘、MoveIt、手眼 TF 常驻供 `main.py` 复用。`launch.py` 不自动启动 `main.py`。

默认托管模式下，总任务按需管理 Astra、YOLO 补白、AprilTag 和抓取子进程。`--external-ros` 只用于所有依赖均已人工准备好的调试环境，不建议用于正式比赛。

权威文件：

```text
/home/eaibot/handeye-calib/src/tag_chassis_align_pick_sequence.py
/home/eaibot/handeye-calib/src/mirobot_pick_test_tag.py
/home/eaibot/handeye-calib/config/tag_pick_place_presets.json
/home/eaibot/handeye-calib/config/delivery_presets.json
/home/eaibot/handeye-calib/config/astra_rgb_640x480.yaml
```

真机 Tag preset 必须保持版本 3。总调度不得用本地 Windows 同名文件覆盖真机，也不得覆盖 CameraInfo 或手眼标定结果。

## 8. 排错入口

日志：

```text
/home/eaibot/logs/zcy_last/<启动时间>/pick_tag.log
```

按顺序分类：YOLO 无框 -> AprilTag 无 TF -> 凑不齐3个新 TF -> 底盘对准失败 -> MoveIt 预抓失败 -> 限位未触发 -> 退到预抓点后方30mm失败 -> 结果 JSON 异常 -> B 点绿灯/专用首转时序异常。

一次只处理一层；不要用新增超时、自动继续、缓存 TF 或堆叠偏移掩盖故障。
