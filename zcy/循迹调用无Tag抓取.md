# 无 Tag 抓取与投递：总调度 AI 接口

> 更新时间：2026-08-13
> 本文只面向 `zcy_last` 总调度、循迹状态机和进程编排。示教、标定和人工单步命令见《无tag的机械臂操作.md》。

## 1. 唯一比赛入口

```bash
source /opt/ros/melodic/setup.bash
source /home/eaibot/robocom_ws/devel/setup.bash
source /home/eaibot/mirobot_ws/devel/setup.bash
source /home/eaibot/handeye-calib/devel/setup.bash
conda activate ww
cd /home/eaibot/robocom_ws/src

python3 -m zcy_last.main \
  --untagged-pick --untagged-pick-count 4 \
  --untagged-delivery
```

只抓不投：

```bash
python3 -m zcy_last.main \
  --untagged-pick --untagged-pick-count 4 \
  --no-untagged-delivery
```

同时启用 B 点有 Tag 和 A 点无 Tag 时，只运行一个总任务：

```bash
python3 -m zcy_last.main \
  --tag-pick --tag-pick-count 4 --tag-delivery \
  --untagged-pick --untagged-pick-count 4 --untagged-delivery
```

抓取数量只能是 `1..4`，表示必须成功入仓的数量，不代表固定类别顺序。不要同时运行两个 `zcy_last.main`，也不要手动并行启动无 Tag 抓取或键盘控制。

## 2. 当前 A 点状态机

无 Tag 抓取在第 3 个路口完成后触发，不是停车后立即把底盘交给抓取：

```text
完成第3个路口
-> 关闭任务 YOLO
-> A_PICK_PREPARE：停车，启动 Astra、机械臂公共栈和无Tag模型
-> 子进程先显示识别窗口，再写 search_ready 文件；零检测也持续刷新
-> A_PICK_SEARCH：总调度以 FOLLOW_SPEED、固定零角速度直行2秒，窗口仅预览
-> 总调度写 search_enable 文件，降为 UNTAGGED_SEARCH_SPEED 低速搜索
-> 无Tag模型在全画面确认要求数量的不同目标
-> 要求数量的不同目标在右侧连续3帧稳定，子进程写 search_trigger 文件
-> 总调度先发布零速度
-> 总调度写 search_release 文件
-> /cmd_vel 所有权切换给抓取子进程
-> A_PICKING：子进程慢速把目标移到左侧抓取区并抓取
-> 成功读取实际库存
-> 关闭 Astra，恢复楼宇任务 YOLO
-> 等待绿灯
-> 按 A 点独立时间直行并完成第 4 个路口左转
-> 识别出口横条并摆正后恢复普通流程
```

如果到达第 4 个路口入口时仍未检测够目标，不因横条停车或进入 `PICK_FAILED`，继续固定直行搜索。

两个 ROI 不能混用：

```text
发现阶段：全画面统计不同目标，没有单独搜索框；窗口红框直接显示慢速抓取对齐 ROI
左侧抓取区：block_mono_grasp.yaml 的 grasp_roi_ratio=(0.06,0.00,0.24,1.00)
模型就绪后快速直行时间：UNTAGGED_SEARCH_FORWARD_TIME=3.5
边走边搜索速度：UNTAGGED_SEARCH_SPEED=0.03，angular.z 固定为0
A 点抓取后直行时间：UNTAGGED_PICK_NEXT_ENTRY_TIME=5.5
A 点抓取后左转时间：UNTAGGED_PICK_NEXT_TURN_TIME=4.0
```

## 3. 协调器实际调用

真实命令由 `zcy_last/control/grasp.py` 构造。A 点比赛流程等价于：

```bash
python3 /home/eaibot/handeye-calib/src/block_pick_main.py \
  --run-chassis-sequence \
  --sequence 1,2,3,4 \
  --max-targets <要求成功数> \
  --allow-partial \
  --result-file <临时JSON> \
  --confidence 0.5 \
  --config /home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml \
  --preset-file /home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json \
  --search-before-chassis \
  --search-ready-file <ready文件> \
  --search-enable-file <enable文件> \
  --search-trigger-file <trigger文件> \
  --search-release-file <release文件> \
  --search-stable-frames 3 \
  --search-poll-hz 3.0
```

`PICK_DEBUG_VIEW=True` 时协调器还会添加 `--show-rgb`。

必须保留 `--run-chassis-sequence`。旧版“循迹直接调用单目标抓取”的接法已废弃；当前通过 ready/enable/trigger/release 明确交接 `/cmd_vel`。

该连续入口一键处理 `1,2,3,4`，物块之间不读取终端输入、不等待 Enter。旧命令即使仍带 `--wait-key-between-targets` 也会忽略该兼容参数并自动继续。

## 4. 单个物块动作契约

| ID | 类别 | 物资 |
| ---: | --- | --- |
| 1 | `power` | 应急电源 |
| 2 | `fire` | 灭火装置 |
| 3 | `gas` | 气体净化装置 |
| 4 | `support` | 结构支撑装置 |

```text
从当前可见剩余目标中选择最左者
-> 底盘慢速对准左侧抓取 ROI 并以4个新鲜帧确认
-> 不回零，从当前机械臂姿态继续
-> 取得5个新鲜稳定定位观测
-> 普通规划到示教预抓点 P 后方30mm
-> 在该后方安全点开启限位检测
-> 以5mm步长保持姿态受保护地直线伸到示教预抓点 P；途中触发立即停止
-> 若尚未触发，再从 P 最多前探65mm，每2mm检查
-> 收到精确限位消息 3\r\n 后停止剩余路点
-> 确认限位后才开泵
-> 沿原路径退到预抓点后方30mm
-> carry -> 对应载物仓 -> idle
```

无论限位在“后方安全点到 P”途中还是 P 前方触发，吸附后都沿原轴直退到 P 后方 `30mm`。限位未触发时也先退到该位置，不得开泵或进入 carry。

无 Tag 与有 Tag 都使用“P 后方30mm -> 开启限位 -> 受保护地到 P -> 未触发再前探 -> P 后方30mm”的安全动作语义。总调度不得直接操作泵串口、修改限位时序或增加自动重试。

## 5. 结果和失败处理

子进程原子写入：

```json
{"completed_ids": [1, 3, 4]}
```

只有以下条件同时满足才成功：

- 整批退出码为 `0`；
- ID 均为 `1..4` 且不重复；
- 数量严格等于 `--untagged-pick-count`。

结果保存到 `untagged_inventory`。不能根据检测画面、目标顺序或计划数量猜库存。单目标返回码 `4` 表示限位未接触，但总调度层仍把未达到整批数量视为抓取失败。

- 单个物资失败：停车、记录原因并跳过，继续尝试其他物资，最后按实际库存继续比赛；只有整批父进程、结果文件或依赖故障才进入 `PICK_FAILED`。
- 投递失败：报警、记录 failed ID、继续循迹，不重试该 ID。

## 6. 无 Tag 投递

| 楼宇 | 库存 ID |
| --- | ---: |
| 电力故障 | 1 |
| 火灾 | 2 |
| 有毒气体 | 3 |
| 坍塌 | 4 |

只有 ID 存在于 `untagged_inventory` 且未在失败集合中，才调用：

```bash
python2 /home/eaibot/handeye-calib/src/mirobot_delivery.py \
  --mode run_delivery \
  --sequence <单个库存ID> \
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

楼宇框中心进入原始320×240画面的红色竖框 `x=54~173`（归一化 `0.17~0.54`）时，状态机直接按原 `YOLO_STOP` 停车；底盘不旋转，也不根据楼宇距离二次前进或后退。标定窗口和正式楼宇 YOLO 窗口显示并使用同一个红框。停车后只使用该楼宇框的左右宽度模型估算真实毫米距离，上下裁切允许，左右边界必须完整；并将“估距 - 450mm示教参考距离”传给机械臂。机械臂沿共享前探轴修正 P；仓内抓取点共用 `delivery_presets.json`，楼宇专用中转点和四类共享的 ID1 P 从 `untagged_delivery_presets.json` 读取。到修正后 P 的后方30mm开启限位，5mm步长到P，再以2mm步长最多前探65mm；释放后直退30mm再回 idle。

走满65mm未触发时按已确认策略强制关泵并返回成功，库存会被消费；限位服务、串口或轨迹执行报错不会走强制释放，泵保持开启并返回投递失败。

## 7. 资源和数据边界

默认托管模式下，总调度管理 Astra、无 Tag 子进程、任务 YOLO 切换以及必要时的机械臂公共栈。底盘可由一键准备脚本启动并作为外部常驻进程复用。

正式比赛前推荐：

```bash
bash /home/eaibot/robocom_ws/src/zcy_last/比赛一键准备.sh
```

权威文件：

```text
/home/eaibot/handeye-calib/src/block_pick_main.py
/home/eaibot/handeye-calib/src/mirobot_pick_test.py
/home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml
/home/eaibot/handeye-calib/config/block_mono_pick_place_presets.json
/home/eaibot/handeye-calib/config/tag_pick_place_presets.json
/home/eaibot/handeye-calib/config/untagged_delivery_presets.json
/home/eaibot/handeye-calib/config/building_delivery_calibration.json
```

无 Tag 四类分别读取自己独立示教的预抓偏移、Link6 姿态、限位前进轴和入仓
放置位姿，全部保存在 `block_mono_pick_place_presets.json` 对应类别中。
有 Tag preset 不参与无 Tag 入仓放置。投递阶段的仓内抓取点仍按用户确认共用
`delivery_presets.json`，不要拆分。总调度不得覆盖 preset、CameraInfo 或
手眼标定结果。

## 8. 排错入口

日志：

```text
/home/eaibot/logs/zcy_last/<启动时间>/pick_untagged.log
```

按顺序分类：搜索模型未 ready -> 右侧未 trigger -> release/底盘所有权失败 -> 左侧对准失败 -> 定位失败 -> MoveIt 失败 -> 限位未触发 -> 退到预抓点后方 30mm 失败 -> 结果 JSON 异常。

一次只处理一层；不要用新增超时、自动继续或堆叠偏移掩盖故障。
