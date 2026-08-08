# Mirobot 抓取项目交接文档

> 更新日期：2026-08-07
> 写给没有上下文的新 AI，请先读本文，再读对应操作文档和代码。

## 1. 当前任务和结论

项目是 Astra RGB 相机 + Mirobot 六轴机械臂 + MoveIt + 吸盘抓取。当前有两条链路：

1. **有 Tag 抓取**：已经基本能抓取和定点放置，但小 Tag、手眼误差和开环步进机会限制鲁棒性。当前不是主要开发线。
2. **无 Tag 抓取**：当前主线。使用 YOLOv5 ONNX + 矫正后 RGB + 单目框尺寸估距，不使用深度图。

用户的硬性要求：

- 无 Tag 链路不使用深度相机数据。
- Python3/conda `ww` 只负责 ONNX 推理；Python2 只负责 ROS Melodic/MoveIt。
- 显示终端和抓取终端分开。
- 只允许红色 ROI 中的物块进入定位、标定和抓取。
- 四类物块分别标定距离，不对四个不同框求共同平均。
- 放置点与有 Tag 链路共用，无 Tag 只需重新示教抓取。
- 运行目录只保留当前实现；旧实现由 Git 历史保留，不在 `src/old/` 或主路径堆放备份副本。

## 2. 工作环境

当前 AI 操作的是 WSL 镜像，不是真机：

```text
WSL 代码根目录：/home/zcy/eaibot
真机对应目录：/home/eaibot
Git 分支：fix/tag-grasp-robustness
ROS：Melodic
```

协作规则：

1. 在 WSL 修改、做静态检查和单元测试。
2. 必须明确告诉用户需同步到真机的文件。
3. 真机的相机、泵、机械臂和 MoveIt 动作由用户验证，不得把 WSL 测试写成真机验收。
4. 工作树有其他改动，不要回退与当前任务无关的文件。

必读文档：

```text
/home/zcy/eaibot/zcy/WSL协作说明.md
/home/zcy/eaibot/zcy/无tag的机械臂操作.md
/home/zcy/eaibot/zcy/机械臂操作.md
```

## 3. 无 Tag 当前架构

```text
/camera/rgb/image_rect_color + CameraInfo
        |
        v
block_pick_main.py (Python3, ONNX Runtime, conda ww)
        |
        | pipe 传递当前 RGB 图像和 YOLO 结果
        v
mirobot_pick_test.py (Python2, rospy, TF, MoveIt, 泵)
```

主要文件：

```text
handeye-calib/src/block_pick_main.py
handeye-calib/src/block_mono_vision.py
handeye-calib/src/mirobot_pick_test.py
handeye-calib/src/block_distance_collect.py
handeye-calib/src/block_distance_calibrate.py
handeye-calib/src/config/block_mono_grasp.yaml
handeye-calib/config/block_mono_pick_place_presets.json
```

模型：

```text
/home/eaibot/handeye-calib/src/model/yolov5/block_occlusion_yolov5n_640_best.onnx
```

模型的 WSL 训练/测试材料：

```text
原始数据集：/home/zcy/model_train/datasets/raicam/Block_v5n
遮挡增强数据集：/home/zcy/model_train/datasets/raicam/Block_v5n_occlusion_aug
训练输出：/home/zcy/models/block_occlusion_yolov5n_640
```

当前模型固定输入为 `640x640`，输出为 `[1,25200,9]`，类别顺序仍为 power/fire/gas/support。增强集包含空背景、单物块及两/三/四物块组合，用于修复 power/support 共现混淆。离线验证集 `mAP50=0.995`、`mAP50-95=0.747`；真机仍从 `--confidence 0.5` 起步，不要把离线指标当成定位精度。

类别与显示缩写：

```text
power   -> POW，电力物资
fire    -> FIR，消防物资
gas     -> GAS，气体净化
support -> SUP，支撑物资
```

定位路径：

```text
矫正 RGB -> YOLOv5 ONNX -> 只接受新时间戳
-> ROI 门控 -> 多帧中位数/MAD 过滤
-> 单目距离 -> CameraInfo 反投影
-> TF 转到 base -> 示教偏移抓取 -> 固定点放置
```

抓取规则：

```text
抓取 XYZ = 当前过滤后物块 XYZ + 当前类别示教偏移
吸盘姿态 = 首次抓取示教锁定的 base 姿态
接近方向 = 首次抓取示教锁定的 base 方向
放置姿态 = 当前类别在 base 下的固定示教姿态
```

不给 Joint6 添加硬路径约束，否则 RRT 容易无解。

无 Tag 方案没有 AprilTag 那样的稳定 6D 姿态，只适合物块高度、正面朝向和车体偏航均在已标定范围内的场景。大角度倾斜、明显旋转或超出标定距离时应拒绝抓取，不要靠继续增加偏移参数掩盖。

## 4. 已完成的无 Tag 改造

- 完全不依赖 depth 话题。
- 使用 `/camera/rgb/image_rect_color` 和有效 CameraInfo。
- Python3 ONNX 与 Python2 ROS 分进程，避免 Melodic/Python 环境冲突。
- 修复 Python2 ROS 日志遇到中文导致的 `UnicodeEncodeError/UnicodeDecodeError`。ROS 日志只输出 ASCII，中文用普通终端打印。
- 实时显示和抓取命令可并行，ROS 节点使用 anonymous name。
- 红色 ROI 共用于距离标定、dry-run、示教和抓取。
- 可用 `--pregrasp-distance-mm` 从命令行设置预抓距离。
- preset v2 已同步有 Tag 的四个放置位置和 idle。当前 preset 里还没有无 Tag 抓取示教数据。
- 引导采集程序按距离优先采集：同一距离先完成 `power/fire/gas/support`，再改下一个距离；失败不会留下伪完成 CSV。
- 每个距离使用多帧框宽/高的中位数，不是算术平均。各距离中位点再用于拟合。
- 无 Tag 抓取示教先保持当前末端姿态，自动移动到检测表面前 `110mm`，再由用户在 RViz 中 Plan/Execute 到接触姿态。该距离使用独立的 `teach_assist_distance_mm`，不影响正式预抓距离。
- `src/config/block_mono_grasp.yaml` 现在是无 Tag 参数的唯一来源；Python 3 检测端和 Python 2 ROS 端读取同一份 YAML，Python 文件不再各自保存一套可调默认值。
- 无 Tag 目标编号与有 Tag 放置点一致：`1=power`、`2=fire`、`3=gas`、`4=support`。`block_pick_main.py --target` 同时接受数字和英文。
- 无 Tag 已增加 `--run-chassis-sequence --sequence 1,2,3,4`：复用有 Tag 的红框对准逻辑，按“最左可见剩余目标 → 低速对准 → 停稳新帧确认 → 抓放 → idle → 启动回零”逐个处理。底盘参数集中在 YAML 的 `chassis_sequence` 段。
- 无 Tag 的 MoveIt 动作使用 `/tmp/mirobot_arm_motion.lock` 跨进程互斥，避免多个残留 `mirobot_pick_test` 客户端互相 PREEMPT；纯 live-preview、dry-run 和标定不再初始化 MoveIt。
- 无 Tag 抓取点允许直接重复示教，不再要求 `--overwrite`；只有完整示教成功后才替换旧偏移，中途失败保留旧 preset。终端确认提示只接受空 Enter，防止误粘贴命令触发运动。

## 5. 当前最高优先级：围绕 400mm 重新标定 RGB 距离

当前使用 `block_occlusion_yolov5n_640_best.onnx`，实际抓取距离约为 `400mm`。旧的 `280~480mm` 等间隔标定点在工作区附近不够密，并且旧参数不应直接沿用到新检测模型。

新采集点为：

```text
340、360、370、380、390、400、410、420、430、440、460mm
```

`360~440mm` 每 `10mm` 一点，端点覆盖约 `400±60mm`；每类每点采集 `10` 帧。标定距离是 RGB 镜头光心到物块正面的垂直 Z 距离。

采集时配置曾暂时设为 `distance_method: theory`，避免旧拟合参数驱动机械臂。采集命令和完整环境初始化见 `zcy/无tag的机械臂操作.md`。

旧数据保留在：

```text
/home/eaibot/handeye-calib/config/block_distance_samples/
```

新数据及拟合结果写入独立目录：

```text
/home/eaibot/handeye-calib/config/block_distance_samples_occlusion640_400/
```

四个 `*_model.yaml` 已在 `2026-08-08` 生成，参数已经写入 `block_mono_grasp.yaml`，并切换为 `distance_method: calibrated`。power/fire/support 拟合可接受；gas 的宽/高 RMSE 为 `11.86mm/13.72mm`，最大残差为 `27.56mm`，是当前风险项。下一步必须对四类分别使用未参与拟合的 `350、405、450mm` 三点做 `--dry-run --show-rgb` 独立验证，尤其检查 gas；通过后才进行示教或抓取。

## 6. 最新真机故障：实际仍要求 20 帧

真机最新日志：

```text
Could not collect 20 stable fresh YOLO observations within 25.0s;
collected 14 unique frames.
```

这不是目标不稳定。`Last filter error: none` 说明程序因为没到 20 帧，还没进入稳定性判断。

WSL 最新代码已确认：

- `block_distance_collect.py` 默认 `--frames 10`。
- 它会向 `block_pick_main.py` 显式传递 `--frames 10`。
- `block_pick_main.py` 会继续传给 Python2 `mirobot_pick_test.py`。
- 日志当时的超时是 25 秒；最新代码已调整为 50 秒。

因此真机是 **文件只同步了一部分**，至少 `block_distance_collect.py` 还是旧版，或真机命令仍显式写了 `--frames 20`。

已给最新采集程序加启动自检输出：

```text
采集配置：每个类别/距离 10 帧，置信度 0.500
采集入口：/home/eaibot/handeye-calib/src/block_pick_main.py
配置文件：/home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml
```

真机下一步必须先同步以下文件：

```text
/home/eaibot/handeye-calib/src/block_distance_collect.py
/home/eaibot/handeye-calib/src/block_distance_calibrate.py
/home/eaibot/handeye-calib/src/block_pick_main.py
/home/eaibot/handeye-calib/src/mirobot_pick_test.py
/home/eaibot/handeye-calib/src/block_mono_vision.py
/home/eaibot/handeye-calib/src/config/block_mono_grasp.yaml
```

同步后运行采集，第一屏必须明确显示 `10 帧`。若仍显示 20，检查执行的绝对路径和实际命令行，不要继续放宽过滤阈值。

## 7. 无 Tag 真机运行顺序

完整命令以 `zcy/无tag的机械臂操作.md` 为准，简化顺序如下：

1. 启动 Astra RGB，确认 `/camera/rgb/image_rect_color` 为 `640x480`。
2. 启动 Mirobot + MoveIt。示教时开 RViz，比赛时可 `start_rviz:=false`。
3. 发布 eye-on-base 手眼 TF。
4. 在 conda `ww` 中运行 `block_pick_main.py --live-preview`。
5. 先做纯视觉 dry-run。
6. 完成四类单目距离标定，启用 `calibrated`。
7. 示教无 Tag 抓取；放置点和 idle 已有。
8. 先 `--preview-taught-block`，再 `--stop-at-taught-pre-grasp`，最后 `--run-taught-block`。

红色 ROI 当前为：

```yaml
grasp_roi_ratio: [0.06, 0.00, 0.24, 1.00]
```

在 `640x480` 中约为 `x=38..154`。

## 8. preset 当前状态

无 Tag preset：

```text
/home/zcy/eaibot/handeye-calib/config/block_mono_pick_place_presets.json
```

当前是 `version: 2`，已包含：

- `idle_joint_values`
- power/fire/gas/support 的 `place_ee_in_base`

抓取后的 `carry_joint_values` 也应直接复用有 Tag preset 的现有中间过渡点，不重新示教。当前 WSL 镜像中的旧 Tag preset 尚未包含该字段；真机运行前应从真机的 `tag_pick_place_presets.json` 复制到无 Tag preset。正式抓取现在强制校验六个关节值，缺失时会在运动和开泵前报错，不再静默跳过。

映射：

```text
power   -> 原 Tag 仓位 1
fire    -> 原 Tag 仓位 2
gas     -> 原 Tag 仓位 3
support -> 原 Tag 仓位 4
```

它尚未包含完整的无 Tag 抓取模型，必须等距离标定完成后在真机重新示教。示教时不要手掰机械臂，用 RViz Plan/Execute 到位后再回终端确认。

## 9. 有 Tag 链路的背景

有 Tag 详细命令在 `zcy/机械臂操作.md`。当前重要结论：

- 使用 1.45cm `tag16h5` ID 1-4，标签不能加大。
- YOLO 补白后给 AprilTag，补白和检测是实时的。
- Tag 位姿需要多帧新时间戳过滤。
- 新抓取模型不让小 Tag 的旋转抖动直接控制吸盘姿态。
- 放置点稳定，已被复用到无 Tag preset。
- 机械臂是开环步进机，每次上电后正确回零很重要。
- 不要恢复曾经导致 RRT 无解的 Joint6 硬约束。

## 10. 底层驱动结论

已对比过参考/原始工作区和现有 `mirobot_arm_controller.cpp`。官方/原始实现本质也是简单 GCode 开环控制，不能直接替换来获得真正编码器闭环。

当前底层安全方向是：

- 只在固件状态为 Idle 且关节角在容差内时返回成功。
- 串口查询失败不能给缓存关节状态重新盖新时间戳。
- 上一个 goal 真正结束前不发下一个。
- 查询失败、超时、Alarm 或关节超差必须 ABORTED，不能伪装 SUCCEEDED。

不要把官方或原始工作区解压到当前 `mirobot_ws/src`，会产生同名 ROS 包冲突。参考包应放在 `/home/eaibot/reference/`。

## 11. 验证状态

最新 WSL 针对性测试：

```text
58 passed
```

覆盖：

- 单目视觉数学和稳定性过滤
- Python3 主进程参数/子进程协议
- Python2 MoveIt 子进程的无 Tag 路径
- 距离采集和拟合

执行过的命令：

```bash
python3 -m py_compile \
  handeye-calib/src/block_distance_collect.py \
  handeye-calib/src/block_distance_calibrate.py \
  handeye-calib/src/block_pick_main.py \
  handeye-calib/src/block_mono_vision.py

python3 -m pytest -q \
  handeye-calib/tests/test_block_distance_collect.py \
  handeye-calib/tests/test_block_distance_calibrate.py \
  handeye-calib/tests/test_block_pick_main.py \
  handeye-calib/tests/test_mirobot_block_mono.py \
  handeye-calib/tests/test_block_mono_vision.py
```

最新输出为 `58 passed in 0.55s`，`git diff --check` 通过。这不代表真机动作已验证。

## 12. 下一个 AI 的建议开工顺序

1. 先读本文和 `zcy/无tag的机械臂操作.md`。
2. 检查 `git status`，不要回退已有未提交工作。
3. 让用户先同步第 6 节的 6 个文件到真机。
4. 确认采集程序启动时显示 `10 帧`。
5. 距离采集和拟合已经完成，不要使用 `--overwrite` 重采现有数据。
6. 同步最新 `block_mono_grasp.yaml` 到真机。
7. 用 `--dry-run --show-rgb` 在不同距离检查单目距离误差，不要直接进入真机抓取。
8. 距离误差合格后才开始四类无 Tag 抓取示教。
9. 先验证单个物块，再做连续抓取。

遇到新故障时，先分类为“视觉没有产生可用定位”、“TF/标定误差”、“MoveIt 无解”或“底层没有真正到位”，不要同时修改视觉、抓取偏移和底层驱动。
