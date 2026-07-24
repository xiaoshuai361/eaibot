# 睿抗比赛机工程 AI 速览

更新时间：2026-05-24
用途：给 AI 或新接手的人快速理解这台比赛机的主车识别、巡线、模型和执行链路。

## 1. 一眼结论

1. 主车比赛主线在 /home/eaibot/robocom_ws/src。
2. 机械臂抓取主线在 /home/eaibot/mirobot_ws 和 /home/eaibot/handeye-calib。
3. 如果目标是理解“睿抗主车怎么识别和巡线”，先看 robocom_ws，不要先陷进机械臂。
4. 当前最值得优先看的主脚本是 /home/eaibot/robocom_ws/src/xjsb_lajitong.py。
5. 当前最值得优先看的巡线单项脚本是 /home/eaibot/robocom_ws/src/over.py。
6. 当前主识别模型是 people_best.pt、fire_best.pt、rubbish_best.pt。
7. 运动控制不是脚本直接发串口，而是脚本发 /cmd_vel，再走 xpkg_vehicle 和 xpkg_comm。

## 2. 工程地图

| 区域         | 目录                        | 主要职责                           |
| ------------ | --------------------------- | ---------------------------------- |
| 主车比赛工程 | /home/eaibot/robocom_ws/src | 巡线、视觉识别、停车检测、底盘控制 |
| 机械臂工程   | /home/eaibot/mirobot_ws     | MoveIt、机械臂驱动、模型与控制     |
| 抓取与标定   | /home/eaibot/handeye-calib  | 手眼标定、抓取脚本、AprilTag/TF    |

主车工程里最重要的是四类东西：

1. 综合主程序：xjsb_lajitong.py、xjsb.py、xjsb_6.py。
2. 单项巡线：over.py、double.py、tracking1.py。
3. 底盘 ROS 包：xpkg_bringup、xpkg_vehicle、xpkg_comm。
4. 模型权重：people_best.pt、fire_best.pt、rubbish_best.pt。

## 3. 主车综合主线脚本

### 3.1 当前最像正式主线的脚本

文件：/home/eaibot/robocom_ws/src/xjsb_lajitong.py

这个脚本同时做：

1. 巡线。
2. 人群识别。
3. 火灾/楼宇识别。
4. 垃圾桶识别。
5. 定时停车检测。
6. 发布 /cmd_vel 控车。

它内部直接内嵌了三个识别模块：

1. PeopleConfig：人群识别配置。
2. FireConfig：火灾识别配置。
3. RubbishDetector：垃圾桶识别器。

### 3.2 双摄像头分工

在 xjsb_lajitong.py 里：

1. cv2.VideoCapture(2) 用于巡线。
2. cv2.VideoCapture(0) 用于目标识别。

因此当前综合方案是双摄：

1. 2 号摄像头看地面黑线。
2. 0 号摄像头看人群、火灾、垃圾桶。

### 3.3 主线不是一直识别

脚本通过 detection_intervals 定义时间窗口，到点先停车，再做检测。
从代码可见：

1. 两次火灾检测窗口。
2. 四次人群检测窗口。
3. 一次垃圾桶检测窗口。
4. 第四次人群检测后进入 slow_mode。
5. 检测到双黑线后进入 ramp_mode。

结论：它是明显按赛题流程硬编码的时序脚本，不是通用导航框架。

## 4. 视觉识别代码与功能

| 识别任务   | 主代码位置                          | 当前主模型      | 作用                                          |
| ---------- | ----------------------------------- | --------------- | --------------------------------------------- |
| 人群识别   | xjsb_lajitong.py 内 PeopleConfig    | people_best.pt  | 区分职业人员和普通人员，统计 A/B/C/D 区域人数 |
| 火灾识别   | xjsb_lajitong.py 内 FireConfig      | fire_best.pt    | 识别建筑物、火点、楼宇类型，并估计火灾楼层    |
| 垃圾桶识别 | xjsb_lajitong.py 内 RubbishDetector | rubbish_best.pt | 识别垃圾桶类别和投放状态                      |

### 4.1 人群识别

模型：/home/eaibot/robocom_ws/src/people_best.pt

当前代码中的类别：

1. zhiye：职业人员。
2. putong：普通人员。

当前逻辑会：

1. 做 YOLOv5 检测。
2. 对近距离框去重。
3. 统计职业/普通人数。
4. 保存图片到 saved_people_images。
5. 在终端输出某区域的人数结果。

### 4.2 火灾与楼宇识别

模型：/home/eaibot/robocom_ws/src/fire_best.pt

当前代码里的可见类别：

1. Building：建筑物。
2. Fire：火点。
3. Meili：美丽商场。
4. DianZi：电子超市。

当前逻辑会：

1. 识别建筑框和火点框。
2. 平滑楼宇类型结果。
3. 根据建筑框内火点相对高度估计楼层。
4. 用 6 层区间输出火灾位置。
5. 保存图片到 saved_fire_images。

### 4.3 垃圾桶识别

模型：/home/eaibot/robocom_ws/src/rubbish_best.pt

当前代码显式映射了 8 个类别：

1. 可回收物\_未投放
2. 厨余垃圾\_已投放
3. 厨余垃圾\_未投放
4. 有害垃圾\_已投放
5. 有害垃圾\_未投放
6. 其他垃圾\_已投放
7. 其他垃圾\_未投放
8. 可回收物\_已投放

它会：

1. 做检测。
2. 统计每类数量。
3. 渲染结果图。
4. 保存到 saved_rubbish_images。

代码里还注释了“已修复标签颠倒问题”，说明类别映射不是原始训练顺序，而是修正后的使用顺序。

## 5. 当前有哪些模型权重

在 /home/eaibot/robocom_ws/src 里当前可见的 .pt 文件有：

1. people_best.pt
2. fire_best.pt
3. rubbish_best.pt
4. peoplebest.pt
5. light_best.pt
6. rubbish_best1111.pt
7. rubbish_best222.pt

可以按重要性分两类：

### 5.1 当前主线明确使用的

1. people_best.pt：人群识别。
2. fire_best.pt：火灾/楼宇识别。
3. rubbish_best.pt：垃圾桶识别。

### 5.2 更像旧版、备用或历史训练产物的

1. peoplebest.pt
2. light_best.pt
3. rubbish_best1111.pt
4. rubbish_best222.pt

如果 AI 要快速判断优先级，只关注前三个模型即可。

## 6. 模型加载方式

当前主线主要通过 torch.hub.load(...) 加载 YOLOv5。

优先逻辑：

1. 如果本地存在 yolov5-master，就从本地加载。
2. 如果本地没有，再尝试从 GitHub 拉。

因此 /home/eaibot/robocom_ws/src/yolov5-master 是当前主视觉识别链路的本地依赖。

结论：

1. 当前主识别框架是 YOLOv5。
2. 不是 PaddleDetection 在驱动当前主线。
3. Paddle 环境更像历史兼容或单独脚本需求。

## 7. 巡线代码分层

| 脚本             | 类型         | 图像来源                   | 功能定位                               |
| ---------------- | ------------ | -------------------------- | -------------------------------------- |
| xjsb_lajitong.py | 综合主线     | VideoCapture(2)            | 综合巡线 + 定时视觉识别                |
| over.py          | 单项稳定版   | /usb_cam/image_raw + /scan | 巡线 + 激光避障                        |
| double.py        | 纯视觉实验版 | VideoCapture(2)            | 黑线检测 + Kalman + PID + 124 秒后直行 |
| tracking1.py     | 最简巡线版   | /usb_cam/image_raw         | 黑线重心 + 增量 PID                    |

### 7.1 xjsb_lajitong.py 里的巡线算法

核心流程：

1. 读取 2 号摄像头。
2. HSV 提黑色。
3. 灰度 + 自适应阈值强化黑线。
4. 在 ROI 内按行找黑线点。
5. 对同一行像素分组，判断单线、双线、多线。
6. 用 KalmanFilter 平滑中线。
7. 用 PID 输出角速度。
8. 发布 /cmd_vel。

比赛特化逻辑：

1. 三条以上线时优先跟最右侧线。
2. 两条线时取中线或边界补偿。
3. 连续丢线时重置 Kalman 状态。
4. 慢速模式、坡道模式会改写运动策略。

### 7.2 over.py 的定位

这是最适合做“小车先跑起来”验证的脚本。

结构：

1. Follower 订阅 /usb_cam/image_raw，根据黑线控制。
2. LaserAvoid 订阅 /scan，丢线后切障碍规避。
3. 两部分都通过 /cmd_vel 控车。

适用场景：

1. 先验证黑线巡线是否正常。
2. 先验证雷达避障是否正常。
3. 不必一上来就进综合比赛主线。

### 7.3 double.py 的定位

这是纯视觉巡线试验版。
特点：

1. 直接读 2 号摄像头。
2. 黑线分组更细。
3. 有 Kalman + PID。
4. 运行 124 秒后切直行模式。

### 7.4 tracking1.py 的定位

这是最小巡线闭环。
只做：

1. 订阅 /usb_cam/image_raw。
2. 提取黑线。
3. 求重心。
4. 用增量 PID 发 /cmd_vel。

适合：快速确认摄像头、阈值、底盘响应是否通。

## 8. 运动控制链路

完整链路不是“视觉脚本 -> 串口”，而是：

视觉脚本 -> /cmd_vel -> xpkg_vehicle -> xpkg_comm -> 底盘控制器

### 8.1 xpkg_vehicle

文件：/home/eaibot/robocom_ws/src/xpkg_vehicle/src/xnode_vehicle.cpp

职责：

1. 订阅 /cmd_vel。
2. 转成底层车辆运动命令。
3. 发布 /odom。
4. 发布 odom -> base_link TF。
5. 把控制数据送给通信层。

### 8.2 xpkg_comm

文件：/home/eaibot/robocom_ws/src/xpkg_comm/src/xnode_comm.cpp

职责：

1. 把 ROS 数据打包成底层通信帧。
2. 通过通信接口发到底盘控制器。
3. 处理返回数据。
4. 检查底盘是否在线。

所以如果 /cmd_vel 正常但小车不动，不能只看 Python 脚本，还要一起看 xpkg_vehicle 和 xpkg_comm。

## 9. 机械臂在整机里的位置

机械臂不是主车巡线识别主线，但它是另一条完整子系统。

关键目录：

1. /home/eaibot/mirobot_ws
2. /home/eaibot/handeye-calib

关键高层脚本：

1. /home/eaibot/handeye-calib/src/mirobot_pick_test.py

支持：

1. --mode home
2. --mode pump
3. --mode grasp

如果 AI 当前任务只关心主车识别与巡线，可以先把机械臂部分放到第二优先级。

## 10. 推荐阅读顺序

如果先理解主车比赛逻辑，按这个顺序读：

1. /home/eaibot/robocom_ws/src/xjsb_lajitong.py
2. /home/eaibot/robocom_ws/src/over.py
3. /home/eaibot/robocom_ws/src/double.py
4. /home/eaibot/robocom_ws/src/tracking1.py
5. /home/eaibot/robocom_ws/src/xpkg_vehicle/src/xnode_vehicle.cpp
6. /home/eaibot/robocom_ws/src/xpkg_comm/src/xnode_comm.cpp

如果再理解机械臂抓取链路，继续读：

1. /home/eaibot/handeye-calib/src/mirobot_pick_test.py
2. /home/eaibot/mirobot_ws/src/mirobot_moveit_config
3. /home/eaibot/mirobot_ws/src/mirobot_urdf_2
4. /home/eaibot/mirobot_ws/src/apriltag_ros
5. /home/eaibot/handeye-calib/src/easy_handeye

## 11. 最后一行给 AI 的摘要

这台睿抗比赛机的主车代码核心不是复杂导航，而是“双摄像头巡线 + 定时视觉识别 + /cmd_vel 控车”的赛题流程系统；主视觉入口是 xjsb_lajitong.py，主巡线备用入口是 over.py，主模型是 people_best.pt、fire_best.pt、rubbish_best.pt，底盘执行链路是 xpkg_vehicle + xpkg_comm。
