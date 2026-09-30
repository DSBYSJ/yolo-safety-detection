# 基于 YOLOv8 的安全帽/口罩佩戴检测系统

目标检测实战项目 · Ultralytics YOLOv8 + Flask + OpenCV + SQLite + ECharts

系统包含 模型训练、模型评估、图片/视频检测、摄像头实时检测、检测记录管理、统计报表 六大模块，
覆盖「数据集构建 → 模型训练 → 模型评估 → Web 部署」的目标检测完整工程链路。

## 一、功能总览

| 模块 | 说明 |
| --- | --- |
| 在线检测 | 上传图片/视频，一键识别安全帽与口罩佩戴情况，输出标注图/标注视频与结构化明细 |
| 实时监控 | 调用本地摄像头或 RTSP 网络摄像头，YOLOv8 逐帧推理 + MJPEG 流回传，自动周期性存档检测记录；可选开启人脸识别并**按人判定佩戴合规**；**支持配置多路摄像头**（名称/类型/地址/位置）并可切换 |
| 手机检测 | 手机浏览器直接开摄像头，逐帧发给电脑推理（无需装 App），画面叠加实时检测框；人脸识别可选「关闭 / 每 2 秒 / 每帧」三档，并当场判定谁没戴 |
| 人脸底库 | 提交人脸照片并命名，建立人脸特征底库；识别时已录入显示姓名，未录入显示临时访客编号 |
| 合规统计 | **按「人」汇总**佩戴情况：安全帽与口罩分开统计、互不交叉；含**按人明细**（谁未戴几次、违规率）、已注册/未注册分组排行 |
| 检测记录 | 全部检测结果入库（SQLite），支持分页、筛选、详情查看、批量删除 |
| 统计报表 | 近 14 天趋势、类别分布、来源占比、置信度分布多维可视化（ECharts） |
| 模型训练 | 网页端配置数据集/模型规模/epochs 后一键启动训练，实时展示损失与 mAP 曲线、日志 |
| 模型评估 | 一键在验证集上评估，输出 P / R / mAP@0.5 / mAP@0.5:0.95、混淆矩阵、PR 曲线 |

两类检测任务（均为两分类）：

| 任务 | 类别 0（合规） | 类别 1（违规） | 数据集 | 规模 |
| --- | --- | --- | --- | --- |
| 安全帽检测 | helmet 安全帽 | head 未戴安全帽 | Roboflow "Hard Hats"（keremberke/hard-hat-detection） | 训练 15,783 / 验证 3,962 |
| 口罩检测 | mask 口罩 | face 未戴口罩 | maksssksksss（hmnshudhmn24/face-mask-detection） | 训练 682 / 验证 171 |

最终训练指标见 runs/ 目录或系统「模型评估」页（mAP@0.5 目标 ≥ 80%）。

## 二、项目结构

    yolo-safety-detection/
    ├── datasets/                  # 数据集
    │   ├── raw/                   # 原始下载数据
    │   ├── prepare_helmet.py      # 安全帽数据集构建（COCO→YOLO，划分）
    │   ├── prepare_mask.py        # 口罩数据集构建（VOC→YOLO，划分）
    │   ├── helmet/                # 转换后的 YOLO 数据集 + data.yaml
    │   └── mask/
    ├── train/
    │   ├── train.py               # 训练脚本（可独立命令行运行）
    │   └── val.py                 # 评估脚本（输出 metrics.json + 可视化）
    ├── pretrained/                # COCO 预训练权重 yolov8n/s.pt
    ├── models/                    # 部署用最优权重 helmet.pt / mask.pt
    │   └── face/models/buffalo_l/ # 人脸识别模型（det_10g / w600k_r50 / 2d106det）
    ├── runs/                      # 训练/评估产物（权重、曲线、混淆矩阵）
    ├── webapp/                    # Flask Web 应用
    │   ├── app.py                 # 主程序（路由/接口）
    │   ├── config.py              # 全局配置
    │   ├── detector.py            # YOLOv8 推理封装 + 中文标注绘制
    │   ├── camera.py              # 摄像头线程 + MJPEG 流 + 人脸识别
    │   ├── face_db.py             # 人脸底库：特征提取、比对识别、合规统计
    │   ├── phone_face.py          # 手机端人脸识别：模式判定、节流、结果整理（与 Flask 解耦）
    │   ├── equip.py               # 装备归属：把安全帽/口罩框按空间重叠配到具体的人
    │   ├── camera_db.py           # 监控摄像头配置：增删改查、地址校验与凭据脱敏
    │   ├── video_jobs.py          # 视频后台检测任务（含 H.264 转码）
    │   ├── train_manager.py       # 在线训练/评估任务管理
    │   ├── db.py                  # SQLite 记录管理
    │   ├── imageio_cn.py          # 图像读写（兼容非 ASCII 路径）
    │   ├── fontutil.py            # 跨平台中文字体加载
    │   ├── templates/             # 10 个页面模板（含手机端 /phone、人脸底库、合规统计）
    │   └── static/                # 样式 / ECharts / 检测结果图 / 人脸缩略图
    ├── scripts/
    │   ├── make_demo_video.py     # 生成演示视频（用于视频检测测试）
    │   ├── list_cameras.py        # 枚举本机设备索引 / 探测网络流地址（接入手机摄像头时用）
    │   └── check_mobile.py        # 移动端适配验证（手机视口逐页截图 + 溢出检测）
    ├── tests/
    │   ├── test_unit.py           # 离线单元测试（139 项，无需服务/GPU）
    │   └── regression.py          # 端到端回归测试（27 项，需服务运行中）
    ├── requirements.txt
    └── README.md

## 三、环境安装

    # 1. 创建虚拟环境（Python ≥ 3.9，推荐 3.10+）
    python -m venv .venv
    .venv\Scripts\activate        # Windows

    # 2. 安装 PyTorch（NVIDIA 显卡装 CUDA 版；仅 CPU 推理则直接 pip install torch）
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

    # 3. 安装其余依赖
    pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

## 四、数据集构建（已提供一键脚本）

    # 安全帽（约 1.1GB，需先下载三个 zip 到 datasets/raw/helmet/）
    python datasets/prepare_helmet.py

    # 口罩（自动从 HF 镜像下载 853 张图与标注）
    python datasets/prepare_mask.py --download

脚本自动完成：解压 → 类别映射（hardhat→helmet、no-hardhat→head；with_mask→mask、
without_mask 与 mask_weared_incorrect→face）→ COCO/VOC 标注转 YOLO txt → 划分 → 生成 data.yaml。

## 五、模型训练与评估

    # 命令行训练（推荐 yolov8s）
    python train/train.py --data datasets/helmet/data.yaml --model pretrained/yolov8s.pt --epochs 80 --name helmet_s --deploy-name helmet
    python train/train.py --data datasets/mask/data.yaml   --model pretrained/yolov8s.pt --epochs 80 --name mask_s   --deploy-name mask

    # 评估（生成 metrics.json / 混淆矩阵 / PR 曲线）
    python train/val.py --weights models/helmet.pt --data datasets/helmet/data.yaml --name helmet
    python train/val.py --weights models/mask.pt   --data datasets/mask/data.yaml   --name mask

也可以直接启动 Web 系统后在「模型训练 / 模型评估」页面完成同样操作。
RTX 4060 上 yolov8s / imgsz 640 / batch 16 的参考速度：安全帽数据集约 3~4 分钟/轮，口罩数据集约 10 秒/轮。

## 六、启动 Web 系统

### 方式一：一键脚本（推荐）

    scripts\run_web.bat          # Windows，双击或命令行执行

### 方式二：命令行

    cd webapp
    python app.py
    # 浏览器访问 http://127.0.0.1:5000

启动成功后终端会打印计算设备：

    ============================================================
      安全帽/口罩佩戴检测系统  |  YOLOv8 + Flask
      计算设备: CUDA (GPU)
      访问地址: http://127.0.0.1:5000
    ============================================================

- 首次使用请先训练，或把训练好的 helmet.pt / mask.pt 放入 models/ 目录；
- `webapp/app.py` 内已把工作目录处理为绝对路径，从任何目录启动均可；
- 摄像头检测需本机连接摄像头设备（含虚拟摄像头，见「六之三」）；
- 停止服务：终端按 `Ctrl+C`；
- 生产环境可用 `waitress-serve --host 0.0.0.0 --port 5000 webapp.app:app` 部署。

### 可用的环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `CAMERA_SOURCE` | 空 | 网络视频流地址（RTSP/HTTP/RTMP）。设置后**优先于** `CAMERA_INDEX` |
| `CAMERA_INDEX` | `1` | 本地摄像头设备索引。**默认 1 是因为本机索引 0 打不开**，换机器务必先跑 `scripts/list_cameras.py` 枚举 |
| `CAMERA_FRAME_WIDTH` | `1280` | 本地摄像头采集宽度（对网络流无效，流分辨率由推流端决定） |
| `CAMERA_FRAME_HEIGHT` | `720` | 本地摄像头采集高度（同上） |
| `CAMERA_STREAM_TIMEOUT` | `8` | 打开网络流时的超时秒数，失败后自动重试 |
| `FLASK_SSL_CERT` | 空 | HTTPS 证书路径。与 `FLASK_SSL_KEY` 同时设置才启用 HTTPS |
| `FLASK_SSL_KEY` | 空 | HTTPS 私钥路径 |
| `SECRET_KEY` | 开发默认值 | Flask 会话密钥，生产环境务必注入 |

    # 把摄像头切到索引 1 的设备
    CAMERA_INDEX=1 python webapp/app.py
    # 改用手机推的 RTSP 流
    CAMERA_SOURCE="rtsp://192.168.1.100:554/h264" python webapp/app.py
    # 启用 HTTPS（手机端调摄像头需要）
    FLASK_SSL_CERT=certs/cert.pem FLASK_SSL_KEY=certs/key.pem python webapp/app.py
    # Windows CMD
    set CAMERA_INDEX=1 && python webapp/app.py

## 六之三、接入监控摄像头（海康威视等 RTSP 设备）

**为什么推荐用 RTSP 摄像头而不是电脑自带摄像头**：笔记本内置镜头视角窄、
画质低、位置固定，拍不到真实的工地/车间场景；而 IP 摄像头能装在需要监控的
位置，且这本就是安防场景的实际部署方式。项目**无需改代码**即可接入 ——
`camera.py` 的 `resolve_source()` 已支持 RTSP/HTTP/RTMP，只要给出流地址即可。

### ⚠️ 先确认设备索引（不然会误以为「检测不到摄像头」）

程序默认 `CAMERA_INDEX=1`，但**不同机器的可用索引不一样**。
遇到「未检测到摄像头」时，**先枚举，别猜**：

```bash
python scripts/list_cameras.py           # 扫描 0-5 号设备
python scripts/list_cameras.py --max 8   # 扫更多
python scripts/list_cameras.py --save    # 存首帧图，确认打开的是哪个镜头
```

输出示例：

```
[索引 0]  不可用                ← 打不开（MSMF: can't grab frame）
[索引 1]  可用  640x480        ← 真实摄像头在这里
```

拿到可用索引后，改 `config.CAMERA_INDEX` 默认值，**并在 `run_web.bat` 里
也显式 `set CAMERA_INDEX=<n>`**（环境变量优先于源码默认值，双保险）。

### 方式一：RTSP 网络摄像头（海康威视 / 大华 / 宇视等）

1. 确认摄像头与运行本系统的电脑在**同一局域网**；
2. 在摄像头后台开启 RTSP（海康默认端口 `554`）；
3. 拼出取流地址：

| 品牌 | 地址格式 |
| --- | --- |
| 海康威视 | `rtsp://admin:密码@摄像头IP:554/Streaming/Channels/101` |
| | `101` = 主码流（高清）；`102` = 子码流（更流畅，检测够用） |
| 大华 | `rtsp://admin:密码@摄像头IP:554/cam/realmonitor?channel=1&subtype=0` |

4. 启动服务时指定：

```bash
# Windows CMD
set CAMERA_SOURCE=rtsp://admin:密码@192.168.1.64:554/Streaming/Channels/102
scripts\run_web.bat

# Git Bash / Linux
CAMERA_SOURCE="rtsp://admin:密码@192.168.1.64:554/Streaming/Channels/102" \
  python webapp/app.py
```

**先验证地址再启动服务**（能取到画面再往下走）：

```bash
python scripts/list_cameras.py --url "rtsp://admin:密码@192.168.1.64:554/Streaming/Channels/102" --save
```

### 方式二：手机推流（HTTP，无需 IP 摄像头）

1. 手机装 **IP Webcam**（Android，免费）；
2. 打开 App 点「启动服务器」，记下地址（形如 `http://手机IP:8080`）；
3. 取流地址在其后加 `/video`：

```bash
set CAMERA_SOURCE=http://192.168.1.100:8080/video
scripts\run_web.bat
```

### 页面上的接入引导

未接入摄像头时，「实时监控」页会在 6 秒内自动显示**接入引导面板**
（含上述两种方式的分步说明与地址格式），不需要对着黑屏排查。
页面同时会显示**当前视频源**：本地设备显示「设备索引 N」，
网络流显示脱敏后的地址（**密码以 `***` 代替，不会明文回显**）。

### 常见问题

| 现象 | 原因与处理 |
| --- | --- |
| 改了 `CAMERA_INDEX`/`CAMERA_SOURCE` 不生效 | 抓帧线程已启动时不会重开设备，**必须重启服务** |
| 提示未检测到摄像头 | 先跑 `scripts/list_cameras.py` 确认索引，别猜默认值 0 |
| RTSP 连不上 | 检查同网段、RTSP 是否开启、端口是否正确；先用浏览器/VLC 试地址 |
| 网络流延迟高 | 属正常。RTSP 比 MJPEG(HTTP) 延迟低；优先 5GHz Wi-Fi 或有线 |

## 六之五、人脸识别与合规统计

在安全帽/口罩检测之外，系统可以再识别「这个人是谁」，并把佩戴情况**按人汇总**。

### 1. 它能做什么

| 场景 | 表现 |
| --- | --- |
| 已在「人脸底库」录入照片并命名 | 摄像头画面里直接标出**姓名**（青色框） |
| 未录入的人 | 标为**临时访客编号**（橙色框），如 `访客-1` |
| 合规统计页 | 按人汇总出现次数；安全帽与口罩**分开统计，互不交叉** |

### 2. 用之前必须先放模型（否则页面会提示「模型未就绪」）

人脸识别依赖 InsightFace 的 `buffalo_l` 模型包，目录结构是固定的：

    models/face/
      └── models/
          └── buffalo_l/
              ├── det_10g.onnx      人脸检测（约 16MB）
              ├── w600k_r50.onnx    特征提取，认人的核心（约 170MB）
              └── 2d106det.onnx     关键点（约 5MB）

⚠️ **国内直连 github / huggingface 下载这些文件会超时**（本项目实测全部连不上）。
若换机器部署，从国内镜像取：

    # 用 hf-mirror 镜像（实测 0.3s 响应，速度可达 14MB/s）
    BASE=https://hf-mirror.com/public-data/insightface/resolve/main/models/buffalo_l
    curl -L -o det_10g.onnx   $BASE/det_10g.onnx
    curl -L -o w600k_r50.onnx $BASE/w600k_r50.onnx
    curl -L -o 2d106det.onnx  $BASE/2d106det.onnx

依赖安装（两个包，insightface 2.0 已是纯 Python wheel，无需编译）：

    pip install insightface onnxruntime

### 3. ⚠️ 一个必须知道的现实：戴口罩会认不准

人脸特征提取依赖**面部下半部分**的细节，而戴口罩正好把它盖住了。
这不是程序缺陷，是这套技术路线的固有限制（InsightFace 官方文档也说明了
「人脸被遮挡（口罩、墨镜）会导致特征点缺失、置信度低」）。

所以实际表现会是：

- 画面里**没戴口罩**的人 → 识别正常
- 画面里**戴着口罩**的人 → 很可能识别不出，回落为「访客-N」

**这是安全的降级**（认不出比认错好），但如果你要的是「戴口罩也能认人」，
需要换用口罩场景专门训练过的识别模型，不是调个参数能解决的。

### 4. 录入流程

1. 进「**人脸底库**」页 → 点上传区选一张**清晰正脸照**
2. 填姓名（必填）+ 备注（可选，如部门/工号）→ 点「提交并注册」
3. 不确定效果？点「**试识别这张照片**」可立即看到相似度与命中判定
4. 之后在「实时监控」页点「**开启人脸识别**」即可

### 5. 阈值与参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| 匹配阈值 | `0.38` | 余弦相似度。**偏严**是刻意的：误把张三认成李四，比偶尔认不出更糟 |
| `FACE_INTERVAL` | `2.0` | 人脸识别执行间隔（秒）。人脸走 CPU 约 200–400ms，逐帧跑会把画面拖成幻灯片 |

调整阈值需改 `webapp/face_db.py` 的 `MATCH_THRESHOLD`。

### 6. 常见问题

| 现象 | 原因与处理 |
| --- | --- |
| 提交照片提示「未检测到人脸」 | 换更清晰的正脸；侧脸、逆光、遮挡都会检不到 |
| 页面提示「模型未就绪」 | 上面第 2 步的模型文件没放全，检查目录层级（是 `models/face/models/buffalo_l/`） |
| 明明录入过却认成访客 | 拍照与现场的光照/角度差异大；或现场戴了口罩（见第 3 节） |
| 两个人被发到同一个访客编号 | 属归并过宽。归并阈值 `_TEMP_MERGE_THRESHOLD` 可在 `face_db.py` 调低 |
| 开启识别后画面变卡 | 调大 `FACE_INTERVAL`（如 3.0），牺牲刷新率换流畅度 |

### 5. 按人判定佩戴合规（equip.py）

原来的合规统计只能回答「本帧有几个未戴安全帽」，
**回答不了「谁没戴」** —— 因为人脸识别和装备检测是两条互不相干的线。

`equip.py` 补上了这层关联：安全帽、口罩都是**同一张脸上的装备**，
位置强相关，因此按**空间重叠**（IoU）把人脸框与装备框配对：

```
┌──────────┐  ← 安全帽框：在人脸上方，水平大致对齐
│  安全帽  │
├──────────┤  ← 人脸框
│ 口│ 罩   │  ← 口罩框：覆盖人脸下半部
└──────────┘
```

判定规则：

| 装备 | 配对条件 |
|------|----------|
| 口罩 | 与人脸框 IoU ≥ `IOU_THRESHOLD`（0.10） |
| 安全帽 | 同上；若 IoU 不足，额外允许「水平对齐 + 帽底在人脸中部以上 + 帽底不得离人脸超过 1.5 倍脸高」 |

> 安全帽那条宽松规则是必需的：帽子比脸大一圈且偏上，典型 IoU 只有 0.05 左右，
> 只认 IoU 会「明明戴了却判没戴」。但**上界同样必需** ——
> 只判断「在上方 + 水平对齐」的话，画面顶部的帽子会被配给画面下方任意一个人
> （写测试时真的抓到了这个 bug：一个离人脸 290px 的帽子被错误配对）。

#### ⭐ 三态设计：`None` 不是 `0`

`face_seen` 的装备字段有**三种**取值，这是本功能最关键的设计：

| 值 | 含义 |
|----|------|
| `1` | 确实佩戴 |
| `0` | **确实检测到未佩戴** |
| `NULL` | 本帧**没有**该类装备框 → 无信息 |

**为什么必须区分 `NULL` 和 `0`**：摄像头没对准、模型漏检、画面里只有半个人，
都会导致没有装备框。若一律记 `0`（未戴），员工会被大面积误判成违规。

统计「违规率」时分母只算**有信息**的次数（`hat_state IS NOT NULL`）。
没有信息时违规率返回 `None`，页面显示「—」而不是 `0%` ——
「从没拍到过帽子」不等于「100% 合规」。

## 六之六、监控摄像头配置管理（多路）

实时监控页的「＋ 添加监控摄像头」支持录入多路摄像头（名称 / 类型 / 地址 / 位置 / 备注），
存在 SQLite（`cameras` 表），可随时切换、测试连通、编辑、删除，**不必再改环境变量并重启**。

| 类型 | 地址示例 | 说明 |
|------|----------|------|
| `rtsp` | `rtsp://admin:密码@10.0.0.8:554/Streaming/Channels/102` | 网络摄像头，工地最常见 |
| `http` | `http://10.0.0.8:8080/video` | MJPEG 流（手机推流 App） |
| `device` | `0` | 本机 USB 摄像头 / 采集卡索引 |

⚠️ **凭据脱敏是硬要求**：RTSP 地址含明文密码，列表接口一律返回
`rtsp://***:***@10.0.0.8:554/...`，只有建立连接时才取原文（`resolve_url()`）。

首次启动会把环境变量里那一路导入配置表（仅在表为空时执行），
因此**旧部署的行为完全不变** —— 没配置过就还走 `CAMERA_SOURCE`。

切换摄像头后前端必须换一个带时间戳的 `feed.src` 才会重连
（浏览器不会因为 src 没变就重连 MJPEG 流）。

## 六之四、使用手机摄像头做实时监测

手机的摄像头**不能直接被 OpenCV 读取** —— `cv2.VideoCapture` 只认系统里注册的
视频设备（Windows 走 DirectShow，Linux 走 V4L2），或一个明确给出的流地址。
因此有两条路：把手机**伪装成虚拟摄像头**（方案 A/B，改 `CAMERA_INDEX`），
或**直接读取手机推的视频流**（方案 C，改 `CAMERA_SOURCE`）。两者都无需改动源码。

### 方案 A：DroidCam / Iriun Webcam（推荐，最省事）

1. 手机安装 **DroidCam**（或 Iriun Webcam），电脑安装对应客户端；
2. 二者连接（Wi-Fi 填手机显示的 IP:端口；或 USB 走 ADB，延迟更低）；
3. 连接成功后电脑会多出一个虚拟摄像头设备；
4. 枚举设备索引，确认新设备是几号：

       python scripts/list_cameras.py

5. 用探到的索引启动：

       CAMERA_INDEX=1 python webapp/app.py

参考延迟：720p < 100ms，1080p 约 200ms（5GHz Wi-Fi 或 USB）。
注意 DroidCam 免费版画面带水印且分辨率受限。

### 方案 B：Android 14+ / Windows 11 原生 UVC

较新机型在「开发者选项」中开启 USB 摄像头类功能后，插上 USB 即被系统识别为
标准摄像头，无需第三方 App。同样用 `list_cameras.py` 确认索引即可。

### 方案 C：IP Webcam 等网络流（原生支持，无需装虚拟摄像头驱动）

手机装 **IP Webcam**（或任意提供 RTSP 的网络摄像机）后，会得到一个流地址。
直接用 `CAMERA_SOURCE` 指定即可：

1. 先在手机上启动服务，记下地址（IP Webcam 默认形如
   `http://手机IP:8080/video`，RTSP 形如 `rtsp://手机IP:8080/h264_pcm.sdp`）；
2. 用脚本探测该地址是否可达（会打印分辨率并保存一帧实拍图）：

       python scripts/list_cameras.py --url "http://192.168.1.100:8080/video" --save

3. 用探测到的地址启动：

       CAMERA_SOURCE="http://192.168.1.100:8080/video" python webapp/app.py

相比方案 A 的好处：**不用在电脑装虚拟摄像头驱动**，且天然支持接入网络摄像机。
代码侧已处理断流重连 —— 网络流允许连续丢 5 帧才判定断线重连（本地设备是 2 帧），
避免一次网络抖动就重建连接导致画面闪烁。

### 常见问题

| 现象 | 原因与处理 |
| --- | --- |
| 画面提示「未检测到摄像头（设备索引 N）」 | 索引不对。运行 `scripts/list_cameras.py` 确认正确索引 |
| 画面提示「无法连接视频流 + 地址」 | 地址不通。确认手机与电脑同一局域网、App 已启动、防火墙未拦截 |
| 改了 `CAMERA_INDEX` / `CAMERA_SOURCE` 但没生效 | 抓帧线程已启动时不会重新打开设备，**需重启服务** |
| 虚拟摄像头连接成功但画面黑屏 | 先确认客户端预览正常；再试 `list_cameras.py --backend dshow` |
| 手机掉线后画面不恢复 | 抓帧线程每 5 秒自动重连，确认手机端 App 仍在运行、IP 未变 |
| 网络流延迟明显高于虚拟摄像头 | 属正常现象。RTSP 比 MJPEG(HTTP) 延迟低；优先用 5GHz Wi-Fi 或网线 |

## 六之四、手机端页面与浏览器直接检测

以上「六之三」是**手机当摄像机给电脑用**。本节是另一条路：
**手机直接打开网页，用手机自己的摄像头做检测**（无需装任何 App、无需推流）。

### 1. 打开方式

手机浏览器访问 `http://<电脑局域网IP>:5000/phone`，或在侧边栏点「手机检测」。

页面里点「启动摄像头」授权后即可看到实时检测画面与违规计数。
右上角显示实时帧率，底部 chips 显示每类目标数量与合规判断。

### 2. ⚠️ 摄像头权限的 HTTPS 门槛

浏览器出于安全策略，**只在 HTTPS 或 localhost 下才允许网页访问摄像头**。
所以直接用 `http://192.168.x.x:5000/phone` 访问时，Chrome/Safari 会拒绝授权。
三种解法（按推荐度排序）：

| 方案 | 做法 | 适用 |
| --- | --- | --- |
| **自签证书 + HTTPS** | 用 `openssl` 生成证书，Flask 以 `ssl_context` 启动 | 局域网长期使用，一劳永逸 |
| **Chrome 白名单** | 打开 `chrome://flags/#unsafely-treat-insecure-origin-as-secure`，填入本站地址后重启 | 临时调试最快 |
| **换浏览器** | 部分国产浏览器对 HTTP 摄像头限制较松 | 应急 |

生成自签证书并用 HTTPS 启动（示例）：

    # 1. 生成证书（有效期 365 天，CN 与 SAN 都填你的局域网 IP）
    mkdir certs
    openssl req -x509 -newkey rsa:2048 -nodes -keyout certs/key.pem -out certs/cert.pem \
      -days 365 -subj "/CN=192.168.1.100" \
      -addext "subjectAltName=IP:192.168.1.100,DNS:localhost"

    # 2. 用 HTTPS 启动
    FLASK_SSL_CERT=certs/cert.pem FLASK_SSL_KEY=certs/key.pem python webapp/app.py

    # Windows CMD
    set FLASK_SSL_CERT=certs/cert.pem && set FLASK_SSL_KEY=certs/key.pem && python webapp/app.py

手机首次访问会提示「证书不受信任」，选择「继续访问」即可（自签证书的必然提示）。
启动日志会打印出手机可访问的完整地址，直接照着输入即可。

⚠️ **`certs/` 已加入 `.gitignore`** —— 里面是私钥，绝对不要提交到仓库。

### 3. 实现要点

- 手机浏览器用 `getUserMedia` 抓画面 → 压成 JPEG（长边 960px）→ base64 → POST `/api/phone/detect`
- 服务端解码后在 GPU 上推理，返回计数与合规判断；**实测单帧推理 36~48ms，往返中位 46ms**
- 前端限流到约 **4 帧/秒**（间隔 250ms），留足服务端处理时间，流畅且不让手机发烫
- **预览不落库**：只有点「抓拍存档」才会写一条检测记录，避免逐帧产生上千条垃圾数据
- 页面切到后台自动暂停，回来继续（省电、不空占 GPU）
- 支持前后摄像头切换（默认后置，拍工地场景更合适）
- **人脸识别可选三档节奏**（关闭 / 每 2 秒 / 每帧），详见「六之六」

### 3.1 手机端人脸识别

与人脸底库、合规统计联动：识别到的人会在画面上叠一个姓名框
（**青框=底库已注册，橙框=未注册访客编号**），并显示在计数区。

**为什么节流是必需的**：人脸识别走 CPU（约 200~400ms/帧），
而手机端画面本身只有约 4fps。若每帧都跑人脸，单帧耗时会涨到 400ms 以上，
画面直接掉成幻灯片。因此：

| 模式 | 行为 | 适用 |
|------|------|------|
| 关闭 | 只跑安全帽/口罩 | 默认，最省 |
| **每 2 秒**（`interval`） | 服务端按 `FACE_INTERVAL` 秒节流 | **推荐**，画面流畅又能看到姓名 |
| 每帧（`every`） | 每帧都识别 | 短暂定点确认某人时用 |

节流状态按**客户端**分别记录（前端用 `localStorage` 生成 `client` id 传给服务端），
否则两台手机会互相顶掉对方的计时。

**两个容易踩的细节**（都已在代码里处理）：

1. **姓名不能逐帧清空**。`interval` 模式下大部分帧`faces` 是空数组，
   若每帧都照它重绘，姓名会以 4Hz 频率闪烁。所以前端用 `lastFaces`
   跨帧保持，只在真正跑识别的那帧更新。
2. **存档必须强制认人**。点「抓拍存档」时无论当前选哪档模式都会识别一次
   （服务端 `save=1` 时无条件放行），因为这张图会写进合规统计。

识别结果**只在存档时落库**，与安全帽/口罩保持一致 —— 逐帧写 `face_seen`
会在几秒内产生上千条记录，把「合规统计」页冲垮。

实现集中在 `webapp/phone_face.py`（模式判定 + 节流 + 结果整理），
刻意与 Flask 解耦，便于脱离服务直接做单元测试。

### 4. 并发安全：为什么推理必须串行、服务必须用 cheroot

⚠️ **这是踩过的真坑，务必了解。**

**症状**：手机页面用一会儿后，**整个网站所有页面都不再响应**（浏览器显示
`ERR_TIMED_OUT`），但服务进程还在、端口仍是 `LISTENING`、日志停在某一刻再无输出。

**根因有三层，缺一层就会复发**：

**第一层：多线程并发调用同一个 YOLO 模型对象**

1. 手机端逐帧请求，每个请求占一个工作线程；
2. 同时「实时监测」页的 `camera._worker` 后台线程也在跑推理；
3. `detector.infer_image()` 原本**没有加锁**，多个线程同时进入
   `model.predict()`，在 CUDA 上下文里互相踩踏 → PyTorch 内部死锁。

**第二层：前端「假超时」导致请求无限泄漏**

前端原本用 `Promise.race` 做超时 —— **但不会取消已发出的 fetch**：

```
超时 → busy 置回 false → 下一帧又发新请求
     → 超时那个请求仍挂在服务端 → 连接持续累积
```

**第三层：Flask 开发服务器在 HTTPS 下不回收连接（最致命）**

客户端正常关闭（FIN）时，正忙于推理的服务端来不及 `close()`，
`CLOSE_WAIT` 不断堆积。实测 **`CLOSE_WAIT` 从 4 涨到 17**，进程彻底无响应。
这是**用 Flask 自带 server 跑 HTTPS 的固有缺陷**，加锁和前端修复都救不了。

**修复（四层，缺一不可）**：

| 层 | 文件 | 做法 |
|---|---|---|
| 推理串行 | `detector.py` | 全局 `_infer_lock`，所有推理入口统一排队 |
| 接口快速失败 | `app.py` | 手机端接口 `timeout=2`，等不到锁返回 `503 {"skip":true}` |
| 前端真取消 | `phone.html` | `AbortController` 真中断 fetch；指数退避；canvas 复用 |
| **换服务器** | `app.py` | **默认用 cheroot**（原生 SSL + 固定线程池 + 连接可回收）|

> **为什么必须换服务器**：Flask 开发服务器定位是调试工具，其 `threaded`
> 模式对 SSL 的处理会在客户端断开后留下 `CLOSE_WAIT`。
> cheroot（CherryPy 内核）通过 `BuiltinSSLAdapter` 原生接管 SSL，
> 连接生命周期由它管理，能正确回收。
>
> 备选是 waitress，但 **waitress 3.x 的 `Adjustments` 不接受 `ssl_context`**
> （传了报 `Unknown adjustment 'ssl_context'`），要接 HTTPS 得自己包 SSL socket，反而更绕。

**实测数据（高压 + 断连攻击）**：

| 检查项 | Flask 开发服务器 | cheroot |
|---|---|---|
| `CLOSE_WAIT` 峰值 | **17**（进程死）| **0** |
| 30 次客户端硬断连后 | 堆积 | **零堆积** |
| 16 并发 × 32 请求 | 全部挂死 | 25 个 200 + 7 个按预期 503 |
| 压力后页面连通性 | 全部超时 | **6/6 全部 200** |
| 内存 | 1420MB 后不响应 | 1281MB 稳定 |

**内存澄清**：进程常驻 1.2~1.3GB 是 **PyTorch + CUDA 的固定基线**，不是泄漏。
实测 120 次推理仅增长 **+2MB**，20 次后即完全持平。

> 结论一：**只要涉及 GPU 推理的接口，都不能裸奔在多线程里。**
>
> 结论二：**前端做超时一定要配 `AbortController`。**
> 只 `Promise.race` 是假超时 —— 请求还在跑，泄漏会悄悄累积到把服务打死。
>
> 结论三：**别用 Flask 自带 server 跑 HTTPS 生产流量。**
> 调试可以，长期运行请用 cheroot。

### 5. 回归测试支持 HTTPS

`tests/regression.py` 已内置自签证书豁免，可直接对 HTTPS 实例测试：

    python tests/regression.py --base https://127.0.0.1:5000

### 6. 移动端适配

全站 8 个页面均已适配手机：

- 侧边栏在 ≤900px 变成**抽屉式**，左上角汉堡按钮开合，带遮罩与滑入动画
- 表格统一包在横向滚动容器里（检测记录有 11 列，手机上必须横滑）
- 按钮、导航项触控高度 ≥44px；图表高度自动降低避免占满整屏
- 超小屏（≤640px）环境徽章只留圆点，顶栏不拥挤
- 适配刘海屏安全区（`env(safe-area-inset-top)`）
- 横屏手机单独调优，打印时隐藏导航

用脚本自查适配效果（手机视口逐页截图 + 横向溢出检测）：

    python scripts/check_mobile.py
    python scripts/check_mobile.py --only /phone --outdir ./shots

它会用 Chrome DevTools Protocol 精确把视口设成 390×844，
测量 `scrollWidth` 是否超过视口，并指出具体是哪个元素溢出。

## 六之二、功能测试

测试分两层：**离线单元测试**不依赖服务与 GPU，可随时运行；**端到端回归测试**需要服务已启动。

### 1. 离线单元测试（139 项）

    python tests/test_unit.py

覆盖 `imageio_cn`（含中文路径读写）、`db`（增删查与统计聚合）、
`detector` 纯函数（kinds 过滤、类别计数映射）、`fontutil`（跨平台字体）、
`config`（模型类别与 `data.yaml` 一致性、环境变量解析）、
`camera`（取流来源解析、取流地址脱敏、人脸框绘制）、手机端 base64 解码兼容性、
推理串行锁（并发安全）、人脸底库（阈值判定、临时编号稳定性、底库增删、统计聚合），
共 139 项：

    ====================================================================
    单元测试结果: 139/139 通过
    ====================================================================

> 人脸比对相关的测试用**构造的单位向量**而不是真实人脸照片：
> 真实照片的相似度随光照/角度浮动，写死数值会变成脆弱测试；
> 而阈值判断是纯数学逻辑，用正交/同向向量可精确验证「该命中」与「该拒绝」。
>
> 手机端人脸识别的测试（`TestPhoneFaceModes`，17 项）刻意把节流逻辑抽到
> `phone_face.py` 里单独验证 —— 逐帧请求是高频路径，节流一旦写反，
> 要么「关了还在跑」（画面卡到没法用），要么「开了永远不跑」（用户以为坏了），
> 这两种故障在手工点击时都很难稳定复现。

### 2. 端到端回归测试（27 项）

服务启动后，另开一个终端执行：

    python tests/regression.py
    python tests/regression.py --base https://127.0.0.1:5000    # HTTPS（自签证书已豁免）

覆盖 8 个页面可达性、运行环境识别、图片检测（单类 / 双类）、结果图落盘校验、
手机端检测接口、异常处理、检测记录读写、统计报表、模型评估指标
（含 ≥80% 达标断言）共 27 项，输出如下：

    ====================================================================
    回归结果: 27/27 通过
    ====================================================================

可选参数：

    python tests/regression.py --base http://127.0.0.1:5000 --conf 0.25

> 回归测试会向数据库写入若干条测试记录（结果图也会落盘到 `static/results/`）。
> 若希望保持记录干净，测试后可在「检测记录」页批量删除。

### 3. 持续集成

`.github/workflows/ci.yml` 在每次 push / PR 时自动执行：
Python 3.10 与 3.12 双版本下进行**语法检查 → 模块导入校验 → 离线单元测试 → 服务冒烟测试**
（启动服务并逐一探测 9 个页面/接口的 HTTP 状态）。

### 手工测试要点

| 模块 | 操作 | 预期 |
| --- | --- | --- |
| 在线检测 | 检测页勾选「安全帽」，上传含安全帽的图片 | 返回标注图，明细表列出每个目标类别与置信度，记录自动入库 |
| 在线检测 | 勾选「口罩」，上传佩戴口罩的图片 | 检出 mask / face 两类，不混入安全帽结果 |
| 在线检测 | 两类同时勾选 | `kinds` 记为 `helmet+mask`，两类结果合并输出 |
| 在线检测 | 拖动某一类的置信度滑块后再检测 | 该类的检出数量随阈值升高而减少 |
| 视频检测 | 上传 `scripts/demo_helmet.mp4` | 进度条实时推进，完成后可在线播放（已转 H.264） |
| 摄像头检测 | 打开实时监控页 | MJPEG 画面流畅，可切换检测类型、抓拍存档 |
| 检测记录 | 筛选 / 分页 / 查看详情 / 批量删除 | 均正常，删除有二次确认 |
| 统计报表 | 打开统计页 | 概览指标 + 趋势 / 类别 / 来源 / 置信度四类图表正常渲染 |
| 模型训练 | 训练页选数据集与轮数后启动 | 子进程启动，损失与 mAP 曲线、日志实时刷新 |
| 模型评估 | 评估页发起评估 | 输出 P / R / mAP 指标与混淆矩阵、PR 曲线 |

> 注意：检测接口的检测类型参数名是 **`kinds`**（可重复提交），不是 `kind`。
> 用 curl 手工测试时写作 `-F "kinds=helmet" -F "kinds=mask"`。
> 置信度阈值参数为 **`conf`**（单个浮点数，对本次请求涉及的模型统一生效）。

## 七、开发注意事项

### 图像读写一律使用 `webapp/imageio_cn.py`

OpenCV 的 `cv2.imread` / `cv2.imwrite` 在 Windows 下遇到**中文路径会静默失败**：
读返回 `None`、写返回 `False`，且不抛异常。若项目路径含中文字符，
直接调用将导致检测结果图不落盘、前端显示裂图，而接口仍返回 `ok: true`，问题极难定位。

因此**禁止直接使用 `cv2.imread` / `cv2.imwrite`**，改用：

    import imageio_cn
    img = imageio_cn.imread(path)          # 兼容中文路径
    imageio_cn.imwrite(path, annotated)    # 返回 bool，失败可感知
    imageio_cn.imread_bytes(jpeg_bytes)    # 从内存字节解码

底层用 `np.fromfile` + `cv2.imdecode` / `cv2.imencode` + `Path.write_bytes` 绕开路径问题，
对 ASCII 路径同样适用。`cv2.VideoWriter` 不受此影响（`isOpened()` 会真实反馈）。

**写图必须检查返回值**：`imageio_cn.imwrite` 返回 `bool`，调用方需据此决定是否落库/返回成功。
`camera.py` 与 `app.py` 均已按「写盘失败则不落库、不报成功」处理，
避免出现「接口成功但结果图不存在」的裂图问题。

### 中文字体统一走 `webapp/fontutil.py`

`cv2.putText` 不支持中文，检测标签与摄像头占位提示都改用 PIL 绘制，需要显式指定字体文件。
早期实现硬编码了 `C:/Windows/Fonts/msyh.ttc`，在 Linux / macOS 上会静默回退到 PIL 默认字体，
导致**中文全部渲染成方块**。现统一由 `fontutil.get_font(size)` 提供，
按「Windows → macOS → Linux」顺序探测候选字体并按字号缓存；
全部未命中时向 `stderr` 打印一次性告警，便于定位。

    import fontutil
    font = fontutil.get_font(18)
    fontutil.available()   # 返回实际命中的字体路径，可用于启动自检

## 八、常见问题

| 问题 | 处理 |
| --- | --- |
| 提示未找到模型权重 | 先训练，或将 best.pt 复制为 models/helmet.pt / models/mask.pt |
| 检测完成但结果图显示裂图 | 确认结果图已落盘（`static/results/`）。若目录为空，检查是否误用了 `cv2.imwrite`，见「七、开发注意事项」 |
| 检测框标签中文显示为方块 | 系统缺少中文字体，见「七、开发注意事项 → 中文字体」。Linux 可装 `fonts-wqy-zenhei` 或 Noto Sans CJK |
| 视频检测结果无法播放 | 系统会自动调用 ffmpeg（imageio-ffmpeg）转 H.264，确认依赖已安装；也可点「下载结果视频」本地播放 |
| 摄像头黑屏/未检测到 | 检查设备占用（其他软件），或修改 webapp/config.py 的 CAMERA_INDEX |
| 训练显存不足 | 调小 batch（如 8）或 imgsz（如 512） |

## 九、参考

- Ultralytics YOLOv8: https://docs.ultralytics.com/
- Flask: https://flask.palletsprojects.com/
- OpenCV: https://docs.opencv.org/master/d6/d00/tutorial_py_root.html
