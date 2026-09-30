# 基于 YOLOv8 的安全帽/口罩佩戴检测系统

目标检测实战项目 · Ultralytics YOLOv8 + Flask + OpenCV + SQLite + ECharts

系统包含 模型训练、模型评估、图片/视频检测、摄像头实时检测、检测记录管理、统计报表 六大模块，
覆盖「数据集构建 → 模型训练 → 模型评估 → Web 部署」的目标检测完整工程链路。

## 一、功能总览

| 模块 | 说明 |
| --- | --- |
| 在线检测 | 上传图片/视频，一键识别安全帽与口罩佩戴情况，输出标注图/标注视频与结构化明细 |
| 实时监控 | 调用本地摄像头，YOLOv8 逐帧推理 + MJPEG 流回传，自动周期性存档检测记录 |
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
    ├── runs/                      # 训练/评估产物（权重、曲线、混淆矩阵）
    ├── webapp/                    # Flask Web 应用
    │   ├── app.py                 # 主程序（路由/接口）
    │   ├── config.py              # 全局配置
    │   ├── detector.py            # YOLOv8 推理封装 + 中文标注绘制
    │   ├── camera.py              # 摄像头线程 + MJPEG 流
    │   ├── video_jobs.py          # 视频后台检测任务（含 H.264 转码）
    │   ├── train_manager.py       # 在线训练/评估任务管理
    │   ├── db.py                  # SQLite 记录管理
    │   ├── imageio_cn.py          # 图像读写（兼容非 ASCII 路径）
    │   ├── fontutil.py            # 跨平台中文字体加载
    │   ├── templates/             # 8 个页面模板（含手机端 /phone）
    │   └── static/                # 样式 / ECharts / 检测结果图
    ├── scripts/
    │   ├── make_demo_video.py     # 生成演示视频（用于视频检测测试）
    │   ├── list_cameras.py        # 枚举本机设备索引 / 探测网络流地址（接入手机摄像头时用）
    │   └── check_mobile.py        # 移动端适配验证（手机视口逐页截图 + 溢出检测）
    ├── tests/
    │   ├── test_unit.py           # 离线单元测试（51 项，无需服务/GPU）
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
| `CAMERA_INDEX` | `0` | 本地摄像头设备索引。接入手机虚拟摄像头时通常要改成 1 或 2 |
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

## 六之三、使用手机摄像头做实时监测

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
- 服务端解码后在 GPU 上推理，返回计数与合规判断；**实测稳态往返约 97ms（≈10 fps）**
- 前端限流到约 **8 帧/秒**，既流畅又不让手机发烫
- **预览不落库**：只有点「抓拍存档」才会写一条检测记录，避免逐帧产生上千条垃圾数据
- 页面切到后台自动暂停，回来继续（省电、不空占 GPU）

### 4. 并发安全：为什么推理必须串行

⚠️ **这是踩过的真坑，务必了解。**

现象：手机页面用一会儿后，**整个网站所有页面都不再响应**（浏览器显示超时），
但服务进程还在、端口仍是 `LISTENING`、日志停在某一刻再无输出。

根因是**多线程并发调用同一个 YOLO 模型对象**：

1. 手机端 8fps 逐帧请求，每个请求是一个 Flask 工作线程；
2. 同时「实时监测」页的摄像头后台线程也在跑推理；
3. `detector.infer_image()` 原本**没有加锁**，多个线程同时进入
   `model.predict()`，在 CUDA 上下文里互相踩踏 → PyTorch 内部死锁 → 所有线程卡住。

修复（三层）：
- **`detector.py` 加全局 `_infer_lock`**：所有推理入口统一串行执行，
  这是根本修复；
- **手机端接口 `timeout=3` 快速失败**：等不到锁就返回 `503 {"skip": true}`，
  **宁可丢这一帧，也不让请求排队堆积**；
- **前端指数退避**：连续失败时把间隔从 120ms 逐步退到最多 2s，
  避免服务端卡顿时被雪崩式重试打死；单帧还有 8 秒超时保护。

实测（12 并发 × 24 请求）：**全部 3116ms 内返回**，其中 7 个按预期返回 503，
压力期间普通页面响应仍为 **13ms**。单手机 8fps 场景零丢帧、中位延迟 99ms。

> 结论：**只要涉及 GPU 推理的接口，都不能裸奔在多线程里。**
> 要么加锁串行，要么换单线程队列模型（如 waitress + 单 worker）。

### 5. 回归测试支持 HTTPS

`tests/regression.py` 已内置自签证书豁免，可直接对 HTTPS 实例测试：

    python tests/regression.py --base https://127.0.0.1:5000
- 支持前后摄像头切换（默认后置，拍工地场景更合适）

### 4. 移动端适配

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

### 1. 离线单元测试（46 项）

    python tests/test_unit.py

覆盖 `imageio_cn`（含中文路径读写）、`db`（增删查与统计聚合）、
`detector` 纯函数（kinds 过滤、类别计数映射）、`fontutil`（跨平台字体）、
`config`（模型类别与 `data.yaml` 一致性、环境变量解析）、
`camera`（取流来源解析），共 46 项：

    ====================================================================
    单元测试结果: 46/46 通过
    ====================================================================

### 2. 端到端回归测试（21 项）

服务启动后，另开一个终端执行：

    python tests/regression.py

覆盖 7 个页面可达性、运行环境识别、图片检测（单类 / 双类）、结果图落盘校验、
异常处理、检测记录读写、统计报表、模型评估指标（含 ≥80% 达标断言）共 21 项，输出如下：

    ====================================================================
    回归结果: 21/21 通过
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
