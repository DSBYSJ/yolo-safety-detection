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
    │   ├── templates/             # 7 个页面模板
    │   └── static/                # 样式 / ECharts / 检测结果图
    ├── scripts/
    │   └── make_demo_video.py     # 生成演示视频（用于视频检测测试）
    ├── tests/
    │   ├── test_unit.py           # 离线单元测试（35 项，无需服务/GPU）
    │   └── regression.py          # 端到端回归测试（21 项，需服务运行中）
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
- 摄像头检测需本机连接摄像头设备；
- 停止服务：终端按 `Ctrl+C`；
- 生产环境可用 `waitress-serve --host 0.0.0.0 --port 5000 webapp.app:app` 部署。

## 六之二、功能测试

测试分两层：**离线单元测试**不依赖服务与 GPU，可随时运行；**端到端回归测试**需要服务已启动。

### 1. 离线单元测试（35 项）

    python tests/test_unit.py

覆盖 `imageio_cn`（含中文路径读写）、`db`（增删查与统计聚合）、
`detector` 纯函数（kinds 过滤、类别计数映射）、`fontutil`（跨平台字体）、
`config`（模型类别与 `data.yaml` 一致性校验），共 35 项：

    ====================================================================
    单元测试结果: 35/35 通过
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
