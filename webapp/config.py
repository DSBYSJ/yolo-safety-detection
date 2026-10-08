# -*- coding: utf-8 -*-
"""全局配置：路径、模型类别、上传限制等

部分配置支持环境变量覆盖，便于在不改动源码的前提下切换运行环境，
例如把摄像头从笔记本内置镜头切到手机虚拟摄像头（DroidCam / Iriun）：

    CAMERA_INDEX=1 python webapp/app.py
"""
import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    """读取整数环境变量，解析失败时回退默认值（不让配置错误导致启动崩溃）。"""
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


def _env_str(name: str, default: str = "") -> str:
    """读取字符串环境变量，空串视为未设置。"""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip()


WEBAPP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEBAPP_DIR.parent

DB_PATH = WEBAPP_DIR / "data.db"

UPLOAD_DIR = WEBAPP_DIR / "uploads"
RESULT_DIR = WEBAPP_DIR / "static" / "results"
TRAIN_LOG_DIR = WEBAPP_DIR / "train_logs"
for _d in (UPLOAD_DIR, RESULT_DIR, TRAIN_LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
MODEL_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "runs"
PRETRAINED_DIR = PROJECT_ROOT / "pretrained"

# ------------------------------------------------------------------ 人脸识别
# 底库特征提取用 InsightFace 的 buffalo_l 模型包（人脸检测 + 512 维特征）。
# 目录结构必须是 <FACE_MODEL_ROOT>/models/<name>/，这是 insightface 的约定：
#   models/face/
#     └── models/
#         └── buffalo_l/
#             ├── det_10g.onnx     人脸检测（16MB）
#             ├── w600k_r50.onnx   特征提取（170MB，认人的核心）
#             └── 2d106det.onnx    关键点（可选，用于对齐）
#
# ⚠️ 国内直连 github / huggingface 下载模型会超时，模型已预置在仓库目录内。
#    若换成新机器部署，需先把上述文件放到该目录，否则识别功能会明确报错。
FACE_MODEL_ROOT = MODEL_DIR / "face"

FACE_THUMB_DIR = WEBAPP_DIR / "static" / "faces"

# 两类检测模型与类别定义（类别 id 与 datasets/*/data.yaml 保持一致）
MODELS = {
    "helmet": {
        "title": "安全帽检测",
        "file": "helmet.pt",
        "names": {0: "helmet", 1: "head"},
        "names_cn": {0: "安全帽", 1: "未戴安全帽"},
    },
    "mask": {
        "title": "口罩检测",
        "file": "mask.pt",
        "names": {0: "mask", 1: "face"},
        "names_cn": {0: "口罩", 1: "未戴口罩"},
    },
    # PPE 综合防护检测（新增类别：安全帽 + 安全背心 + 人员）
    "ppe": {
        "title": "PPE 防护装备检测",
        "file": "ppe.pt",
        "names": {0: "helmet", 1: "no-helmet", 2: "no-vest", 3: "person", 4: "vest"},
        "names_cn": {0: "安全帽", 1: "未戴安全帽", 2: "未穿安全背心", 3: "人员", 4: "安全背心"},
    },
}

ALLOWED_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}

MAX_CONTENT_LENGTH = 300 * 1024 * 1024  # 单文件上传上限 300MB

# ------------------------------------------------------------------ 类别名汇总
# 把全部模型的「类别中文名」和「违规类别」汇总成全局映射，
# 供前端渲染标签/标红使用；以后在 MODELS 里新增模型会自动生效，不用再改前端。
CLASS_CN: dict = {}
for _meta in MODELS.values():
    CLASS_CN.update(_meta["names_cn"])

# 判为「违规」的类别英文名（前端标红）
BAD_CLASSES = {"head", "face", "no-helmet", "no-vest"}

# 摄像头实时检测
# 取流地址解析优先级：CAMERA_SOURCE（网络流）> CAMERA_INDEX（本地设备索引）
#
# 方式一：本地摄像头 / 虚拟摄像头（DroidCam、Iriun 等），用整数索引
#   CAMERA_INDEX = 0 通常是笔记本内置摄像头
#   CAMERA_INDEX = 1/2 是外接 USB 摄像头或虚拟摄像头
#   不确定时先运行 `python scripts/list_cameras.py` 枚举本机设备。
#
#   ⚠️ 本机实测：索引 0 打不开（MSMF 报 can't grab frame），
#      真实摄像头在索引 1，所以默认值取 1。
#      换机器或插拔设备后索引可能变，务必先跑上面的枚举脚本确认。
#
# 方式二：网络视频流（手机 App「IP Webcam」、网络摄像机、RTSP 摄像头等）
#   CAMERA_SOURCE=rtsp://user:pass@192.168.1.100:554/h264
#   CAMERA_SOURCE=http://192.168.1.100:8080/video
#   注意：设了 CAMERA_SOURCE 后 CAMERA_INDEX 与分辨率设置会被忽略，
#   因为网络流的分辨率由推流端决定。
CAMERA_INDEX = _env_int("CAMERA_INDEX", 1)
CAMERA_FRAME_WIDTH = _env_int("CAMERA_FRAME_WIDTH", 1280)
CAMERA_FRAME_HEIGHT = _env_int("CAMERA_FRAME_HEIGHT", 720)
CAMERA_RECORD_INTERVAL = 5.0  # 有目标时的最小落库间隔（秒），防止记录爆炸

# 网络流地址（RTSP / HTTP / RTMP）。留空则使用本地摄像头索引。
CAMERA_SOURCE = _env_str("CAMERA_SOURCE", "")

# 打开网络流时的超时（秒）。无人响应时不必一直等，便于失败后自动重试。
CAMERA_STREAM_TIMEOUT = _env_int("CAMERA_STREAM_TIMEOUT", 8)

# 摄像头实时人脸识别的执行间隔（秒）。
# 人脸识别走 CPU，单次约 200-400ms，若逐帧执行会把抓帧循环拖成幻灯片
# （安全帽/口罩检测的 fps 会从 100+ 掉到个位数）。
# 按间隔执行可让识别结果稳定跟随，中间帧沿用上次结果，观感是「名字挂着」。
FACE_INTERVAL = float(os.environ.get("FACE_INTERVAL", "2.0") or 2.0)
