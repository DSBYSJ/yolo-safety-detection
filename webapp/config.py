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
}

ALLOWED_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}

MAX_CONTENT_LENGTH = 300 * 1024 * 1024  # 单文件上传上限 300MB

# 摄像头实时检测
# CAMERA_INDEX 为 OpenCV VideoCapture 的设备索引（整数）：
#   0 = 通常是笔记本内置摄像头
#   1/2 = 外接 USB 摄像头，或 DroidCam / Iriun 等虚拟摄像头
# 不确定时先运行 `python scripts/list_cameras.py` 枚举本机设备。
# 可用环境变量覆盖：CAMERA_INDEX=1 python webapp/app.py
CAMERA_INDEX = _env_int("CAMERA_INDEX", 0)
CAMERA_FRAME_WIDTH = _env_int("CAMERA_FRAME_WIDTH", 1280)
CAMERA_FRAME_HEIGHT = _env_int("CAMERA_FRAME_HEIGHT", 720)
CAMERA_RECORD_INTERVAL = 5.0  # 有目标时的最小落库间隔（秒），防止记录爆炸
