# -*- coding: utf-8 -*-
"""全局配置：路径、模型类别、上传限制等"""
from pathlib import Path

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
CAMERA_INDEX = 0
CAMERA_FRAME_WIDTH = 1280
CAMERA_FRAME_HEIGHT = 720
CAMERA_RECORD_INTERVAL = 5.0  # 有目标时的最小落库间隔（秒），防止记录爆炸
