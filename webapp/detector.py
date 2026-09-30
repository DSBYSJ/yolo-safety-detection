# -*- coding: utf-8 -*-
"""YOLOv8 推理封装：图片/视频帧检测 + 中文标签绘制

约定（与 datasets/*/data.yaml 一致）：
    helmet 模型: 0 = helmet 安全帽（合规）  1 = head 未戴安全帽（违规）
    mask   模型: 0 = mask   口罩（合规）    1 = face 未戴口罩（违规）
"""
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

import config
import fontutil

_model_cache: dict = {}

# 推理串行锁：YOLO/PyTorch 的模型对象不是线程安全的，多个线程同时调用
# model.predict() 会在 CUDA 上下文里互相踩踏，极端情况直接死锁把整个进程冻住
# （现象：端口还在 LISTENING，但所有请求都不再响应）。
# 因此所有推理入口（摄像头线程、图片上传、手机端逐帧）统一排队执行。
_infer_lock = threading.Lock()

# 绘制颜色（BGR）：合规 = 绿色系，违规 = 红色系
COLORS = {
    "helmet": (75, 180, 60),
    "head": (60, 60, 225),
    "mask": (75, 180, 60),
    "face": (60, 60, 225),
}


class ModelMissingError(RuntimeError):
    """模型权重缺失"""


class InferBusy(RuntimeError):
    """推理通道繁忙：等锁超时。调用方应放弃本次检测而不是继续排队。"""


def _font():
    """标签绘制用中文字体（跨平台，见 fontutil）。"""
    return fontutil.get_font(18)


def model_path(kind: str) -> Path:
    """解析模型权重路径：优先 models/<kind>.pt，其次 runs 下最新训练产物。"""
    p = config.MODEL_DIR / config.MODELS[kind]["file"]
    if p.exists():
        return p
    if config.RUNS_DIR.exists():
        cands = sorted(
            config.RUNS_DIR.glob(f"{kind}*/weights/best.pt"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if cands:
            return cands[0]
    raise ModelMissingError(
        f"未找到「{config.MODELS[kind]['title']}」模型权重，"
        f"请先在【模型训练】页完成训练，或将权重放置到 {p}"
    )


def get_model(kind: str) -> YOLO:
    """按 kind 取模型，进程内缓存。

    注意：调用方应已持有 _infer_lock。这里的缓存是为了避免每帧都重新
    从磁盘加载权重（一次约数百毫秒），而不是为了并发安全。
    """
    if kind in _model_cache:
        return _model_cache[kind]
    model = YOLO(str(model_path(kind)))
    _model_cache[kind] = model
    return model


def detect_kinds(kinds) -> list:
    valid = [k for k in (kinds or []) if k in config.MODELS]
    return valid or ["helmet"]


def counts_to_db(counts: dict) -> dict:
    """类别名计数 -> 数据库四列"""
    return {
        "helmet": counts.get("helmet", 0),
        "head": counts.get("head", 0),
        "mask": counts.get("mask", 0),
        "face": counts.get("face", 0),
    }


def infer_image(img_bgr: np.ndarray, kinds, conf: float = 0.25, iou: float = 0.45,
                timeout: float | None = None):
    """对一帧图像执行检测（多线程安全，内部串行化）。

    返回 (detections, counts, annotated_bgr, time_ms)
        detections: [{kind, class, class_cn, conf, box:[x1,y1,x2,y2]}]
        counts:     {类别英文名: 数量}

    timeout: 等待推理锁的最长秒数。超时抛 InfeBusy 而不排队，
            用于手机端逐帧检测 —— 宁肯丢这一帧，也不能让请求越堆越多。
    """
    acquired = _infer_lock.acquire(timeout=timeout) if timeout else _infer_lock.acquire()
    if not acquired:
        raise InferBusy("推理通道繁忙，请稍后重试")
    try:
        t0 = time.time()
        detections: list = []
        counts: dict = {}
        annotated = img_bgr
        for kind in detect_kinds(kinds):
            model = get_model(kind)
            result = model.predict(img_bgr, conf=conf, iou=iou, verbose=False)[0]
            boxes = result.boxes
            meta = config.MODELS[kind]
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                c = float(boxes.conf[i].item())
                x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i].tolist())
                name = model.names.get(cls_id, str(cls_id))
                detections.append(
                    {
                        "kind": kind,
                        "class": name,
                        "class_cn": meta["names_cn"].get(cls_id, name),
                        "conf": round(c, 3),
                        "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    }
                )
                counts[name] = counts.get(name, 0) + 1
            annotated = _draw(annotated, boxes, model.names, meta)
        time_ms = round((time.time() - t0) * 1000, 1)
        return detections, counts, annotated, time_ms
    finally:
        _infer_lock.release()


def _draw(img_bgr, boxes, names, meta) -> np.ndarray:
    """绘制检测框与中文标签（cv2 不支持中文，用 PIL 绘制）。"""
    if len(boxes) == 0:
        return img_bgr
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _font()
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        c = float(boxes.conf[i].item())
        x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i].tolist())
        name = names.get(cls_id, str(cls_id))
        cn = meta["names_cn"].get(cls_id, name)
        bgr = COLORS.get(name, (60, 60, 225))
        rgb = (bgr[2], bgr[1], bgr[0])
        draw.rectangle([x1, y1, x2, y2], outline=rgb, width=3)
        label = f"{cn} {c:.2f}"
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = y1 - th - 8
        if ty < 2:
            ty = y1 + 2
        draw.rectangle([x1, ty, x1 + tw + 10, ty + th + 8], fill=rgb)
        draw.text((x1 + 5, ty + 2), label, fill=(255, 255, 255), font=font)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
