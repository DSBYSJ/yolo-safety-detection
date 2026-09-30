# -*- coding: utf-8 -*-
"""摄像头实时检测：单例抓帧线程 + MJPEG 输出 + 周期性落库"""
import threading
import time
import uuid

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
import db
import detector
import fontutil
import imageio_cn

_lock = threading.Lock()
_thread = None
_stop = False
_state = {
    "jpeg": None,
    "ok": False,          # 摄像头是否正常
    "kind": "helmet",     # 当前检测类型
    "fps": 0.0,
    "counts": {},         # 最近一帧各类别计数
    "time": 0.0,
}


def _placeholder(text: str) -> np.ndarray:
    img = np.full((480, 720, 3), 28, dtype=np.uint8)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    d.text((360, 220), text, fill=(180, 190, 205), font=fontutil.get_font(26), anchor="mm")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _publish(frame_or_jpeg_bytes, ok=None, counts=None, fps=None):
    with _lock:
        if isinstance(frame_or_jpeg_bytes, (bytes, bytearray)):
            _state["jpeg"] = bytes(frame_or_jpeg_bytes)
        else:
            _state["jpeg"] = cv2.imencode(".jpg", frame_or_jpeg_bytes, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        if ok is not None:
            _state["ok"] = ok
        if counts is not None:
            _state["counts"] = counts
        if fps is not None:
            _state["fps"] = fps
        _state["time"] = time.time()


def _save_record(counts: dict, annotated: np.ndarray, kind: str, tms: float) -> None:
    """把当前帧落盘并写入一条检测记录。

    必须先确认结果图真正写入成功再落库，否则记录会指向不存在的图片，
    前端表现为裂图。写盘失败时直接跳过本次落库（不抛异常，避免中断抓帧循环）。
    """
    name = f"camera_{uuid.uuid4().hex[:10]}.jpg"
    out = config.RESULT_DIR / name
    if not imageio_cn.imwrite(out, annotated):
        return
    cols = detector.counts_to_db(counts)
    db.insert_record(
        source_type="camera",
        source_name="摄像头实时流",
        kinds=kind,
        num_objects=sum(counts.values()),
        duration_ms=tms,
        image_path=f"results/{name}",
        **cols,
    )


def _worker() -> None:
    cap = None
    last_record = 0.0
    while not _stop:
        # 打开/重试摄像头
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(config.CAMERA_INDEX)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_FRAME_HEIGHT)
            if not cap.isOpened():
                _publish(_placeholder("未检测到摄像头 / 摄像头被占用"), ok=False, counts={})
                time.sleep(5.0)
                continue
            _publish(_placeholder("摄像头已连接，正在启动检测..."), ok=True)

        ok, frame = cap.read()
        if not ok:
            cap.release()
            cap = None
            continue

        with _lock:
            kind = _state["kind"] or "helmet"
        try:
            _, counts, annotated, tms = detector.infer_image(frame, [kind])
        except Exception:
            counts, annotated, tms = {}, frame, 0.0
        fps = round(1000.0 / tms, 1) if tms else 0.0

        now = time.time()
        if now - last_record >= config.CAMERA_RECORD_INTERVAL:
            last_record = now
            if sum(counts.values()) > 0:
                try:
                    _save_record(counts, annotated, kind, tms)
                except Exception:
                    pass

        _publish(annotated, ok=True, counts=counts, fps=fps)
        time.sleep(0.005)

    if cap is not None:
        cap.release()


def start(kind: str = "helmet") -> None:
    """确保抓帧线程运行，并切换检测类型。"""
    global _thread, _stop
    with _lock:
        _state["kind"] = kind if kind in config.MODELS else "helmet"
    if _thread is None or not _thread.is_alive():
        _stop = False
        _thread = threading.Thread(target=_worker, daemon=True)
        _thread.start()


def set_kind(kind: str) -> None:
    with _lock:
        _state["kind"] = kind if kind in config.MODELS else "helmet"


def get_state() -> dict:
    with _lock:
        s = dict(_state)
        s.pop("jpeg", None)
        return s


def mjpeg_generator():
    """MJPEG 流生成器"""
    import itertools

    for _ in itertools.count():
        with _lock:
            jpeg = _state.get("jpeg")
        if jpeg:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )
        time.sleep(0.04)


def snapshot():
    """对当前帧截图并保存为一条检测记录，返回记录 id 与图片路径。"""
    with _lock:
        jpeg = _state.get("jpeg")
        kind = _state.get("kind")
        counts = dict(_state.get("counts") or {})
        ok = _state.get("ok")
    if not jpeg or not ok:
        return None

    arr = imageio_cn.imread_bytes(jpeg)
    if arr is None:
        return None
    name = f"camera_snap_{uuid.uuid4().hex[:10]}.jpg"
    if not imageio_cn.imwrite(config.RESULT_DIR / name, arr):
        return None
    cols = detector.counts_to_db(counts)
    rid = db.insert_record(
        source_type="camera",
        source_name="摄像头抓拍",
        kinds=kind,
        num_objects=sum(counts.values()),
        image_path=f"results/{name}",
        **cols,
    )
    return rid, f"results/{name}"
