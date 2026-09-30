# -*- coding: utf-8 -*-
"""图像读写工具：解决 OpenCV 在非 ASCII（中文）路径下的静默失败问题。

背景
----
``cv2.imread`` / ``cv2.imwrite`` 在 Windows 下会把路径交给底层 C 接口处理，
遇到中文、日文等非 ASCII 字符时会**静默失败**：读返回 None，写返回 False，
且不抛出任何异常。若工作区路径含中文字符，
所有结果图、截图的落盘都会受影响，表现为前端 ``<img>`` 显示裂图。

方案
----
- 读：先以二进制读入内存，再用 ``cv2.imdecode`` 解码；
- 写：先用 ``cv2.imencode`` 编码到内存，再用 ``Path.write_bytes`` 落盘。

这样完全绕开 OpenCV 对路径的处理，对 ASCII 路径同样适用，因此可以全局替换。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path, flags: int = cv2.IMREAD_COLOR):
    """读图，兼容中文路径。失败返回 None（与 cv2.imread 行为一致）。"""
    path = Path(path)
    if not path.exists():
        return None
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    img = cv2.imdecode(buf, flags)
    return img


def imread_bytes(data: bytes, flags: int = cv2.IMREAD_COLOR):
    """从内存字节串解码图像（用于摄像头帧缓冲）。"""
    if not data:
        return None
    buf = np.frombuffer(data, dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def imwrite(path, img, ext: str | None = None) -> bool:
    """写图，兼容中文路径。返回是否成功（与 cv2.imwrite 行为一致）。

    ext 用于指定编码格式（如 ".jpg"）；不传时从路径后缀推断。
    """
    path = Path(path)
    ext = (ext or path.suffix or ".jpg").lower()
    if not ext.startswith("."):
        ext = "." + ext
    if img is None:
        return False
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(buf.tobytes())
        return True
    except OSError:
        return False


def write_jpeg_bytes(img, quality: int = 82) -> bytes | None:
    """把 BGR 图像编码为 JPEG 字节串（内存态，供 MJPEG 流使用）。"""
    if img is None:
        return None
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return buf.tobytes() if ok else None
