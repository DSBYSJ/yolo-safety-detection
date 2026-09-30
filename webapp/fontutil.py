# -*- coding: utf-8 -*-
"""跨平台中文字体加载（供检测框标签、摄像头占位图共用）。

背景
----
OpenCV 的 ``cv2.putText`` 不支持中文，会渲染成问号或方块，因此检测标签与
占位提示都改用 PIL 的 ``ImageDraw.text`` 绘制。PIL 需要显式指定字体文件，
而各操作系统中文字体位置不同，早期实现只写了 ``C:/Windows/Fonts/msyh.ttc``，
在 Linux / macOS 上会静默回退到 ``ImageFont.load_default()``，
导致中文全部显示为方块。

方案
----
按「Windows → macOS → Linux」顺序遍历候选字体，命中即缓存复用。
找不到任何中文字体时回退到 PIL 默认字体，并给出一次性告警，
便于使用者察觉「中文显示为方块」的原因。
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import ImageFont

# 按平台优先级排列的中文字体候选路径
_CANDIDATES = [
    # Windows
    "C:/Windows/Fonts/msyh.ttc",       # 微软雅黑
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",     # 黑体
    "C:/Windows/Fonts/simsun.ttc",     # 宋体
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    # Linux
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]

# 字号 -> 字体对象（PIL 字体对象与字号绑定，需按字号缓存）
_cache: dict[int, ImageFont.FreeTypeFont] = {}
_warned = False


def _pick(size: int) -> ImageFont.FreeTypeFont | None:
    """按候选顺序尝试加载，返回首个可用的字体对象。"""
    for path in _CANDIDATES:
        if not Path(path).exists():
            continue
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return None


def get_font(size: int = 18) -> ImageFont.FreeTypeFont:
    """获取指定字号的中文字体，结果按字号缓存。

    找不到中文字体时返回 PIL 默认字体（中文会显示为方块），
    并在首次发生时向 stderr 打印一次性提示。
    """
    global _warned
    if size in _cache:
        return _cache[size]

    font = _pick(size)
    if font is None:
        if not _warned:
            print(
                "[fontutil] 未找到可用的中文字体，标签中文将显示为方块。"
                "可安装如 fonts-wqy-zenhei / Noto Sans CJK 等字体，"
                "或在 fontutil.py 的 _CANDIDATES 中补充字体路径。",
                file=sys.stderr,
            )
            _warned = True
        font = ImageFont.load_default()

    _cache[size] = font
    return font


def available() -> str | None:
    """返回实际命中的字体文件路径（未命中返回 None），供启动自检使用。"""
    for path in _CANDIDATES:
        if not Path(path).exists():
            continue
        try:
            ImageFont.truetype(path, 12)
            return path
        except OSError:
            continue
    return None
