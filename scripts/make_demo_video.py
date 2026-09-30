# -*- coding: utf-8 -*-
"""生成演示视频：从安全帽数据集中抽取图片拼接成 mp4，用于测试视频检测链路

用法：python scripts/make_demo_video.py [输出路径] [时长秒数]
"""
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "datasets" / "helmet" / "images" / "val"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "scripts" / "demo_helmet.mp4"
SECONDS = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0

FPS = 25
W, H = 1280, 720


def main() -> None:
    imgs = sorted(IMG_DIR.glob("*.jpg"))
    if not imgs:
        raise SystemExit(f"未找到图片: {IMG_DIR}")
    random.seed(7)
    picks = [random.choice(imgs) for _ in range(max(4, int(SECONDS)))]
    per_img = int(FPS * SECONDS / len(picks))

    writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for p in picks:
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        ih, iw = img.shape[:2]
        scale = min(W / iw, H / ih)
        img = cv2.resize(img, (int(iw * scale), int(ih * scale)))
        canvas = np.full((H, W, 3), 16, dtype=np.uint8)
        y0, x0 = (H - img.shape[0]) // 2, (W - img.shape[1]) // 2
        canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
        for _ in range(per_img):
            writer.write(canvas)
    writer.release()
    print(f"[done] 演示视频已生成: {OUT} ({OUT.stat().st_size / 1048576:.1f} MB)")


if __name__ == "__main__":
    main()
