# -*- coding: utf-8 -*-
"""枚举本机可用摄像头，帮助确定 config.CAMERA_INDEX 该填几。

用途
----
接入手机摄像头（DroidCam / Iriun 等虚拟摄像头）后，它的设备索引通常不是 0，
需要先枚举出来再通过环境变量指定：

    python scripts/list_cameras.py              # 扫描 0-5 号设备
    python scripts/list_cameras.py --max 8      # 扫描 0-7 号设备
    python scripts/list_cameras.py --save       # 同时把每个设备的实拍图存到 tests/
    CAMERA_INDEX=1 python webapp/app.py         # 用探测到的索引启动

Windows 上 OpenCV 走 DirectShow 后端，设备顺序与「设备管理器 → 成像设备」一致。
"""
import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]


def probe(index: int, backend: int | None = None):
    """尝试打开指定索引的设备并抓一帧，返回 (是否可用, 分辨率, 帧图像)。"""
    cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
    try:
        if not cap.isOpened():
            return False, None, None
        # 有些虚拟摄像头首帧需要预热，多读几次
        frame = None
        for _ in range(5):
            ok, f = cap.read()
            if ok and f is not None and f.size > 0:
                frame = f
                break
        if frame is None:
            return False, None, None
        h, w = frame.shape[:2]
        return True, (w, h), frame
    finally:
        cap.release()


def main() -> int:
    ap = argparse.ArgumentParser(description="枚举本机可用摄像头")
    ap.add_argument("--max", type=int, default=6, help="扫描的索引上限（默认 6，即 0-5）")
    ap.add_argument("--save", action="store_true", help="把实拍首帧保存到 tests/ 便于确认画面")
    ap.add_argument("--backend", choices=["dshow", "msmf", "any"], default="any",
                    help="Windows 下用的后端（默认 any，即 OpenCV 自动选择）")
    args = ap.parse_args()

    backend = None
    if args.backend == "dshow":
        backend = cv2.CAP_DSHOW
    elif args.backend == "msmf":
        backend = cv2.CAP_MSMF

    print("=" * 62)
    print("  摄像头设备枚举")
    print("=" * 62)
    print(f"  扫描范围: 0 - {args.max - 1}")
    print()

    found = []
    for i in range(args.max):
        ok, size, frame = probe(i, backend)
        if ok:
            found.append((i, size))
            print(f"  [索引 {i}]  可用    分辨率 {size[0]}x{size[1]}")
            if args.save:
                out = ROOT / "tests" / f"camera_probe_{i}.jpg"
                try:
                    cv2.imwrite(str(out), frame)
                    print(f"              首帧已保存: {out}")
                except Exception as e:  # noqa: BLE001
                    print(f"              首帧保存失败: {e}")
        else:
            print(f"  [索引 {i}]  不可用")

    print()
    print("-" * 62)
    if not found:
        print("  未发现任何可用摄像头。")
        print("  排查建议：")
        print("    1. 确认摄像头已连接，且未被其他程序（会议/直播软件）占用；")
        print("    2. 若用 DroidCam/Iriun，先启动手机 App 与电脑客户端并确认已连接；")
        print("    3. 可尝试换后端：--backend dshow")
        print("=" * 62)
        return 1

    print(f"  共发现 {len(found)} 个可用设备：{', '.join(str(i) for i, _ in found)}")
    print()
    print("  使用方法：")
    print(f"    CAMERA_INDEX={found[0][0]} python webapp/app.py")
    if len(found) > 1:
        print(f"    或（切换另一个设备）")
        print(f"    CAMERA_INDEX={found[1][0]} python webapp/app.py")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
