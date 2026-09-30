# -*- coding: utf-8 -*-
"""枚举本机可用摄像头，或探测网络视频流，帮助确定 config 该怎么配。

用途
----
接入手机摄像头有两条路，本脚本两种都能验证：

**路径一：虚拟摄像头（DroidCam / Iriun）—— 它的设备索引通常不是 0**

    python scripts/list_cameras.py              # 扫描 0-5 号设备
    python scripts/list_cameras.py --max 8      # 扫描 0-7 号设备
    python scripts/list_cameras.py --save       # 同时把每个设备的实拍图存到 tests/
    CAMERA_INDEX=1 python webapp/app.py         # 用探测到的索引启动

**路径二：网络视频流（IP Webcam / RTSP 摄像机）—— 无需装虚拟摄像头驱动**

    python scripts/list_cameras.py --url "http://192.168.1.100:8080/video"
    python scripts/list_cameras.py --url "rtsp://user:pass@192.168.1.100:554/h264"
    CAMERA_SOURCE="http://192.168.1.100:8080/video" python webapp/app.py

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


def probe_stream(url: str, timeout: int = 8):
    """探测网络视频流（RTSP/HTTP/RTMP），返回 (是否可用, 分辨率, 帧图像)。

    与本地设备不同：网络流打不开时通常要等超时，所以单独设置 OPEN/READ 超时，
    避免脚本卡死。
    """
    params = [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout * 1000,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout * 1000,
    ]
    try:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)
    except (cv2.error, TypeError):
        cap = cv2.VideoCapture(url)
    try:
        if not cap.isOpened():
            return False, None, None
        frame = None
        for _ in range(10):  # 网络流起播需要缓冲，多等几帧
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


def _probe_url(url: str, timeout: int, save: bool) -> int:
    """探测单个网络流地址并打印结果。"""
    print("=" * 62)
    print("  网络视频流探测")
    print("=" * 62)
    print(f"  地址: {url}")
    print(f"  超时: {timeout}s")
    print()
    ok, size, frame = probe_stream(url, timeout)
    if not ok:
        print("  结果: 连接失败或未取到画面")
        print()
        print("  排查建议：")
        print("    1. 手机与电脑是否在同一局域网（不要一个连 Wi-Fi 一个连热点）；")
        print("    2. IP Webcam 等 App 是否已点「启动服务器」，端口是否与地址一致；")
        print("    3. 电脑防火墙是否拦截了该端口；")
        print("    4. 先用浏览器直接打开该地址，确认能看到画面。")
        print("=" * 62)
        return 1

    print(f"  结果: 可用    分辨率 {size[0]}x{size[1]}")
    if save:
        out = ROOT / "tests" / "camera_probe_stream.jpg"
        try:
            cv2.imwrite(str(out), frame)
            print(f"  首帧已保存: {out}")
        except Exception as e:  # noqa: BLE001
            print(f"  首帧保存失败: {e}")
    print()
    print("  使用方法：")
    print(f'    CAMERA_SOURCE="{url}" python webapp/app.py')
    print("=" * 62)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="枚举本机可用摄像头，或探测网络视频流")
    ap.add_argument("--max", type=int, default=6, help="扫描的索引上限（默认 6，即 0-5）")
    ap.add_argument("--save", action="store_true", help="把实拍首帧保存到 tests/ 便于确认画面")
    ap.add_argument("--backend", choices=["dshow", "msmf", "any"], default="any",
                    help="Windows 下用的后端（默认 any，即 OpenCV 自动选择）")
    ap.add_argument("--url", default="", help="直接探测网络流地址，如 rtsp://... 或 http://.../video")
    ap.add_argument("--timeout", type=int, default=8, help="探测网络流的超时秒数（默认 8）")
    args = ap.parse_args()

    if args.url:
        return _probe_url(args.url, args.timeout, args.save)

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
