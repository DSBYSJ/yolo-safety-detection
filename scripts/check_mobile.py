# -*- coding: utf-8 -*-
"""移动端适配验证：用无头 Chrome 以手机视口逐页截图，并检查横向溢出。

用法：
    python scripts/check_mobile.py                # 截图 + 溢出检查
    python scripts/check_mobile.py --only /phone  # 只查某个页面

为什么需要它：仅靠肉眼看代码无法确认窄屏下是否出现横向滚动条、
元素重叠或文字截断，而这些问题在手机上会直接毁掉可用性。
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
BASE = "http://127.0.0.1:5000"

PAGES = [
    ("/", "首页概览"),
    ("/detect", "在线检测"),
    ("/camera", "实时监控"),
    ("/phone", "手机检测"),
    ("/faces", "人脸底库"),
    ("/compliance", "合规统计"),
    ("/attendance", "人脸打卡"),
    ("/records", "检测记录"),
    ("/stats", "统计报表"),
    ("/train", "模型训练"),
    ("/eval", "模型评估"),
]

# 常见手机视口：iPhone 14 Pro / 小屏安卓
VIEWPORTS = [(390, 844)]


def shot(path: str, w: int, h: int, out: Path) -> bool:
    """对指定页面截图，返回是否成功。"""
    cmd = [
        CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
        "--hide-scrollbars", "--virtual-time-budget=6000",
        f"--window-size={w},{h}",
        f"--screenshot={out}",          # 必须给 Windows 路径
        f"{BASE}{path}",
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=90)
    return out.exists() and out.stat().st_size > 0


def overflow_check(path: str, w: int, h: int, timeout: int = 30, shot_path=None) -> dict:
    """用 CDP 测量页面是否横向溢出。

    起一个临时 Chrome（带远程调试端口），通过 WebSocket 连上后
    Runtime.evaluate 读取 scrollWidth 与 innerWidth。
    比截图更客观：scrollWidth > innerWidth 就说明有横向滚动条。

    返回 {"scrollWidth": int, "innerWidth": int, "overflow": bool, "wide": [...]}
    wide 是超出视口右边界的元素（最多 5 个），便于定位问题元素。
    """
    import json as _json
    import os as _os
    import socket
    import struct
    import time as _time
    import urllib.request
    import base64 as _b64

    port = 9333
    profile = tempfile.mkdtemp(prefix="cdp_prof_")
    proc = subprocess.Popen(
        [
            CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile}",   # 正斜杠 Windows 路径
            "--remote-allow-origins=*",     # Chrome 116+ 必须带，否则拒连
            f"--window-size={w},{h}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        # 等调试端口就绪
        ws_url = None
        for _ in range(40):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=2) as r:
                    tabs = _json.loads(r.read())
                cand = [t for t in tabs if t.get("type") == "page"]
                if cand:
                    ws_url = cand[0]["webSocketDebuggerUrl"]
                    break
            except Exception:
                pass
            _time.sleep(0.4)
        if not ws_url:
            return {"error": "无法连接 Chrome 调试端口"}

        # 极简 WebSocket 客户端（避免引入额外依赖）
        from urllib.parse import urlparse
        u = urlparse(ws_url)
        sock = socket.create_connection((u.hostname, u.port), timeout=10)
        key = _b64.b64encode(_os.urandom(16)).decode()
        sock.sendall((
            f"GET {u.path} HTTP/1.1\r\nHost: {u.hostname}:{u.port}\r\n"
            f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        ).encode())
        # 跳过握手响应
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += sock.recv(4096)
        buf = buf.split(b"\r\n\r\n", 1)[1]

        def send(obj):
            data = _json.dumps(obj).encode()
            hdr = bytearray([0x81])
            n = len(data)
            if n < 126:
                hdr.append(0x80 | n)
            elif n < 65536:
                hdr.append(0x80 | 126); hdr += struct.pack(">H", n)
            else:
                hdr.append(0x80 | 127); hdr += struct.pack(">Q", n)
            mask = _os.urandom(4)
            hdr += mask
            sock.sendall(bytes(hdr) + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

        def recv():
            nonlocal buf
            while True:
                if len(buf) >= 2:
                    b1, b2 = buf[0], buf[1]
                    ln = b2 & 0x7F
                    off = 2
                    if ln == 126:
                        if len(buf) < 4: buf += sock.recv(4096); continue
                        ln = struct.unpack(">H", buf[2:4])[0]; off = 4
                    elif ln == 127:
                        if len(buf) < 10: buf += sock.recv(4096); continue
                        ln = struct.unpack(">Q", buf[2:10])[0]; off = 10
                    if len(buf) < off + ln:
                        buf += sock.recv(max(4096, off + ln - len(buf))); continue
                    payload = buf[off:off + ln]
                    buf = buf[off + ln:]
                    return _json.loads(payload)
                buf += sock.recv(4096)

        mid = [0]
        def cmd(method, params=None):
            mid[0] += 1
            send({"id": mid[0], "method": method, "params": params or {}})
            while True:
                m = recv()
                if m.get("id") == mid[0]:
                    return m

        cmd("Page.enable")
        # 关键：用 Emulation 精确设定设备视口与 DPR。
        # 只靠 --window-size 不够 —— headless 下窗口尺寸含边框，
        # 且页面有 width=device-width 的 viewport meta，实测会得到 500 而非 390。
        cmd("Emulation.setDeviceMetricsOverride", {
            "width": w, "height": h,
            "deviceScaleFactor": 2,
            "mobile": True,
        })
        cmd("Emulation.setTouchEmulationEnabled", {"enabled": True, "maxTouchPoints": 5})
        cmd("Page.navigate", {"url": f"{BASE}{path}"})
        _time.sleep(3.5)   # 等页面与请求渲染完

        expr = """(() => {
          const d = document.documentElement;
          const vw = window.innerWidth;
          const wide = [];
          document.querySelectorAll('body *').forEach(el => {
            const r = el.getBoundingClientRect();
            if (r.width > 0 && r.right > vw + 1) {
              const tag = el.tagName.toLowerCase();
              const cls = (el.className && typeof el.className === 'string')
                ? '.' + el.className.trim().split(/\\s+/).slice(0,2).join('.') : '';
              wide.push(tag + cls + ' (right=' + Math.round(r.right) + ')');
            }
          });
          return JSON.stringify({
            scrollWidth: d.scrollWidth,
            innerWidth: vw,
            wide: [...new Set(wide)].slice(0, 5)
          });
        })()"""
        res = cmd("Runtime.evaluate", {"expression": expr, "returnByValue": True})
        val = _json.loads(res["result"]["result"]["value"])
        val["overflow"] = val["scrollWidth"] > val["innerWidth"] + 1

        # 顺便用同一个会话截图，保证与测量用的是完全相同的视口
        if shot_path is not None:
            try:
                cap = cmd("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True})
                data = cap["result"]["data"]
                Path(shot_path).write_bytes(_b64.b64decode(data))
                val["shot"] = str(shot_path)
            except Exception as e:  # noqa: BLE001
                val["shot_error"] = str(e)
        return val
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


def main() -> int:
    ap = argparse.ArgumentParser(description="移动端适配验证")
    ap.add_argument("--only", default="", help="只检查指定路径，如 /phone")
    ap.add_argument("--outdir", default="", help="截图输出目录")
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else Path(tempfile.mkdtemp(prefix="mobile_shots_"))
    outdir.mkdir(parents=True, exist_ok=True)

    pages = [p for p in PAGES if not args.only or p[0] == args.only]
    if not pages:
        print(f"未找到页面: {args.only}")
        return 1

    print("=" * 64)
    print("  移动端适配验证")
    print("=" * 64)
    print(f"  服务: {BASE}")
    print(f"  输出: {outdir}")
    print()

    ok_count = 0
    bad = []
    for path, name in pages:
        out = outdir / f"{path.strip('/').replace('/', '_') or 'index'}_{VIEWPORTS[0][0]}x{VIEWPORTS[0][1]}.png"
        check = overflow_check(path, *VIEWPORTS[0], shot_path=out)
        size_kb = out.stat().st_size // 1024 if out.exists() else 0

        if "error" in check:
            flag, note = "?? ", check["error"]
        elif check["overflow"]:
            flag = "!! "
            note = f'scrollWidth={check["scrollWidth"]} > 视口 {check["innerWidth"]}'
            bad.append((name, path, check))
        else:
            flag, note = "OK ", f'宽度 {check["scrollWidth"]} = 视口 {check["innerWidth"]}'

        print(f"  [{flag}] {name:<8} {path:<10} 截图 {size_kb}KB  {note}")
        if not check.get("overflow") and "error" not in check:
            ok_count += 1

    print()
    print("-" * 64)
    if bad:
        print(f"  ⚠ 发现 {len(bad)} 个页面存在横向溢出：")
        for name, path, c in bad:
            print(f"    · {name} ({path})")
            for e in c.get("wide", []):
                print(f"        超出: {e}")
        print()
    print(f"  通过：{ok_count}/{len(pages)}")
    print(f"  截图目录：{outdir}")
    print("=" * 64)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
