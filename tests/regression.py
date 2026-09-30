# -*- coding: utf-8 -*-
"""端到端回归测试：验证 Web 系统各模块是否正常

用法：
    # 1. 先启动服务（另开一个终端）
    python webapp/app.py
    # 2. 再跑本脚本
    python tests/regression.py

可选参数：
    python tests/regression.py --base http://127.0.0.1:5000 --conf 0.25
"""
import argparse
import json
import mimetypes
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = "http://127.0.0.1:5000"

OK, FAIL, SKIP = "[PASS]", "[FAIL]", "[SKIP]"
_results = []


def check(name, cond, detail=""):
    _results.append((bool(cond), name, detail))
    print(f"{OK if cond else FAIL} {name}  {detail}")


def skip(name, reason):
    _results.append((True, name, f"(跳过) {reason}"))
    print(f"{SKIP} {name}  {reason}")


# --------------------------------------------------------------------------- #
# HTTP 工具（纯标准库，避免额外依赖）
# --------------------------------------------------------------------------- #
def post_file(base, path, kinds, conf=0.25, field="kinds", endpoint="/api/detect/image",
              timeout=300):
    """按 multipart/form-data 上传文件。kinds 以 field 指定的字段名重复提交，
    与前端 <input name='kinds'> 的行为一致。"""
    boundary = "----wbtest" + uuid.uuid4().hex
    p = Path(path)
    ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    body = b""
    for k in kinds:
        body += (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"\r\n\r\n{k}\r\n'
        ).encode()
    body += (
        f'--{boundary}\r\nContent-Disposition: form-data; name="conf"\r\n\r\n{conf}\r\n'
    ).encode()
    body += (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        f'filename="{p.name}"\r\nContent-Type: {ctype}\r\n\r\n'
    ).encode() + p.read_bytes() + b"\r\n"
    body += f"--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        base + endpoint, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def get_json(base, path, timeout=60):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.load(r)


def http_code(base, path, timeout=20):
    try:
        with urllib.request.urlopen(base + path, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# 测试项
# --------------------------------------------------------------------------- #
def test_pages(base):
    print("\n【1】页面可达性（7 个）")
    for p in ["/", "/detect", "/camera", "/records", "/stats", "/train", "/eval"]:
        code = http_code(base, p)
        check(f"页面 {p}", code == 200, f"HTTP {code}")


def test_env(base):
    print("\n【2】运行环境")
    env = get_json(base, "/api/env")
    check("/api/env", bool(env.get("device")), f"device={env.get('device')} torch={env.get('torch')}")


def test_detect_image(base, imgs, conf):
    print("\n【3】图片检测")

    # 单类检测
    for label, kind in [("安全帽", "helmet"), ("口罩", "mask")]:
        img = imgs.get(kind)
        if not img:
            skip(f"检测({label})", f"缺少测试图 {img}")
            continue
        try:
            d = post_file(base, img, [kind], conf=conf)
            counts = d.get("counts", {})
            dets = d.get("detections", [])
            # 断言：返回的目标全部属于所请求的模型
            wrong = {x["kind"] for x in dets} - {kind}
            check(
                f"检测({label})",
                d.get("ok") and len(dets) > 0 and not wrong,
                f"counts={counts} 检出{len(dets)}个 耗时{d.get('time_ms')}ms 记录#{d.get('record_id')}",
            )
            if wrong:
                check(f"检测({label}) 模型归属", False, f"混入了其他模型的结果: {wrong}")
        except Exception as e:
            check(f"检测({label})", False, f"异常 {e}")

    # 双模型同时检测
    grp = imgs.get("group")
    if grp:
        try:
            d = post_file(base, grp, ["helmet", "mask"], conf=conf)
            dets = d.get("detections", [])
            kinds_seen = {x["kind"] for x in dets}
            check(
                "检测(安全帽+口罩同时)",
                d.get("ok") and len(dets) > 0,
                f"counts={d.get('counts')} 涉及模型={kinds_seen or '无'}",
            )
        except Exception as e:
            check("检测(安全帽+口罩同时)", False, f"异常 {e}")
    else:
        skip("检测(安全帽+口罩同时)", "缺少多人拼图")

    # 结果图必须真实落盘且可访问（防「中文路径下 cv2.imwrite 静默失败」回归）
    img = imgs.get("helmet")
    if img:
        try:
            d = post_file(base, img, ["helmet"], conf=conf)
            url = d.get("image_url", "")
            full = f"{base}/{url}" if url else ""
            code, blob = 0, b""
            if full:
                with urllib.request.urlopen(full, timeout=30) as r:
                    code, blob = r.status, r.read()
            check(
                "结果图落盘并可访问",
                code == 200 and len(blob) > 1024,
                f"{url} HTTP {code} {len(blob)} 字节",
            )
        except Exception as e:
            check("结果图落盘并可访问", False, f"异常 {e}（可能触发了中文路径写图失败）")


def test_error_handling(base, imgs):
    print("\n【4】异常处理")
    # 非法扩展名
    bad = ROOT / "tests" / "_bad_format.txt"
    bad.write_text("not an image")
    try:
        post_file(base, bad, ["helmet"])
        check("非法文件格式被拒", False, "应返回 400，实际未报错")
    except urllib.error.HTTPError as e:
        check("非法文件格式被拒", e.code == 400, f"HTTP {e.code}")
    except Exception as e:
        check("非法文件格式被拒", False, f"异常 {e}")

    # 非法 kinds 应回退到默认，不应 500
    img = imgs.get("helmet")
    if img:
        try:
            d = post_file(base, img, ["nonexistent_kind"], conf=0.25)
            check("非法 kinds 兜底不崩溃", d.get("ok") is not None, f"ok={d.get('ok')}")
        except Exception as e:
            check("非法 kinds 兜底不崩溃", False, f"异常 {e}")


def test_records(base):
    print("\n【5】检测记录")
    d = get_json(base, "/api/records?page=1&page_size=5")
    recs = d.get("records", [])
    check("/api/records 分页查询", d.get("ok") and d.get("total", 0) >= 0,
          f"total={d.get('total')} 本页{len(recs)}条")
    if recs:
        r = recs[0]
        detail = get_json(base, f"/api/records/{r['id']}")
        check(f"/api/records/{r['id']} 详情", bool(detail), f"字段={list(detail.keys())[:6]}")


def test_stats(base):
    print("\n【6】统计报表")
    s = get_json(base, "/api/stats")
    need = {"summary", "daily", "classes", "sources", "conf"}
    missing = need - set(s.keys())
    check("/api/stats 统计", s.get("ok") and not missing,
          f"字段={sorted(s.keys())}" + (f" 缺少={missing}" if missing else ""))


def test_eval_metrics(base):
    print("\n【7】模型评估指标")
    for k in ("helmet", "mask"):
        try:
            m = get_json(base, f"/api/eval/metrics/{k}")
            metrics = m.get("metrics") or {}
            map50 = metrics.get("mAP50")
            check(
                f"/api/eval/metrics/{k}",
                map50 is not None,
                f"mAP@0.5={map50} 图表{len(m.get('images', []))}张",
            )
            if map50 is not None:
                check(f"{k} 指标达标(≥80%)", map50 >= 0.80, f"mAP@0.5={map50:.4f}")
        except Exception as e:
            check(f"/api/eval/metrics/{k}", False, f"异常 {e}")


def main():
    ap = argparse.ArgumentParser(description="Web 系统端到端回归测试")
    ap.add_argument("--base", default=DEFAULT_BASE, help="服务地址")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    # 连通性
    if http_code(base, "/") != 200:
        print(f"[FAIL] 无法连接 {base}，请先运行：python webapp/app.py")
        return 2
    print(f"目标服务: {base}")

    # 测试素材
    t = ROOT / "tests"
    imgs = {
        "helmet": next(iter(sorted(t.glob("helmet*.jpg"))), None) or next(iter(sorted(t.glob("*.jpg"))), None),
        "mask": next(iter(sorted(t.glob("mask_*.jpg"))), None) or next(iter(sorted(t.glob("*.jpg"))), None),
        "group": t / "mask_group.jpg" if (t / "mask_group.jpg").exists() else None,
    }

    test_pages(base)
    test_env(base)
    test_detect_image(base, imgs, args.conf)
    test_error_handling(base, imgs)
    test_records(base)
    test_stats(base)
    test_eval_metrics(base)

    print("\n" + "=" * 68)
    passed = sum(1 for c, _, _ in _results if c)
    failed = [(n, d) for c, n, d in _results if not c]
    print(f"回归结果: {passed}/{len(_results)} 通过")
    for n, d in failed:
        print(f"  {FAIL} {n} -> {d}")
    print("=" * 68)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
