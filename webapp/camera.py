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
    "face": False,        # 是否同时对画面做人脸识别
    "faces": [],          # 最近一帧识别到的人：[{name, score, is_temp, box}]
}


def resolve_source():
    """解析取流目标，返回 (传给 VideoCapture 的源, 是否网络流, 展示名)。

    优先使用 CAMERA_SOURCE（RTSP/HTTP/RTMP 网络流），否则回退本地设备索引。
    抽成函数便于单测覆盖，也避免 _worker 里堆判断分支。
    """
    src = (config.CAMERA_SOURCE or "").strip()
    if src:
        return src, True, src
    return config.CAMERA_INDEX, False, f"设备索引 {config.CAMERA_INDEX}"


def _open_capture():
    """按配置打开视频源。网络流需额外设置超时，避免长时间阻塞。"""
    source, is_stream, _ = resolve_source()
    if is_stream:
        # FFMPEG 后端读网络流更稳；超时用微秒，避免无人响应时无限等待
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, config.CAMERA_STREAM_TIMEOUT * 1000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, config.CAMERA_STREAM_TIMEOUT * 1000,
        ]
        try:
            return cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
        except (cv2.error, TypeError):
            # 老版本 OpenCV 不接受 params 参数，退回普通调用
            return cv2.VideoCapture(source)

    cap = cv2.VideoCapture(source)
    # 分辨率只对本地设备生效；网络流由推流端决定，强行设置可能失败
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_FRAME_HEIGHT)
    return cap


def _placeholder(text: str, size: int = 26) -> np.ndarray:
    """生成占位画面。text 支持多行（用 \\n 分隔），整体居中绘制。"""
    img = np.full((480, 720, 3), 28, dtype=np.uint8)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    d.multiline_text(
        (360, 240), text, fill=(180, 190, 205),
        font=fontutil.get_font(size), anchor="mm", align="center", spacing=10,
    )
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


def _recognize_frame(frame, annotated):
    """对一帧做识别，返回 (标注后的图, [{name, score, is_temp, box}])。

    显式返回标注图而不是原地改入参：调用方常把 frame 和 annotated 传成
    两个不同对象（frame 给推理、annotated 给绘制），原地改容易让人以为
    改的是 frame。返回新图可以让调用方明确知道自己拿到的是什么。

    这里刻意不抛异常：人脸识别失败（模型缺失、没有人脸、各种意外）
    都不应该中断抓帧主循环——安全帽/口罩检测才是主线功能，
    人脸识别属于附加能力，坏了就静默降级，画面照常出。
    """
    try:
        import face_db

        if not face_db.face_model_ready():
            return annotated, []
        faces = face_db.extract_faces(frame)
    except Exception:  # noqa: BLE001
        return annotated, []

    out = []
    for fc in faces:
        try:
            pid, name, score, is_temp = face_db.identify(fc["embedding"])
        except Exception:  # noqa: BLE001
            continue
        out.append(
            {
                "name": name,
                "score": score,
                "is_temp": is_temp,
                "person_id": pid,
                "box": [round(float(v), 1) for v in fc["bbox"]],
            }
        )
        try:
            face_db.add_seen(pid, name, is_temp, score)
        except Exception:  # noqa: BLE001
            pass          # 落库失败不影响画面标注

    if out:
        annotated = _annotate_faces(annotated, out)
    return annotated, out


def _annotate_faces(img, faces: list):
    """在图上画人脸框 + 姓名（用 PIL 以支持中文），返回新图。

    用 cv2 画不出中文（会变成一串 ?），所以走 PIL 转换往返。
    绘制失败时返回原图，绝不让画框问题盖掉识别结果。
    """
    try:
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        font = fontutil.get_font(18)
        for f in faces:
            x1, y1, x2, y2 = f["box"]
            # 已注册用青色，未注册用橙色，一眼区分「认识」和「不认识」
            rgb = (255, 190, 40) if f["is_temp"] else (60, 210, 210)
            draw.rectangle([x1, y1, x2, y2], outline=rgb, width=2)
            label = f'{f["name"]}'
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            ty = y1 - th - 8
            if ty < 2:
                ty = y2 + 2
            draw.rectangle([x1, ty, x1 + tw + 10, ty + th + 8], fill=rgb)
            draw.text((x1 + 5, ty + 2), label, fill=(20, 20, 20), font=font)
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:  # noqa: BLE001
        return img


def _worker() -> None:
    cap = None
    last_record = 0.0
    last_face = 0.0          # 上次人脸识别的时间戳（人脸识别按间隔跑，不逐帧）
    fails = 0   # 连续读帧失败次数，用于区分偶发丢帧与真的断流
    is_stream = False
    while not _stop:
        # 打开/重试摄像头
        if cap is None or not cap.isOpened():
            _, is_stream, shown = resolve_source()
            cap = _open_capture()
            if not cap.isOpened():
                if is_stream:
                    hint = (
                        f"无法连接视频流\n{shown}\n"
                        f"请确认手机与电脑在同一局域网、App 已开启服务"
                    )
                else:
                    hint = (
                        f"未检测到摄像头（{shown}）\n"
                        f"请运行 scripts/list_cameras.py 查看可用设备"
                    )
                _publish(_placeholder(hint, size=22), ok=False, counts={})
                cap.release()
                time.sleep(5.0)
                continue
            _publish(_placeholder("视频源已连接，正在启动检测..."), ok=True)
            fails = 0

        ok, frame = cap.read()
        if not ok:
            fails += 1
            # 网络流偶发丢帧很常见，连续失败若干次才判定断流并重连，
            # 避免一次抖动就重建连接造成画面反复闪烁
            if fails < (5 if is_stream else 2):
                time.sleep(0.05)
                continue
            cap.release()
            cap = None
            fails = 0
            continue
        fails = 0

        with _lock:
            kind = _state["kind"] or "helmet"
            face_on = _state.get("face", False)
        try:
            _, counts, annotated, tms = detector.infer_image(frame, [kind])
        except Exception:
            counts, annotated, tms = {}, frame, 0.0
        fps = round(1000.0 / tms, 1) if tms else 0.0

        now = time.time()

        # ---- 人脸识别（按间隔执行，不逐帧跑）----
        # insightface 走 CPU，单帧约 200-400ms；若每帧都跑会把抓帧循环拖垮，
        # 画面直接掉成幻灯片。因此按 config.FACE_INTERVAL 秒节流，
        # 中间帧沿用上一次的识别结果，前端表现为「名字稳定挂着」而不是闪烁。
        if face_on and now - last_face >= config.FACE_INTERVAL:
            last_face = now
            # 返回的是「重新绘制过的图」，必须接住并替换 annotated，
            # 否则人脸姓名框不会出现在推给前端的画面里。
            annotated, face_results = _recognize_frame(frame, annotated)
            with _lock:
                _state["faces"] = face_results
        elif not face_on:
            with _lock:
                if _state.get("faces"):
                    _state["faces"] = []

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


def start(kind: str = "helmet", face: bool | None = None) -> None:
    """确保抓帧线程运行，并切换检测类型（可选开关人脸识别）。"""
    global _thread, _stop
    with _lock:
        _state["kind"] = kind if kind in config.MODELS else "helmet"
        if face is not None:
            _state["face"] = bool(face)
    if _thread is None or not _thread.is_alive():
        _stop = False
        _thread = threading.Thread(target=_worker, daemon=True)
        _thread.start()


def set_kind(kind: str) -> None:
    with _lock:
        _state["kind"] = kind if kind in config.MODELS else "helmet"


def set_face(on: bool) -> None:
    """开关摄像头实时人脸识别。"""
    with _lock:
        _state["face"] = bool(on)
        if not on:
            _state["faces"] = []


def _mask_source(url: str) -> str:
    """脱敏取流地址里的密码。

    RTSP 地址形如 rtsp://admin:密码@192.168.1.64:554/...，
    这个字符串会经 /api/camera/state 回传并显示在页面上。
    若原样返回，摄像头密码就会暴露在浏览器和接口响应里，
    截个图或看一眼网络请求就泄漏了。这里把用户名/密码替换成 ***。
    """
    if not url:
        return url
    try:
        from urllib.parse import urlsplit, urlunsplit

        parts = urlsplit(url)
        if parts.password is None:
            return url          # 本来就没密码，原样返回
        host = parts.hostname or ""
        if parts.port:
            host = f"{host}:{parts.port}"
        # 保留用户名便于辨认用的是哪个账号，只把密码打掉
        user = parts.username or ""
        netloc = f"{user}:***@{host}" if user else f"***@{host}"
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    except Exception:  # noqa: BLE001
        # 解析失败就整串打码，宁可看不清也不泄漏
        return "***"


def get_state() -> dict:
    with _lock:
        s = dict(_state)
        s.pop("jpeg", None)

    # 附带当前取流来源，供前端判断：是「已接入正在出画面」，
    # 还是「没配来源/配错了」需要显示接入引导。
    # 注意这里返回的是配置态（每次读环境变量），不是线程内的运行态，
    # 所以用户改了配置重启后，页面能立刻反映新来源。
    source, is_stream, shown = resolve_source()
    s["source"] = {
        "value": _mask_source(source) if is_stream else "",
        "kind": "stream" if is_stream else "device",
        "label": _mask_source(shown),
    }
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
