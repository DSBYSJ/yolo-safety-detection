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
    "faces": [],          # 最近一帧识别到的人：[{name, score, is_temp, box, equip, missing}]
    "person_compliance": {},  # 按人汇总的佩戴情况（equip.summarize 的输出）
    "active_camera": None,    # 当前选用的摄像头 id（None = 用环境变量那一路）
    "reload": 0,              # 递增即让抓帧线程重连视频源（切换摄像头时用）
}


def resolve_source():
    """解析取流目标，返回 (传给 VideoCapture 的源, 是否网络流, 展示名)。

    优先级：
        1. 若通过 `/api/cameras/switch` 选择了某路已配置的摄像头 → 用它
        2. 否则回退环境变量 CAMERA_SOURCE（RTSP/HTTP/RTMP 网络流）
        3. 再否则回退 CAMERA_INDEX（本机设备索引）

    保留 2/3 两条回退路径是为了兼容旧部署：没有在页面上配置过摄像头时，
    行为与以前完全一致，不会因为新增功能而改变既有使用方式。
    抽成函数便于单测覆盖，也避免 _worker 里堆判断分支。
    """
    with _lock:
        active = _state.get("active_camera")
    if active:
        try:
            import camera_db
            res = camera_db.resolve_url(int(active))
            if res:
                ctype, val = res
                row = camera_db.get(int(active)) or {}
                label = row.get("name") or f"摄像头 #{active}"
                if ctype == "device":
                    return val, False, f"{label}（设备索引 {val}）"
                return val, True, f"{label}（{camera_db.mask_url(str(val))}）"
        except Exception:  # noqa: BLE001
            pass          # 配置读取失败就退回环境变量，不让画面因为配置问题彻底黑掉

    src = (config.CAMERA_SOURCE or "").strip()
    if src:
        return src, True, src
    return config.CAMERA_INDEX, False, f"设备索引 {config.CAMERA_INDEX}"


def set_active_camera(cid) -> bool:
    """切换当前使用的摄像头（None 表示回到环境变量配置的那一路）。

    切换后会**主动让抓帧线程重连**：线程已经把旧 VideoCapture 打开了，
    只改配置不会生效，必须让它 detect 到 reload 计数变化并重建连接。
    """
    if cid is not None:
        try:
            import camera_db
            if not camera_db.get(int(cid)):
                return False
        except Exception:  # noqa: BLE001
            return False
    with _lock:
        _state["active_camera"] = cid
        _state["reload"] = (_state.get("reload") or 0) + 1
    return True


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


def _recognize_frame(frame, annotated, detections=None):
    """对一帧做识别，返回 (标注后的图, [{name, score, is_temp, box, equip, missing}])。

    显式返回标注图而不是原地改入参：调用方常把 frame 和 annotated 传成
    两个不同对象（frame 给推理、annotated 给绘制），原地改容易让人以为
    改的是 frame。返回新图可以让调用方明确知道自己拿到的是什么。

    detections 是同一帧的装备检测框。传入后会把「安全帽/口罩」按空间重叠
    归属到具体的人头上，于是「张三没戴安全帽」才成为一句可以回答的话 ——
    否则人脸和装备只是两条互不相干的线。

    这里刻意不抛异常：人脸识别失败（模型缺失、没有人脸、各种意外）
    都不应该中断抓帧主循环——安全帽/口罩检测才是主线功能，
    人脸识别属于附加能力，坏了就静默降级，画面照常出。
    """
    try:
        import face_db
        import equip

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

    # 装备归属：把人脸框与装备框按重叠关系配对
    if out and detections:
        try:
            out = equip.attach(detections, out)
        except Exception:  # noqa: BLE001
            pass          # 归属失败就退化成「只识别人」，不影响画框

    for f in out:
        try:
            equip_state = f.get("equip") or {}
            face_db.add_seen(
                f.get("person_id"),
                f["name"],
                f.get("is_temp"),
                f.get("score", 0.0),
                hat_state=_tri_state(equip_state, "hat"),
                mask_state=_tri_state(equip_state, "mask"),
                source="camera",
            )
        except Exception:  # noqa: BLE001
            pass          # 落库失败不影响画面标注

    if out:
        annotated = _annotate_faces(annotated, out)
    return annotated, out


def _tri_state(equip_state: dict, key: str):
    """把归属结果压成三态：True->1（戴了）/ False->0（没戴）/ None（无信息）。

    这个转换是合规统计准确性的关键：没有检测到该类装备框时必须是 None，
    不能当成 0（未佩戴），否则漏检会被算成员工违规。
    """
    v = equip_state.get(key)
    if v is None:
        return None
    return bool(v.get("compliant"))


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
    seen_reload = 0          # 已处理的 reload 计数
    while not _stop:
        # 切换摄像头时前端会递增 reload；这里比对后主动释放旧连接，
        # 让下面走一次「重新打开」流程。不加这个的话，页面选了新摄像头
        # 但画面还是旧的 —— 因为线程手里的 VideoCapture 从没变过。
        with _lock:
            cur_reload = _state.get("reload") or 0
        if cur_reload != seen_reload:
            seen_reload = cur_reload
            if cap is not None:
                cap.release()
                cap = None

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
            dets, counts, annotated, tms = detector.infer_image(frame, [kind])
        except Exception:
            dets, counts, annotated, tms = [], {}, frame, 0.0
        fps = round(1000.0 / tms, 1) if tms else 0.0

        now = time.time()

        # ---- 人脸识别（按间隔执行，不逐帧跑）----
        # insightface 走 CPU，单帧约 200-400ms；若每帧都跑会把抓帧循环拖垮，
        # 画面直接掉成幻灯片。因此按 config.FACE_INTERVAL 秒节流，
        # 中间帧沿用上一次的识别结果，前端表现为「名字稳定挂着」而不是闪烁。
        if face_on and now - last_face >= config.FACE_INTERVAL:
            last_face = now
            # 把本帧的装备框一起传进去，让人脸与装备按空间重叠配对，
            # 这样落库的每条记录都带「这个人当时戴没戴」
            annotated, face_results = _recognize_frame(frame, annotated, dets)
            try:
                import equip as _equip
                person_sum = _equip.summarize(face_results)
            except Exception:  # noqa: BLE001
                person_sum = {}
            with _lock:
                _state["faces"] = face_results
                _state["person_compliance"] = person_sum
        elif not face_on:
            with _lock:
                if _state.get("faces"):
                    _state["faces"] = []
                    _state["person_compliance"] = {}

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
