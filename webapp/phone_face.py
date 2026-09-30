# -*- coding: utf-8 -*-
"""手机端人脸识别：模式判定、节流与结果整理

为什么单独一个模块
------------------
摄像头端的人脸识别跑在 `camera.py` 的后台线程里，状态（上次识别时间、
人脸开关）由线程自己维护。手机端走的是「每帧一个 HTTP 请求」，
没有常驻线程，也就没有地方存状态 —— 只能由服务端进程级地记一下
「上一次给这个客户端跑人脸识别是什么时候」。

把这块逻辑抽出来，一是避免 `app.py` 里堆一堆全局计时变量，
二是可以脱离 Flask 直接做单元测试（节流这种东西最容易写错，
又最不容易在手工点击中发现）。

三种模式（前端可选）
--------------------
- ``off``      ：不做人脸识别，只跑安全帽/口罩。
- ``interval`` ：服务端按 ``FACE_INTERVAL`` 秒节流。人脸推理走 CPU 约
                 200~400ms，逐帧跑会把手机端画面拖到 1~2fps；
                 按 2 秒一次既能看见姓名，又不影响画面流畅度。
- ``every``    ：每帧都跑。实时性最好，但画面会明显变卡，
                 适合短暂定点查看，不适合长时间开。

另外：``save`` 为真（用户点「抓拍存档」）时**无条件**跑一次人脸识别，
不受模式与节流影响 —— 存档图只有一张，这一次必须认准。
"""
import threading
import time

import numpy as np

import config
import face_db

# 客户端标识 -> 上次跑人脸识别的时间戳
# 进程级字典即可：手机端通常是单页面单会话，无需过期回收，
# 但要加锁，因为 Flask 可能并发处理同一客户端的请求。
_last_face_at: dict = {}
_lock = threading.Lock()

# 节流状态保留时长：超过这个时间没来请求就把条目丢掉，防止字典无限增长
_STALE_SEC = 3600.0

VALID_MODES = ("off", "interval", "every")


def normalize_mode(raw) -> str:
    """把前端传来的任意值收敛成合法模式，非法一律当 off。

    刻意不抛异常：手机端是逐帧高频调用，为一个小参数传错就返回 400，
    会让整个画面链路断掉 —— 用户看到的是「检测坏了」，
    实际上只是模式字段没写对。降级成 off 最不伤体验。
    """
    if not isinstance(raw, str):
        if raw is True:
            return "interval"      # 兼容老的布尔写法
        return "off"
    m = raw.strip().lower()
    return m if m in VALID_MODES else ("interval" if m in ("1", "true", "yes", "on") else "off")


def should_run(client_key: str, mode: str, save: bool = False, now: float = None) -> bool:
    """判断本次请求要不要跑人脸识别。

    只判断、不更新状态 —— 更新由 `mark_ran` 完成。
    分开是为了让「先判断该不该跑，跑失败了就不该记时间」这件事
    在调用方看得明白：识别人脸可能失败（无脸/模型缺失），
    失败也记时间的话，之后两秒内都不会再试，体验上像是卡住了。
    """
    if save:
        return True                      # 存档必须认人
    if mode == "every":
        return True
    if mode != "interval":
        return False
    t = time.time() if now is None else now
    with _lock:
        last = _last_face_at.get(client_key)
    if last is None:
        return True
    return (t - last) >= float(config.FACE_INTERVAL)


def mark_ran(client_key: str, now: float = None) -> None:
    """记录一次实际执行的人脸识别时间。"""
    t = time.time() if now is None else now
    with _lock:
        _last_face_at[client_key] = t
        if len(_last_face_at) > 256:     # 简单的容量保护
            cutoff = t - _STALE_SEC
            for k in [k for k, v in _last_face_at.items() if v < cutoff]:
                _last_face_at.pop(k, None)


def next_interval(client_key: str, mode: str, now: float = None) -> float:
    """告诉前端「最快多少秒后再来人脸请求」。

    interval 模式下前端可以据此主动降频，减少无意义的空转请求；
    其余模式返回 0，表示不用管。
    """
    if mode != "interval":
        return 0.0
    t = time.time() if now is None else now
    with _lock:
        last = _last_face_at.get(client_key)
    if last is None:
        return 0.0
    return max(0.0, round(float(config.FACE_INTERVAL) - (t - last), 2))


def reset(client_key: str = None) -> None:
    """清空节流状态（测试用；client_key 为空时全清）。"""
    with _lock:
        if client_key is None:
            _last_face_at.clear()
        else:
            _last_face_at.pop(client_key, None)


def recognize(img_bgr, save: bool = False, detections=None, source: str = "phone") -> list:
    """对一帧做识别，返回 [{name, score, is_temp, person_id, box, equip, missing}]。

    不抛异常：人脸识别是附加能力，坏了要静默降级，不能连累
    安全帽/口罩这条主线（这是手机端的主要用途）。

    ``detections`` 是同一帧的装备检测框。传入后会把安全帽/口罩按空间重叠
    归属到具体的人头上，落库时就带上了「这个人当时戴没戴」——
    否则合规统计只能按「次」聚合，答不出「谁没戴」。

    ``save=True`` 时才把结果写进 face_seen —— 逐帧写库会在几秒内
    产生上千条记录，把「合规统计」页冲垮。
    """
    try:
        if not face_db.face_model_ready():
            return []
        faces = face_db.extract_faces(img_bgr)
    except Exception:  # noqa: BLE001
        return []

    out = []
    for fc in faces:
        try:
            pid, name, score, is_temp = face_db.identify(fc["embedding"])
        except Exception:  # noqa: BLE001
            continue
        item = {
            "name": name,
            "score": score,
            "is_temp": bool(is_temp),
            "person_id": pid,
            "box": [round(float(v), 1) for v in fc["bbox"]],
        }
        out.append(item)

    # 装备归属：只保留「属于某个人」的装备，答出谁没戴
    if out and detections:
        try:
            import equip
            out = equip.attach(detections, out)
        except Exception:  # noqa: BLE001
            pass

    if save:
        for f in out:
            try:
                eq = f.get("equip") or {}
                face_db.add_seen(
                    f.get("person_id"),
                    f["name"],
                    f.get("is_temp"),
                    f.get("score", 0.0),
                    hat_state=_tri_state(eq, "hat"),
                    mask_state=_tri_state(eq, "mask"),
                    source=source,
                )
            except Exception:  # noqa: BLE001
                pass
    return out


def _tri_state(equip_state: dict, key: str):
    """三态转换：True->1（戴了）/ False->0（没戴）/ None（无信息）。

    无信息必须是 None 而不是 0 —— 否则摄像头没对准、模型漏检
    都会被统计成「未佩戴」，把员工大面积误判成违规。
    """
    v = equip_state.get(key)
    if v is None:
        return None
    return bool(v.get("compliant"))


def scale_boxes(faces: list, src_size, dst_size) -> list:
    """把框从「识别用的帧」坐标换算到「前端显示用的帧」坐标。

    手机端有一个容易漏的坑：识别时服务端为了控制上传体积，
    可能会把收到的图再缩放一次；而前端画框用的是它自己上传的那张图的
    尺寸（``dets`` 就是按这个尺寸给的）。两边尺寸不一致时人脸框会整体偏移。
    这里统一按比例缩放，让调用方不必关心内部用了多大分辨率。
    """
    if not faces:
        return faces
    try:
        sw, sh = float(src_size[0]), float(src_size[1])
        dw, dh = float(dst_size[0]), float(dst_size[1])
        if sw <= 0 or sh <= 0 or dw <= 0 or dh <= 0:
            return faces
        if abs(sw - dw) < 0.5 and abs(sh - dh) < 0.5:
            return faces
        kx, ky = dw / sw, dh / sh
        for f in faces:
            box = f.get("box")
            if not box or len(box) != 4:
                continue
            f["box"] = [round(box[0] * kx, 1), round(box[1] * ky, 1),
                        round(box[2] * kx, 1), round(box[3] * ky, 1)]
    except Exception:  # noqa: BLE001
        return faces
    return faces
