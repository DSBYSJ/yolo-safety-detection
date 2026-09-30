# -*- coding: utf-8 -*-
"""面板打卡（人脸打卡页专属）—— 人脸识别打卡，与 attendance.py 完全独立

它做什么
--------
在「人脸打卡」页提供一条**独立的人脸识别打卡**链路：
对着一帧画面做 1:N 人脸比对，**只有命中底库已注册的人**才算打卡成功，
把姓名与打卡时刻写进本模块自己的表。

与 attendance.py 的分工（必须分清，否则报表会互相污染）
--------------------------------------------------------
``attendance.py`` 的数据源是 ``face_seen`` 流水，它统计的是
「摄像头**持续**看到的出勤情况」，是**被动、连续**的观测。

本模块的数据源是自己的 ``panel_checkin_events`` 表，记的是
「在打卡面板上**主动刷脸**留下的那一次确认」，是**主动、一次性**的动作。

两者不可互相替代，典型差异：
    有人从摄像头前走过   → attendance 记为「打卡」；面板若没刷脸，仍是「未打卡」
    在面板前刷脸但没去镜头前 → 面板记为「已打卡」；attendance 仍是无记录

**绝不能把刷脸结果写进 `face_seen`**：那张表是「事实」，
插一条就等于宣布「这个人来过」，会立即污染人脸打卡 / 合规统计 /
趋势图等所有相关报表，而页面上完全看不出多出来的数据是哪来的。
因此本模块：

1. 只写自己的表 ``panel_checkin_events``（表名带 ``panel_`` 前缀，便于一眼认出）；
2. 只被 ``/attendance`` 一个页面调用，路由统一挂在
   ``/api/attendance/panel/*`` 下，与既有 ``/api/attendance/*`` 不冲突；
3. 不调用 attendance / compliance 的任何函数，也不写 records。

人脸比对复用 ``face_db``，但**只借它的识别能力，绝不碰它的记录**：
调的是 ``extract_faces`` / ``identify`` 这两个只读函数，
从头到尾不调 ``add_seen``。

只有已注册的人能打卡
--------------------
未注册的人会被 ``identify`` 分配成「访客-N」临时编号。临时编号是
按特征指纹临时发的，进程重启后重新发号，拿它打卡会产出大量幽灵记录
（同一个人今天叫访客-1、明天叫访客-3）。因此**访客一律拒绝打卡**，
并明确提示「未识别到已注册人员，请先到人脸底库录入」。

三态：识别不出人 ≠ 打卡失败
---------------------------
``done``     该人当日已打卡（附首次打卡时间）
``pending``  已注册但当日尚未打卡
页面级还有第三态 ``recognized=false``（本帧没认到已注册的人），
它与「这个人没打卡」是两回事，必须分开提示 ——
前者要用户「把脸对准镜头」，后者要用户「点一下打卡」。
"""

import sqlite3
import threading
import time

import config

# 状态常量。沿用 attendance 的英文短串风格，前端不必再译一次。
ST_DONE = "done"            # 已打卡
ST_PENDING = "pending"      # 未打卡（= 今天还没刷过脸）

STATUS_LABEL = {
    ST_DONE: "已打卡",
    ST_PENDING: "未打卡",
}

# 单条备注长度上限。备注会直接渲染进页面，过长会撑坏表格；
# 截断而不是拒绝，避免用户因为多打几个字而丢一次打卡。
NOTE_MAX = 60


class PanelCheckinError(RuntimeError):
    """本模块的业务错误（姓名非法、打卡失败等），由调用方转 400/500。"""


SCHEMA = """
CREATE TABLE IF NOT EXISTS panel_checkin_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER,                    -- 命中的底库人员 id
    person_name TEXT NOT NULL,              -- 打卡人姓名（来自底库）
    note        TEXT,                       -- 备注（可选）
    score       REAL DEFAULT 0,             -- 本次刷脸的相似度
    source      TEXT DEFAULT 'face',        -- 打卡方式标记
    checkin_at  TEXT NOT NULL,              -- 打卡时刻 'YYYY-MM-DD HH:MM:SS'
    day         TEXT NOT NULL               -- 冗余出日期，便于按日索引查询
);
CREATE INDEX IF NOT EXISTS idx_panel_checkin_day
    ON panel_checkin_events(day);
CREATE INDEX IF NOT EXISTS idx_panel_checkin_name_day
    ON panel_checkin_events(person_name, day);
"""

# ------------------------------------------------------------------ 防重复刷
# 同一张脸在镜头前会连续命中，逐帧都打卡会在几秒内堆出上百条记录。
# 因此按「人 + 当日」做冷却：冷却期内的重复刷脸一律忽略，不写库。
# 取 8 秒是个折中 —— 足够盖住「人还站在镜头前」的这段时间，
# 又不至于让一个人出去转一圈回来再刷时被拒。
COOLDOWN_SEC = 8.0

_cool: dict = {}                 # (person_name, day) -> 上次打卡时间戳
_cool_lock = threading.Lock()

_local = threading.local()


# ---------------------------------------------------------------- 数据库

def get_conn() -> sqlite3.Connection:
    """线程局部连接，与项目其它模块（db.py / face_db.py）保持一致的做法。

    共用一个 data.db 文件，但**各管各的表**：
    本模块从不查询 records / face_seen / face_persons，
    因此即便 attendance 那边的数据被清空，本模块的状态也不受影响。
    """
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(str(config.DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        _local.conn = conn
    return conn


def init_db() -> None:
    conn = get_conn()
    conn.executescript(SCHEMA)
    _upgrade_columns(conn)
    conn.commit()


# 后加的列。老库升级用 —— SQLite 没有 ADD COLUMN IF NOT EXISTS，
# 重复添加会抛 duplicate column name，因此先查现有列名再决定加不加。
_COL_UPGRADE = """
ALTER TABLE panel_checkin_events ADD COLUMN person_id INTEGER;
ALTER TABLE panel_checkin_events ADD COLUMN score     REAL DEFAULT 0;
ALTER TABLE panel_checkin_events ADD COLUMN source    TEXT DEFAULT 'face';
"""


def _upgrade_columns(conn) -> None:
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(panel_checkin_events)")}
    for stmt in _COL_UPGRADE.strip().split(";"):
        stmt = stmt.strip()
        if not stmt:
            continue
        parts = stmt.split()
        # ALTER TABLE t ADD COLUMN <name> <type>  → 列名在索引 5
        if len(parts) >= 6 and parts[5] in cols:
            continue
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            pass          # 并发初始化时可能已被别的连接加过


def close() -> None:
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        finally:
            _local.conn = None


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- 校验

def validate_day(day: str) -> str:
    """校验 YYYY-MM-DD。非法格式抛 PanelCheckinError，由调用方转 400。"""
    s = (day or "").strip()
    try:
        time.strptime(s, "%Y-%m-%d")
    except ValueError:
        raise PanelCheckinError("日期格式应为 YYYY-MM-DD")
    return s


def _clean_name(name: str) -> str:
    """姓名来自人脸识别结果，这里只做非空与限长校验。

    刻意**不与底库做存在性校验**：能走到这里的姓名已经过
    ``identify`` 比对，若再查一次 face_persons 只是重复劳动；
    真正需要拦住的是「访客编号」，那一条在 ``check_in_by_face`` 里单独判。
    """
    s = (name or "").strip()
    if not s:
        raise PanelCheckinError("缺少打卡人姓名")
    if len(s) > 40:
        raise PanelCheckinError("姓名过长（最多 40 字）")
    return s


def _clean_note(note: str) -> str:
    return (note or "").strip()[:NOTE_MAX]


# ---------------------------------------------------------------- 读

def today() -> str:
    return time.strftime("%Y-%m-%d")


def list_events(day: str, limit: int = 500) -> list:
    """某日全部打卡事件，按打卡时间倒序（最新的一条在最上面）。"""
    day = validate_day(day)
    rows = get_conn().execute(
        "SELECT id, person_id, person_name, note, score, source, checkin_at, day "
        "FROM panel_checkin_events WHERE day = ? "
        "ORDER BY checkin_at DESC, id DESC LIMIT ?",
        (day, limit),
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["hm"] = (d["checkin_at"] or "")[11:16]     # 页面只需要时分
        d["score"] = round(d["score"] or 0, 3)
        d["status"] = ST_DONE
        d["status_cn"] = STATUS_LABEL[ST_DONE]
        d["note"] = d["note"] or ""
        out.append(d)
    return out


def status_of(name: str, day: str = "") -> dict:
    """单人当日的打卡状态 —— 页面状态块的数据源。

    返回 ``{person_name, day, status, status_cn, checkin_at, hm, times, score}``：

        status = done    → 已打卡，附首次打卡时间
        status = pending → 未打卡（今天还没刷过脸）

    一个人当天可以刷多次（中途外出回来再刷），
    **首次打卡时间**才是「打卡时刻」，因此取 MIN(checkin_at)；
    同时给出 ``times`` 让用户知道自己刷了几次。
    """
    name = _clean_name(name)
    day = validate_day(day) if day else today()
    row = get_conn().execute(
        "SELECT MIN(checkin_at) AS first_at, COUNT(*) AS n, MAX(score) AS best "
        "FROM panel_checkin_events WHERE person_name = ? AND day = ?",
        (name, day),
    ).fetchone()

    n = (row["n"] or 0) if row else 0
    if n:
        first_at = row["first_at"] or ""
        return {
            "person_name": name,
            "day": day,
            "status": ST_DONE,
            "status_cn": STATUS_LABEL[ST_DONE],
            "checkin_at": first_at,
            "hm": first_at[11:16],
            "times": n,
            "score": round(row["best"] or 0, 3),
        }
    return {
        "person_name": name,
        "day": day,
        "status": ST_PENDING,
        "status_cn": STATUS_LABEL[ST_PENDING],
        "checkin_at": "",
        "hm": "",
        "times": 0,
        "score": 0.0,
    }


def summary(day: str = "") -> dict:
    """当日汇总：打卡人数（去重）、打卡次数、最新一条打卡时间。"""
    day = validate_day(day) if day else today()
    row = get_conn().execute(
        "SELECT COUNT(DISTINCT person_name) AS persons, "
        "COUNT(*) AS events, MAX(checkin_at) AS last_at "
        "FROM panel_checkin_events WHERE day = ?",
        (day,),
    ).fetchone()
    d = dict(row) if row else {"persons": 0, "events": 0, "last_at": ""}
    return {
        "day": day,
        "persons": d["persons"] or 0,
        "events": d["events"] or 0,
        "last_at": d["last_at"] or "",
        "last_hm": (d["last_at"] or "")[11:16],
    }


# ---------------------------------------------------------------- 写

def _in_cooldown(name: str, day: str, now: float = None):
    """判断该人当日是否处于打卡冷却期。返回 (是否冷却, 上次打卡的秒数差)。"""
    t = time.time() if now is None else now
    with _cool_lock:
        last = _cool.get((name, day))
    if last is None:
        return False, None
    gap = t - last
    return (gap < COOLDOWN_SEC), gap


def _mark_cool(name: str, day: str, now: float = None) -> None:
    t = time.time() if now is None else now
    with _cool_lock:
        _cool[(name, day)] = t
        if len(_cool) > 512:             # 简单的容量保护
            cutoff = t - 86400
            for k in [k for k, v in _cool.items() if v < cutoff]:
                _cool.pop(k, None)


def reset_cooldown() -> None:
    """清空冷却状态（测试用）。"""
    with _cool_lock:
        _cool.clear()


def check_in(person_name: str, person_id=None, score: float = 0.0,
             note: str = "", day: str = "", source: str = "face",
             bypass_cooldown: bool = False) -> dict:
    """写入一条打卡记录。**只应由 ``check_in_by_face`` 调用**。

    保留成独立函数是为了单测能直接构造记录，不必真的跑人脸模型
    （人脸推理要 200~400ms 且依赖 188MB 的 onnx 模型，
    在 CI 里跑既慢又脆）。

    打卡时间一律取**服务器当前时刻**：不接受前端传入，
    否则时间记录可以被随意伪造，考勤也就失去意义。
    """
    name = _clean_name(person_name)
    note = _clean_note(note)
    day = validate_day(day) if day else today()

    at = _now()
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO panel_checkin_events "
        "(person_id, person_name, note, score, source, checkin_at, day) "
        "VALUES (?,?,?,?,?,?,?)",
        (person_id, name, note, round(float(score or 0), 4), source or "face", at, day),
    )
    conn.commit()
    if not bypass_cooldown:
        _mark_cool(name, day)

    st = status_of(name, day)
    return {
        "id": cur.lastrowid,
        "person_id": person_id,
        "person_name": name,
        "note": note,
        "score": round(float(score or 0), 3),
        "source": source or "face",
        "checkin_at": at,
        "hm": at[11:16],
        "day": day,
        "times": st["times"],
        # 第一次打卡与重复打卡分开返回，前端好给不同的提示文案
        "repeat": st["times"] > 1,
        "status": ST_DONE,
        "status_cn": STATUS_LABEL[ST_DONE],
    }


def check_in_by_face(img_bgr, day: str = "", note: str = "",
                     source: str = "face") -> dict:
    """⭐ 主入口：对一帧画面做人脸识别打卡。

    返回统一结构，**任何情况都不抛异常**（人脸是附加能力，
    坏了要能降级提示，不能把页面打崩）：:

        {
          ok:          是否完成打卡（只有识别到已注册的人且不在冷却期才为 True）
          reason:      ok=False 时的原因码
                       no_face / no_registered / cooldown / model_missing /
                       no_gallery / error
          message:     给用户看的中文提示（前端直接显示，不必再翻译一遍）
          recognized:  本帧是否认出了已注册人员
          faces:       本帧所有人脸（含访客），供前端画框
          record:      打卡成功时的记录；cooldown 时给上次的记录信息
          status:      该人当日打卡状态
        }

    为什么所有分支都返回 ``ok`` 字段而不是用 HTTP 状态码：
    刷脸是「用户站在镜头前反复发生」的动作，分不清「网络错了」
    和「没认出我」会让提示要么全是红色、要么全是空的。
    统一用 200 + ``reason`` 让前端按原因给不同的文案与颜色。
    """
    import face_db

    day = validate_day(day) if day else today()

    if not face_db.face_model_ready():
        return _fail("model_missing", "人脸模型未就绪，请先按 README「六之五」放入模型文件")

    # 底库为空时任何脸都会被判成访客，此时提示「先去录人脸」比
    # 「未识别到已注册人员」准确得多
    try:
        if face_db.count_persons() == 0:
            return _fail("no_gallery", "人脸底库为空，请先到「人脸底库」录入人员")
    except Exception:  # noqa: BLE001
        pass

    try:
        faces_raw = face_db.extract_faces(img_bgr)
    except face_db.FaceModelMissingError:
        return _fail("model_missing", "人脸模型未就绪")
    except Exception as e:  # noqa: BLE001
        return _fail("error", f"人脸检测失败：{e}")

    if not faces_raw:
        return _fail("no_face", "未检测到人脸，请将面部对准镜头再试")

    # 按检测置信度排序：画面里有多人时优先处理最清晰的那张脸
    faces_raw = sorted(faces_raw, key=lambda f: -float(f.get("det_score", 0)))

    faces, best = [], None
    for fc in faces_raw:
        try:
            pid, name, score, is_temp = face_db.identify(fc["embedding"])
        except Exception:  # noqa: BLE001
            continue
        faces.append({
            "name": name,
            "score": round(float(score), 3),
            "is_temp": bool(is_temp),
            "person_id": pid,
            "box": [round(float(v), 1) for v in fc["bbox"]],
            "det_score": round(float(fc.get("det_score", 0)), 3),
        })
        # 取第一个（即最清晰的）已注册的人作为打卡人
        if best is None and not is_temp and pid:
            best = (pid, name, score)

    if best is None:
        return {
            "ok": False,
            "reason": "no_registered",
            "message": "识别到人脸，但不在人脸底库中；请先到「人脸底库」录入后再打卡",
            "recognized": False,
            "faces": faces,
            "record": None,
            "status": None,
            "day": day,
        }

    pid, name, score = best
    st = status_of(name, day)

    # 冷却期：人还站在镜头前，不重复写库，但要告诉前端「已经打过了」
    cooling, gap = _in_cooldown(name, day)
    if cooling:
        return {
            "ok": False,
            "reason": "cooldown",
            "message": f"{name} 刚刚已打卡（{round(COOLDOWN_SEC - gap)} 秒后可再次识别）",
            "recognized": True,
            "faces": faces,
            "record": None,
            "status": st,
            "day": day,
        }

    rec = check_in(name, person_id=pid, score=score, note=note, day=day, source=source)
    rec["message"] = (
        f"{name} 已更新打卡时间 {rec['hm']}（今日第 {rec['times']} 次）"
        if rec["repeat"] else
        f"{name} 打卡成功，打卡时间 {rec['hm']}"
    )
    return {
        "ok": True,
        "reason": "done",
        "message": rec["message"],
        "recognized": True,
        "faces": faces,
        "record": rec,
        "status": status_of(name, day),
        "day": day,
    }


def _fail(reason: str, message: str) -> dict:
    return {
        "ok": False,
        "reason": reason,
        "message": message,
        "recognized": False,
        "faces": [],
        "record": None,
        "status": None,
    }


def delete_event(eid: int) -> bool:
    conn = get_conn()
    cur = conn.execute("DELETE FROM panel_checkin_events WHERE id = ?", (int(eid),))
    conn.commit()
    return cur.rowcount > 0


def clear_day(day: str = "") -> int:
    """清空某日全部打卡事件，返回删除条数。给「重置今日」按钮用。"""
    day = validate_day(day) if day else today()
    conn = get_conn()
    n = conn.execute(
        "SELECT COUNT(*) FROM panel_checkin_events WHERE day = ?", (day,)
    ).fetchone()[0]
    conn.execute("DELETE FROM panel_checkin_events WHERE day = ?", (day,))
    conn.commit()
    with _cool_lock:
        for k in [k for k in _cool if k[1] == day]:
            _cool.pop(k, None)
    return n
