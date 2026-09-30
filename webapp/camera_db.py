# -*- coding: utf-8 -*-
"""监控摄像头配置：增删改查、地址校验与脱敏

原来摄像头只有「一个」：由环境变量 `CAMERA_SOURCE` / `CAMERA_INDEX` 决定，
想换一个必须改配置文件并重启服务。现场有多路摄像头时这不现实，
因此把摄像头做成可配置的列表，存在 SQLite 里，前端可以随时添加/切换/删除。

支持的三类来源
--------------
| type       | 值示例                                  | 说明 |
|------------|-----------------------------------------|------|
| `rtsp`     | `rtsp://user:pass@10.0.0.8:554/stream1` | 网络摄像头（IPC），工地最常见 |
| `http`     | `http://10.0.0.8:8080/video`            | MJPEG 流（部分 IPC / 手机推流 App） |
| `device`   | `0` / `1`                               | 本机 USB 摄像头或采集卡的设备索引 |

安全考虑（重要）
----------------
RTSP 地址里常带 `user:pass@`，**这是明文凭据**。因此：
1. 列表接口一律返回**脱敏后**的地址（`rtsp://***:***@10.0.0.8:554/...`）；
2. 只有真正要连流时才取原文（`resolve_url()`），且不写日志；
3. 提交到 GitHub 前必须确认数据库文件不在版本控制内。
"""
import re
import sqlite3
import threading
import time

import config

_local = threading.local()
_lock = threading.Lock()

SOURCE_TYPES = ("rtsp", "http", "device")
TYPE_CN = {"rtsp": "RTSP 网络摄像头", "http": "HTTP / MJPEG 流", "device": "本机设备索引"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,                 -- 展示名，如「东门岗亭」
    type        TEXT NOT NULL,                 -- rtsp / http / device
    url         TEXT NOT NULL,                 -- 原始地址（含凭据，勿外泄）
    location    TEXT,                          -- 安装位置备注
    note        TEXT,                          -- 其它备注
    enabled     INTEGER DEFAULT 1,             -- 是否启用
    last_ok_at  TEXT,                          -- 最近一次连通成功时间
    last_error  TEXT,                          -- 最近一次失败原因
    created_at  TEXT NOT NULL,
    updated_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_cameras_type ON cameras(type);
"""


class CameraConfigError(ValueError):
    """摄像头配置非法"""


def get_conn() -> sqlite3.Connection:
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
    conn.commit()


def close() -> None:
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        finally:
            _local.conn = None


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- 校验与脱敏

# 地址里 user:pass@ 的部分。用非贪婪匹配，避免把端口后的斜杠也吃进去。
_CRED_RE = re.compile(r"^(?P<scheme>[a-zA-Z][\w+.-]*://)(?P<user>[^:/@]+)(:(?P<pwd>[^@/]*))?@")


def mask_url(url: str) -> str:
    """把地址里的凭据替换成 ***，用于对外展示。

    即使密码本身为空（`user@host`）也要遮住用户名 ——
    用户名同样属于敏感信息，且能提示「这里有凭据」。
    """
    if not url:
        return ""
    s = str(url).strip()
    m = _CRED_RE.match(s)
    if m:
        return s[:m.start("user")] + "***:***@" + s[m.end():]
    return s


def validate(name: str, ctype: str, url: str) -> tuple:
    """校验并归一化摄像头配置，返回 (name, type, url)。

    抛 CameraConfigError 时消息可直接展示给用户 ——
    这些错误都是用户填错字段造成的，说得越具体越好。
    """
    name = (name or "").strip()
    ctype = (ctype or "").strip().lower()
    url = (url or "").strip()

    if not name:
        raise CameraConfigError("请填写摄像头名称")
    if len(name) > 40:
        raise CameraConfigError("名称过长（最多 40 字）")
    if ctype not in SOURCE_TYPES:
        raise CameraConfigError(f"类型必须是 {'/'.join(SOURCE_TYPES)} 之一")
    if not url:
        raise CameraConfigError("请填写摄像头地址")

    if ctype == "device":
        # 设备索引必须是非负整数，越界交给 OpenCV 报错（它最清楚本机有几个设备）
        if not url.isdigit():
            raise CameraConfigError("本机设备索引必须是非负整数，例如 0、1")
        if int(url) > 64:
            raise CameraConfigError("设备索引过大，通常不会超过 64")
        return name, ctype, url

    low = url.lower()
    if ctype == "rtsp" and not low.startswith("rtsp://"):
        raise CameraConfigError("RTSP 地址必须以 rtsp:// 开头")
    if ctype == "http" and not (low.startswith("http://") or low.startswith("https://")):
        raise CameraConfigError("HTTP 地址必须以 http:// 或 https:// 开头")
    # 明显不是地址的输入早拦掉，省得连半天超时才发现是打错了
    if " " in url:
        raise CameraConfigError("地址中不能含空格")
    if len(url) > 500:
        raise CameraConfigError("地址过长")
    return name, ctype, url


# ---------------------------------------------------------------- CRUD

def add(name, ctype, url, location="", note="") -> int:
    name, ctype, url = validate(name, ctype, url)
    conn = get_conn()
    now = _now()
    cur = conn.execute(
        "INSERT INTO cameras (name, type, url, location, note, enabled, created_at) "
        "VALUES (?,?,?,?,?,1,?)",
        (name, ctype, url, (location or "").strip()[:80], (note or "").strip()[:200], now),
    )
    conn.commit()
    return cur.lastrowid


def update(cid: int, name=None, ctype=None, url=None,
           location=None, note=None, enabled=None) -> bool:
    """局部更新。只更新传入的字段，None 表示「不改」。"""
    cur = get_person_exists(cid)
    if not cur:
        return False
    row = get(cid)
    n_name = row["name"] if name is None else name
    n_type = row["type"] if ctype is None else ctype
    n_url = row["url"] if url is None else url
    n_name, n_type, n_url = validate(n_name, n_type, n_url)

    sets, vals = ["name=?", "type=?", "url=?"], [n_name, n_type, n_url]
    if location is not None:
        sets.append("location=?"); vals.append(str(location).strip()[:80])
    if note is not None:
        sets.append("note=?"); vals.append(str(note).strip()[:200])
    if enabled is not None:
        sets.append("enabled=?"); vals.append(1 if enabled else 0)
    sets.append("updated_at=?"); vals.append(_now())
    vals.append(cid)

    conn = get_conn()
    conn.execute(f"UPDATE cameras SET {', '.join(sets)} WHERE id=?", vals)
    conn.commit()
    return True


def get_person_exists(cid) -> bool:
    try:
        cid = int(cid)
    except (TypeError, ValueError):
        return False
    return get_conn().execute("SELECT 1 FROM cameras WHERE id=?", (cid,)).fetchone() is not None


def get(cid: int):
    row = get_conn().execute("SELECT * FROM cameras WHERE id=?", (int(cid),)).fetchone()
    return dict(row) if row else None


def list_cameras(masked: bool = True):
    """列出全部摄像头。默认脱敏，`masked=False` 才返回原始地址。"""
    rows = get_conn().execute("SELECT * FROM cameras ORDER BY id").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["type_cn"] = TYPE_CN.get(d["type"], d["type"])
        if masked:
            d["url"] = mask_url(d["url"])
        out.append(d)
    return out


def delete(cid: int) -> bool:
    conn = get_conn()
    cur = conn.execute("DELETE FROM cameras WHERE id=?", (int(cid),))
    conn.commit()
    return cur.rowcount > 0


def count() -> int:
    return get_conn().execute("SELECT COUNT(*) FROM cameras").fetchone()[0]


def resolve_url(cid: int):
    """取用于连流的地址与类型（**含明文凭据**，只在建立连接时调用）。

    返回 (type, url 或设备索引)：
        - device 类型返回 int 索引（OpenCV 需要整数）
        - 其余返回原始 URL 字符串
    """
    row = get(int(cid))
    if not row:
        return None
    if row["type"] == "device":
        try:
            return "device", int(row["url"])
        except (TypeError, ValueError):
            return None
    return row["type"], row["url"]


def mark_result(cid: int, ok: bool, error: str = "") -> None:
    """记录最近一次连通结果，便于前端显示「在线 / 异常」。"""
    try:
        conn = get_conn()
        if ok:
            conn.execute(
                "UPDATE cameras SET last_ok_at=?, last_error='' WHERE id=?",
                (_now(), int(cid)),
            )
        else:
            conn.execute(
                "UPDATE cameras SET last_error=? WHERE id=?",
                (str(error)[:200], int(cid)),
            )
        conn.commit()
    except Exception:  # noqa: BLE001
        pass          # 只是状态记录，失败不该影响主流程


def seed_from_env() -> int:
    """首次启动时把环境变量里的那路摄像头导入配置表。

    只在表为空时执行，避免每次启动都塞一条重复记录。
    """
    if count() > 0:
        return 0
    src = (config.CAMERA_SOURCE or "").strip()
    if src:
        low = src.lower()
        ctype = "rtsp" if low.startswith("rtsp://") else "http"
        try:
            cid = add("默认摄像头", ctype, src, location="由环境变量导入")
        except CameraConfigError:
            return 0
    else:
        try:
            cid = add("本机摄像头", "device", str(config.CAMERA_INDEX),
                      location="由环境变量导入")
        except CameraConfigError:
            return 0
    return 1 if cid else 0
