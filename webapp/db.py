# -*- coding: utf-8 -*-
"""SQLite 检测记录管理（线程安全的薄封装）"""
import json
import sqlite3
import threading
from datetime import datetime, timedelta

import config

_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,
    source_type  TEXT NOT NULL,          -- image / video / camera
    source_name  TEXT,                   -- 文件名或来源描述
    kinds        TEXT,                   -- helmet / mask / helmet+mask
    num_objects  INTEGER DEFAULT 0,      -- 检出目标总数
    helmet       INTEGER DEFAULT 0,      -- 安全帽数
    head         INTEGER DEFAULT 0,      -- 未戴安全帽数（违规）
    mask         INTEGER DEFAULT 0,      -- 口罩数
    face         INTEGER DEFAULT 0,      -- 未戴口罩数（违规）
    avg_conf     REAL DEFAULT 0,         -- 平均置信度
    duration_ms  REAL DEFAULT 0,         -- 推理耗时(毫秒)
    image_path   TEXT,                   -- 结果图（static 下相对路径）
    video_path   TEXT,                   -- 结果视频（static 下相对路径）
    details      TEXT                    -- JSON 目标明细
);
CREATE INDEX IF NOT EXISTS idx_records_created ON records(created_at);
CREATE INDEX IF NOT EXISTS idx_records_source ON records(source_type);
"""


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
    """关闭当前线程持有的连接。

    连接是线程局部（thread-local）持有的，进程退出时由解释器回收；
    但测试或需要释放文件句柄（例如删除数据库文件）时应显式调用，
    否则 Windows 上会因文件被占用而无法删除。
    """
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        finally:
            _local.conn = None


def insert_record(**kw) -> int:
    kw.setdefault("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if isinstance(kw.get("details"), (list, dict)):
        kw["details"] = json.dumps(kw["details"], ensure_ascii=False)
    cols = ", ".join(kw.keys())
    marks = ", ".join("?" for _ in kw)
    cur = get_conn().execute(f"INSERT INTO records ({cols}) VALUES ({marks})", list(kw.values()))
    get_conn().commit()
    return cur.lastrowid


def query_records(page=1, size=10, kind=None, source=None, keyword=None):
    where, params = [], []
    if kind:
        where.append("kinds LIKE ?")
        params.append(f"%{kind}%")
    if source:
        where.append("source_type = ?")
        params.append(source)
    if keyword:
        where.append("source_name LIKE ?")
        params.append(f"%{keyword}%")
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    total = get_conn().execute(
        f"SELECT COUNT(*) FROM records {clause}", params
    ).fetchone()[0]
    rows = get_conn().execute(
        f"SELECT * FROM records {clause} ORDER BY id DESC LIMIT ? OFFSET ?",
        params + [size, (page - 1) * size],
    ).fetchall()
    return [dict(r) for r in rows], total


def get_record(rid: int):
    r = get_conn().execute("SELECT * FROM records WHERE id=?", (rid,)).fetchone()
    return dict(r) if r else None


def delete_records(ids):
    qs = ",".join("?" for _ in ids)
    get_conn().execute(f"DELETE FROM records WHERE id IN ({qs})", list(ids))
    get_conn().commit()


def stats_summary() -> dict:
    c = get_conn()
    total = c.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    today = datetime.now().strftime("%Y-%m-%d")
    today_n = c.execute(
        "SELECT COUNT(*) FROM records WHERE created_at LIKE ?", (today + "%",)
    ).fetchone()[0]
    row = c.execute(
        "SELECT COALESCE(SUM(helmet),0), COALESCE(SUM(head),0), "
        "COALESCE(SUM(mask),0), COALESCE(SUM(face),0), COALESCE(SUM(num_objects),0) FROM records"
    ).fetchone()
    return {
        "total_records": total,
        "today_records": today_n,
        "helmet": row[0],
        "head": row[1],
        "mask": row[2],
        "face": row[3],
        "total_objects": row[4],
        "violations": row[1] + row[3],
    }


def stats_daily(days: int = 14) -> list:
    """近 N 天每日记录数 / 目标数 / 违规数（补齐空日期）。"""
    c = get_conn()
    since = (datetime.now() - timedelta(days=days - 1)).strftime("%Y-%m-%d")
    rows = c.execute(
        "SELECT substr(created_at,1,10) d, COUNT(*) n, "
        "COALESCE(SUM(num_objects),0) objs, "
        "COALESCE(SUM(head),0)+COALESCE(SUM(face),0) viol "
        "FROM records WHERE created_at >= ? GROUP BY d",
        (since,),
    ).fetchall()
    by_date = {r[0]: r for r in rows}
    out = []
    for i in range(days):
        day = (datetime.now() - timedelta(days=days - 1 - i)).strftime("%Y-%m-%d")
        r = by_date.get(day)
        out.append(
            {
                "date": day[5:],  # MM-DD
                "records": r[1] if r else 0,
                "objects": r[2] if r else 0,
                "violations": r[3] if r else 0,
            }
        )
    return out


def stats_classes() -> dict:
    row = get_conn().execute(
        "SELECT COALESCE(SUM(helmet),0), COALESCE(SUM(head),0), "
        "COALESCE(SUM(mask),0), COALESCE(SUM(face),0) FROM records"
    ).fetchone()
    return {"安全帽": row[0], "未戴安全帽": row[1], "口罩": row[2], "未戴口罩": row[3]}


def stats_sources() -> dict:
    rows = get_conn().execute(
        "SELECT source_type, COUNT(*) FROM records GROUP BY source_type"
    ).fetchall()
    name_map = {"image": "图片检测", "video": "视频检测", "camera": "实时监控"}
    return {name_map.get(r[0], r[0]): r[1] for r in rows}


def stats_conf() -> list:
    """平均置信度分布直方图。"""
    rows = get_conn().execute(
        "SELECT avg_conf FROM records WHERE num_objects > 0 AND avg_conf > 0"
    ).fetchall()
    edges = [0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    labels = ["<0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", ">0.9"]
    counts = [0] * len(edges)
    for (v,) in rows:
        for i, e in enumerate(edges):
            if v < e:
                counts[i] += 1
                break
    return [{"label": l, "count": c} for l, c in zip(labels, counts)]
