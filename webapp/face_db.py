# -*- coding: utf-8 -*-
"""人脸底库：注册、识别与兼容统计

功能定位
--------
在既有的「安全帽 / 口罩」检测之外，补上「这个人是谁」的能力：
    - 已注册（有照片、有姓名）→ 识别出姓名
    - 未注册                    → 分配临时编号（访客-1、访客-2 ...）

设计取舍
--------
1. **只用 CPU**：本项目的 YOLO 检测已经占用 GPU，人脸识别是低频操作
   （注册时跑一次、识别时按需跑），CPU 推理（约 200-400ms/张）完全够用，
   还能避免与 YOLO 抢显存导致推理锁竞争。providers 固定 CPUExecutionProvider。

2. **模型必须离线加载**：insightface 默认在模型缺失时联网下载，
   而国内直连 GitHub/HuggingFace 常失败（本项目实测全部超时）。
   因此启动前统一清空代理解析路径，并把模型预置在 models/face/models/buffalo_l/。
   若预置模型缺失，直接抛 ModelMissingError，**绝不联网下载**——
   宁可明确报错告诉用户怎么放模型，也不要卡在一个永远不返回的请求上。

3. **特征归一化后存库**：比对用余弦相似度。入库前 L2 归一化，
   这样比对时只需点积，省掉每次算模长；也让阈值有统一的物理含义。

4. **阈值取 0.38**：ArcFace/MobileFaceNet 在 buffalo_l 上的常规经验区间是
   0.28~0.45。取 0.38 偏严，是为了降低「把张三认成李四」的误报——
   在门禁/考勤这类场景里，认错人比偶尔认不出更严重。
   戴口罩时特征退化，识别会失败，此时会自然回落到临时编号（安全降级）。
"""
import os
import sqlite3
import threading
import time

import numpy as np

import config

_local = threading.local()
_lock = threading.Lock()

# 单张脸相似度低于此值就不认，转而当作未注册人员
MATCH_THRESHOLD = 0.38

SCHEMA = """
CREATE TABLE IF NOT EXISTS face_persons (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,              -- 姓名
    note        TEXT,                       -- 备注：部门 / 工号等
    embedding   BLOB NOT NULL,              -- L2 归一化后的 512 维 float32
    thumb       TEXT,                       -- 注册照缩略图（static 下相对路径）
    created_at  TEXT NOT NULL,
    updated_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_face_persons_name ON face_persons(name);

CREATE TABLE IF NOT EXISTS face_seen (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    person_id   INTEGER,                    -- 命中的底库人员；NULL = 未注册
    person_name TEXT NOT NULL,              -- 姓名或临时编号
    is_temp     INTEGER DEFAULT 0,          -- 1 = 临时编号（未注册）
    score       REAL DEFAULT 0,             -- 相似度
    image_path  TEXT                        -- 抓拍图（static 下相对路径）
);
CREATE INDEX IF NOT EXISTS idx_face_seen_created ON face_seen(created_at);
CREATE INDEX IF NOT EXISTS idx_face_seen_person ON face_seen(person_id);
"""


class FaceModelMissingError(RuntimeError):
    """人脸模型未就绪"""


class NoFaceFound(RuntimeError):
    """图中未检出人脸"""


# 进程内单例，避免每次识别都重新加载 170MB 的 recognition 模型
_app = None
_app_lock = threading.Lock()


def _model_root() -> str:
    """insightface 要求的模型根目录：<root>/models/<name>/"""
    return str(config.FACE_MODEL_ROOT)


def face_model_ready() -> bool:
    """预置模型是否齐全（不触发加载，供接口/页面快速判断）。"""
    d = config.FACE_MODEL_ROOT / "models" / "buffalo_l"
    need = ("det_10g.onnx", "w600k_r50.onnx")
    return d.is_dir() and all((d / f).is_file() for f in need)


def _clean_proxy_env() -> None:
    """清掉代理相关环境变量。

    本机环境里有 https_proxy=http://127.0.0.1:59790，而 requests 会读取它。
    一旦 insightface 尝试下载模型，请求就会走这个代理并得到 502，
    报错信息还是「Tunnel connection failed」，很难联想到是代理问题。
    这里统一清空，保证离线加载路径干净。
    """
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(k, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


def get_app():
    """懒加载 FaceAnalysis（进程内单例）。"""
    global _app
    if _app is not None:
        return _app
    with _app_lock:
        if _app is not None:
            return _app
        if not face_model_ready():
            raise FaceModelMissingError(
                "人脸模型未就绪。请将 buffalo_l 的 det_10g.onnx / w600k_r50.onnx "
                f"放入 {config.FACE_MODEL_ROOT / 'models' / 'buffalo_l'}"
            )
        _clean_proxy_env()
        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:  # noqa: BLE001
            raise FaceModelMissingError(
                f"未安装 insightface（{e}）。请执行 pip install insightface onnxruntime"
            ) from e
        app = FaceAnalysis(
            name="buffalo_l",
            root=_model_root(),
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _app = app
        return _app


def warmup() -> bool:
    """预热模型。返回是否成功；失败不抛异常，由调用方决定是否提示。"""
    try:
        get_app()
        return True
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------- 特征提取

def extract_faces(img_bgr) -> list:
    """检测图中所有人脸，返回 [{bbox, score, embedding(归一化), det_score}]。"""
    app = get_app()
    faces = app.get(img_bgr)
    out = []
    for f in faces:
        emb = np.asarray(f.embedding, dtype=np.float32)
        n = float(np.linalg.norm(emb))
        if n > 0:
            emb = emb / n          # L2 归一化，后续比对直接点积
        out.append(
            {
                "bbox": [float(v) for v in f.bbox],
                "det_score": float(f.det_score),
                "embedding": emb,
            }
        )
    return out


def extract_single(img_bgr) -> np.ndarray:
    """只取图中最可信的一张脸的特征。检测到多张时取检测分数最高的。"""
    faces = extract_faces(img_bgr)
    if not faces:
        raise NoFaceFound("未在图片中检测到人脸，请换一张更清晰的正脸照片")
    best = max(faces, key=lambda x: x["det_score"])
    return best["embedding"]


# ---------------------------------------------------------------- 数据库

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


def _blob(emb: np.ndarray) -> bytes:
    return np.asarray(emb, dtype=np.float32).tobytes()


def _unblob(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- 底库 CRUD

def register(name: str, embedding, note: str = "", thumb: str = "") -> int:
    """注册一个人。同名允许重复（不同人可能重名），由 id 区分。"""
    name = (name or "").strip()
    if not name:
        raise ValueError("姓名不能为空")
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO face_persons (name, note, embedding, thumb, created_at) "
        "VALUES (?,?,?,?,?)",
        (name, note or "", _blob(embedding), thumb or "", _now()),
    )
    conn.commit()
    return cur.lastrowid


def list_persons():
    rows = get_conn().execute(
        "SELECT id, name, note, thumb, created_at FROM face_persons ORDER BY id DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def get_person(pid: int):
    r = get_conn().execute(
        "SELECT * FROM face_persons WHERE id=?", (pid,)
    ).fetchone()
    return dict(r) if r else None


def delete_person(pid: int) -> bool:
    conn = get_conn()
    cur = conn.execute("DELETE FROM face_persons WHERE id=?", (pid,))
    conn.commit()
    return cur.rowcount > 0


def count_persons() -> int:
    return get_conn().execute("SELECT COUNT(*) FROM face_persons").fetchone()[0]


# ---------------------------------------------------------------- 识别

def _load_gallery():
    """把底库全部加载成矩阵，便于一次性向量化比对。"""
    rows = get_conn().execute(
        "SELECT id, name, embedding FROM face_persons"
    ).fetchall()
    if not rows:
        return [], [], None
    ids, names, vecs = [], [], []
    dim = None
    for r in rows:
        v = _unblob(r["embedding"])
        if vecs and len(v) != dim:
            continue          # 维度不一致的脏数据跳过，不让它污染整个比对
        dim = len(v)
        ids.append(r["id"])
        names.append(r["name"])
        vecs.append(v)
    if not vecs:
        return [], [], None
    return ids, names, np.vstack(vecs)


def identify(embedding):
    """在底库中找最相似的人。

    返回 (person_id, name, score, is_temp)
        - 命中：person_id 为底库 id，is_temp=False
        - 未命中：person_id=None，name 为临时编号，is_temp=True
    """
    ids, names, mat = _load_gallery()
    emb = np.asarray(embedding, dtype=np.float32)
    n = float(np.linalg.norm(emb))
    if n > 0:
        emb = emb / n

    if mat is None or not ids:
        return None, _temp_name(emb), 0.0, True

    sims = mat @ emb                     # 双方都已归一化，点积即余弦相似度
    idx = int(np.argmax(sims))
    best = float(sims[idx])
    # 先舍入再比阈值，让「显示多少就按多少判」。
    # 否则会出现在页面上看到 0.380（已舍入到 4 位）却被判为未命中的怪现象——
    # 实际是 0.3799999 这类浮点误差造成的，用户无法理解也无法复现。
    best_r = round(best, 4)
    if best_r >= MATCH_THRESHOLD:
        return ids[idx], names[idx], best_r, False
    return None, _temp_name(emb), best_r, True


# 临时编号：同一个未注册的人在本次运行期内应保持同一个编号，
# 否则他在底库里会被拆成访客-1、访客-2、访客-3，统计失去意义。
# 做法是把特征做一次粗量化当指纹，在内存里查表归并。
# 不落库、不跨重启——重启后重新发号是可以接受的，
# 因为「访客」本身就是会话性概念，且落库会让底库被临时数据污染。
_temp_registry: dict = {}
_temp_names: list = []          # 已知的访客特征矩阵（归一化）
_temp_counter = 0

# 归并为同一个访客的相似度门槛。刻意比 MATCH_THRESHOLD 更松：
# 宁可把同一个访客拆成两个，也不要让两个不同的人共用一个编号——
# 后者会直接导致「甲的违规记录算到乙头上」。
_TEMP_MERGE_THRESHOLD = 0.30


def _temp_name(embedding=None) -> str:
    """为未注册人脸分配稳定的临时编号。

    传入 embedding 时会先在已发号的访客里找相似者，命中则复用原编号。
    """
    global _temp_counter, _temp_names
    if embedding is None:
        with _lock:
            _temp_counter += 1
            return f"访客-{_temp_counter}"

    emb = np.asarray(embedding, dtype=np.float32)
    n = float(np.linalg.norm(emb))
    if n > 0:
        emb = emb / n
    with _lock:
        if _temp_names:
            mat = np.vstack(_temp_names)
            sims = mat @ emb
            idx = int(np.argmax(sims))
            if float(sims[idx]) >= _TEMP_MERGE_THRESHOLD:
                return f"访客-{idx + 1}"
        _temp_counter += 1
        _temp_names.append(emb)
        return f"访客-{_temp_counter}"


def reset_temp_registry() -> None:
    """清空临时编号表（测试用，也便于用户手动重置访客计数）。"""
    global _temp_counter, _temp_names
    with _lock:
        _temp_counter = 0
        _temp_names = []


# ---------------------------------------------------------------- 识别记录

def add_seen(person_id, person_name, is_temp, score=0.0, image_path="") -> int:
    """记录一次人脸识别结果。"""
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO face_seen (created_at, person_id, person_name, is_temp, "
        "score, image_path) VALUES (?,?,?,?,?,?)",
        (
            _now(),
            person_id,
            person_name,
            1 if is_temp else 0,
            round(float(score), 4),
            image_path or "",
        ),
    )
    conn.commit()
    return cur.lastrowid


def query_seen(page=1, size=20, person=None, only_temp=None):
    """分页查询识别记录。person 支持按姓名/临时编号模糊匹配。"""
    where, params = [], []
    if person:
        where.append("person_name LIKE ?")
        params.append(f"%{person}%")
    if only_temp is not None:
        where.append("is_temp = ?")
        params.append(1 if only_temp else 0)
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    conn = get_conn()
    total = conn.execute(
        f"SELECT COUNT(*) FROM face_seen {clause}", params
    ).fetchone()[0]
    rows = conn.execute(
        f"SELECT * FROM face_seen {clause} ORDER BY id DESC LIMIT ? OFFSET ?",
        params + [size, (page - 1) * size],
    ).fetchall()
    return [dict(r) for r in rows], total


def delete_seen(ids):
    qs = ",".join("?" for _ in ids)
    conn = get_conn()
    conn.execute(f"DELETE FROM face_seen WHERE id IN ({qs})", list(ids))
    conn.commit()


def clear_seen() -> int:
    """清空全部识别记录，返回删除条数。"""
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) FROM face_seen").fetchone()[0]
    conn.execute("DELETE FROM face_seen")
    conn.commit()
    return n


# ---------------------------------------------------------------- 统计

def stats_persons(limit: int = 50):
    """按人聚合的识别次数，用于合规统计页的排行。

    已注册人员与临时访客分组返回，因为二者含义不同：
    已注册的是「这个人来过几次」，临时访客只能说明「有未登记的人出现过」。
    """
    conn = get_conn()
    reg = conn.execute(
        "SELECT person_name, COUNT(*) n, MAX(created_at) last_at, "
        "ROUND(AVG(score),3) avg_score "
        "FROM face_seen WHERE is_temp=0 GROUP BY person_name "
        "ORDER BY n DESC LIMIT ?",
        (limit,),
    ).fetchall()
    tmp = conn.execute(
        "SELECT person_name, COUNT(*) n, MAX(created_at) last_at, "
        "ROUND(AVG(score),3) avg_score "
        "FROM face_seen WHERE is_temp=1 GROUP BY person_name "
        "ORDER BY n DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return {
        "registered": [dict(r) for r in reg],
        "temp": [dict(r) for r in tmp],
    }


def stats_seen_summary() -> dict:
    """识别总览：总次数、注册/未注册占比、底库人数。"""
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) FROM face_seen").fetchone()[0]
    temp = conn.execute(
        "SELECT COUNT(*) FROM face_seen WHERE is_temp=1"
    ).fetchone()[0]
    today = time.strftime("%Y-%m-%d")
    today_n = conn.execute(
        "SELECT COUNT(*) FROM face_seen WHERE created_at LIKE ?", (today + "%",)
    ).fetchone()[0]
    distinct_reg = conn.execute(
        "SELECT COUNT(DISTINCT person_name) FROM face_seen WHERE is_temp=0"
    ).fetchone()[0]
    distinct_temp = conn.execute(
        "SELECT COUNT(DISTINCT person_name) FROM face_seen WHERE is_temp=1"
    ).fetchone()[0]
    return {
        "total": total,
        "today": today_n,
        "registered_seen": total - temp,
        "temp_seen": temp,
        "distinct_registered": distinct_reg,
        "distinct_temp": distinct_temp,
        "gallery_size": count_persons(),
    }


def stats_seen_daily(days: int = 14):
    """近 N 天识别次数（注册 / 未注册分开），补齐空日期。"""
    conn = get_conn()
    since = time.strftime(
        "%Y-%m-%d", time.localtime(time.time() - (days - 1) * 86400)
    )
    rows = conn.execute(
        "SELECT substr(created_at,1,10) d, "
        "SUM(CASE WHEN is_temp=0 THEN 1 ELSE 0 END) reg, "
        "SUM(CASE WHEN is_temp=1 THEN 1 ELSE 0 END) tmp "
        "FROM face_seen WHERE created_at >= ? GROUP BY d",
        (since,),
    ).fetchall()
    by_date = {r[0]: r for r in rows}
    out = []
    for i in range(days):
        ts = time.localtime(time.time() - (days - 1 - i) * 86400)
        day = time.strftime("%Y-%m-%d", ts)
        r = by_date.get(day)
        out.append(
            {
                "date": day[5:],
                "registered": r[1] if r else 0,
                "temp": r[2] if r else 0,
            }
        )
    return out
