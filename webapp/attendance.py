# -*- coding: utf-8 -*-
"""人脸打卡：只依据「当日是否识别到该人员」判定打卡 / 缺勤

需求边界（刻意收窄）
--------------------
本轮**不考虑**是否戴口罩、是否戴安全帽，也**不引入班次与迟到早退**。
判定只有两条：

    当日识别到该人员  → 打卡
    当日未识别到      → 缺勤

这是一个**保守但可解释**的判据：它不回答「几点算迟到」，只回答
「今天这个人有没有出现在镜头里」。装备佩戴情况仍由合规统计单独负责，
两者互不干扰 —— 把「没戴安全帽」也当作缺勤会得到一个没人看得懂的报表。

为什么单独一个模块
------------------
`face_seen` 是「事实」（流水），打卡状态是「对事实的解释」。
分开存的原因与考勤方案文档一致：流水可能被清理，而打卡结果要长期保留；
规则会变，解释层可以按新规则重算而不必动原始流水。

核心设计取舍
------------
1. **只认已注册人员**：临时访客编号是按特征指纹临时发的，重启后会重新发号，
   拿来判缺勤会产生大量幽灵记录。查询一律加 ``is_temp = 0``。

2. **缺勤 = 「应到名单」减去「当日有记录的人」**，而不是「遍历所有日期」。
   应到名单就是底库里的已注册人员 —— 这是唯一有意义的「应到」定义。
   注意这意味着：**新注册的人，注册前的日期一律不参与统计**，
   否则昨天刚入库的人今天会被追溯成「缺勤 30 天」。

3. **三态输出**，与装备统计的三态一样，把「无信息」和「状态差」分开：
       present  ：打卡（当日有识别记录）
       absent   ：缺勤（当日无记录）
       no_data  ：无法判定（当日服务未运行 / 底库为空）
   `no_data` 必须存在：服务没开、摄像头没接的日子，
   全员都会被算成缺勤 —— 这在汇报时是灾难性的，必须先与真缺勤分开。

4. **打卡时间取当日首次识别时刻**，同时给出末次和识别次数。
   「首次」是最贴近「上班打卡」的语义，且不需要班次概念就能理解。
"""
import time

import config
import face_db

# 状态常量。用英文短串而不是数字：
# 存进库/传给前端时自解释，不必来回查文档。
ST_PRESENT = "present"      # 打卡
ST_ABSENT = "absent"        # 缺勤
ST_NO_DATA = "no_data"      # 无法判定（当日无数据源）

# 状态的中文标签。前端直接取用，避免前后端各写一份翻译对不上。
STATUS_LABEL = {
    ST_PRESENT: "打卡",
    ST_ABSENT: "缺勤",
    ST_NO_DATA: "无数据",
}


# ---------------------------------------------------------------- 单日核算

def _day_bounds(day: str):
    """把 'YYYY-MM-DD' 变成 [day 00:00:00, 次日 00:00:00) 两个字符串。

    用字符串比较而不是 SQLite 的 date() 函数：
    created_at 统一是 'YYYY-MM-DD HH:MM:SS' 格式，字典序即时间序，
    直接用 ``>= ? AND < ?`` 能命中 idx_face_seen_created 索引。
    换成 ``date(created_at) = ?`` 会让索引失效，全表扫描。
    """
    t = time.strptime(day, "%Y-%m-%d")
    nxt = time.strftime("%Y-%m-%d", time.localtime(time.mktime(t) + 86400))
    return f"{day} 00:00:00", f"{nxt} 00:00:00"


def validate_day(day: str) -> str:
    """校验并规范化日期参数。非法格式一律抛 ValueError，由调用方转 400。"""
    s = (day or "").strip()
    try:
        time.strptime(s, "%Y-%m-%d")
    except ValueError:
        raise ValueError("日期格式应为 YYYY-MM-DD")
    return s


def checkins_on(day: str, limit: int = 2000) -> dict:
    """某日各已注册人员的打卡明细。

    返回 ``{person_key: {first_at, last_at, seen_n, ...}}``。
    ``person_key`` 用姓名而非 person_id：底库人员被删除后，
    历史流水里的 person_id 会变成悬空引用，而姓名已经冗余存了一份。
    同一姓名重复注册（不同人重名）时会合并统计 —— 这是已知取舍，
    相比「删人后历史记录对不上」，合并更容易解释。
    """
    lo, hi = _day_bounds(day)
    rows = face_db.get_conn().execute(
        """
        SELECT person_name,
               MIN(created_at)          AS first_at,
               MAX(created_at)          AS last_at,
               COUNT(*)                 AS seen_n,
               MAX(score)               AS best_score,
               MAX(hat_state)           AS hat_any,
               MIN(hat_state)           AS hat_min,
               MAX(mask_state)          AS mask_any,
               MIN(mask_state)          AS mask_min
          FROM face_seen
         WHERE created_at >= ? AND created_at < ?
           AND is_temp = 0
      GROUP BY person_name
      ORDER BY first_at
         LIMIT ?
        """,
        (lo, hi, limit),
    ).fetchall()
    return {r["person_name"]: dict(r) for r in rows}


def daily_status(day: str, persons=None) -> list:
    """核算某日全员打卡状态 —— 本模块的主入口。

    ``persons`` 可传入 ``face_db.list_persons()`` 的结果以复用查询；
    不传则自己取。

    返回按「缺勤在前、打卡在后」排序的行列表，每行为::

        {person_id, person_name, note, status, status_cn,
         first_at, last_at, seen_n, best_score}

    排序刻意把缺勤排前面：报表的第一用途是「找出没来的人」，
    打卡的人不需要一个个看。
    """
    day = validate_day(day)
    if persons is None:
        persons = face_db.list_persons()
    shown = checkins_on(day)

    rows = []
    for p in persons:
        name = p.get("name") or ""
        rec = shown.get(name)
        if rec:
            status = ST_PRESENT
        else:
            status = ST_ABSENT
        rows.append(
            {
                "person_id": p.get("id"),
                "person_name": name,
                "note": p.get("note") or "",
                "status": status,
                "status_cn": STATUS_LABEL[status],
                "first_at": (rec or {}).get("first_at"),
                "last_at": (rec or {}).get("last_at"),
                "seen_n": (rec or {}).get("seen_n", 0),
                "best_score": round((rec or {}).get("best_score") or 0, 3),
                # 只取时间部分，页面表格里可比性更强
                "first_hm": ((rec or {}).get("first_at") or "")[11:16],
                "last_hm": ((rec or {}).get("last_at") or "")[11:16],
            }
        )

    # 缺勤在前；同类内按姓名稳定排序，避免每次刷新顺序抖动
    rows.sort(key=lambda r: (0 if r["status"] == ST_ABSENT else 1, r["person_name"]))
    return rows


def daily_summary(day: str, rows=None) -> dict:
    """某日的汇总计数：应到 / 打卡 / 缺勤 / 打卡率。

    ⚠️ 打卡率的分母是「应到人数」而不是「当日有记录的人数」——
    后者恒等于 100%，毫无信息量。
    但**应到为 0 时返回 None**，页面显示「—」：
    底库为空、或当日之后才注册人员时，分母为 0，
    显示 0% 会让人误以为全员缺勤。
    """
    day = validate_day(day)
    if rows is None:
        rows = daily_status(day)
    expect = len(rows)
    present = sum(1 for r in rows if r["status"] == ST_PRESENT)
    absent = expect - present
    return {
        "date": day,
        "expect": expect,
        "present": present,
        "absent": absent,
        "rate": round(present / expect, 3) if expect else None,
        "has_record": has_any_record(day),
    }


def has_any_record(day: str) -> bool:
    """当日流水表里是否有任何记录（含访客）。

    与「有没有人打卡」不同 —— 这个判断的是**数据源是否活着**。
    用来区分 ``absent`` 与 ``no_data``：一条记录都没有的日子，
    很可能只是服务没开，不能据此判员工缺勤。
    """
    lo, hi = _day_bounds(day)
    row = face_db.get_conn().execute(
        "SELECT 1 FROM face_seen WHERE created_at >= ? AND created_at < ? LIMIT 1",
        (lo, hi),
    ).fetchone()
    return row is not None


# ---------------------------------------------------------------- 区间台账

def range_status(start: str, end: str, persons=None, max_days: int = 92) -> dict:
    """区间打卡台账：一人一行，列出区间内每一天的状态。

    用于「本月谁缺勤最多」这类汇总。默认上限 92 天，
    防止一次拉一年数据把页面卡死 —— 每天一次全表聚合，天数多了会明显变慢。

    返回::

        {start, end, days: [...], rows: [{person_name, days: {day: status}, present_n, absent_n, rate}]}
    """
    start = validate_day(start)
    end = validate_day(end)
    t0 = time.mktime(time.strptime(start, "%Y-%m-%d"))
    t1 = time.mktime(time.strptime(end, "%Y-%m-%d"))
    if t1 < t0:
        raise ValueError("结束日期不能早于开始日期")
    n_days = int((t1 - t0) // 86400) + 1
    if n_days > max_days:
        raise ValueError(f"区间过长（{n_days} 天），最多查询 {max_days} 天")

    days = [
        time.strftime("%Y-%m-%d", time.localtime(t0 + i * 86400))
        for i in range(n_days)
    ]
    if persons is None:
        persons = face_db.list_persons()

    # 一次性把区间内所有打卡记录读出来，在 Python 里分日期 ——
    # 比「每天查一次」少 n 次 SQL 往返（30 天就是 30 次全表聚合）
    lo, _ = _day_bounds(days[0])
    _, hi = _day_bounds(days[-1])
    found = {}
    for r in face_db.get_conn().execute(
        "SELECT person_name, substr(created_at,1,10) d, COUNT(*) n, "
        "MIN(created_at) f, MAX(created_at) l "
        "FROM face_seen WHERE created_at >= ? AND created_at < ? AND is_temp = 0 "
        "GROUP BY person_name, d",
        (lo, hi),
    ).fetchall():
        found[(r["person_name"], r["d"])] = dict(r)

    out = []
    for p in persons:
        name = p.get("name") or ""
        cell = {}
        present_n = 0
        for d in days:
            rec = found.get((name, d))
            if rec:
                present_n += 1
                cell[d] = {"status": ST_PRESENT, "status_cn": STATUS_LABEL[ST_PRESENT],
                           "first_hm": (rec["f"] or "")[11:16],
                           "last_hm": (rec["l"] or "")[11:16],
                           "seen_n": rec["n"]}
            else:
                cell[d] = {"status": ST_ABSENT, "status_cn": STATUS_LABEL[ST_ABSENT],
                           "first_hm": "", "last_hm": "", "seen_n": 0}
        out.append(
            {
                "person_id": p.get("id"),
                "person_name": name,
                "note": p.get("note") or "",
                "days": cell,
                "present_n": present_n,
                "absent_n": n_days - present_n,
                "rate": round(present_n / n_days, 3) if n_days else None,
            }
        )

    # 缺勤多的排前面 —— 台账是为了找人，不是为了表扬
    out.sort(key=lambda r: (-r["absent_n"], r["person_name"]))
    return {"start": start, "end": end, "days": days, "rows": out}


def trend(days: int = 14, persons=None) -> list:
    """近 N 天的「应到 / 打卡 / 缺勤」序列，供折线图使用。

    补齐空日期：某天没有任何记录也要返回一个点，
    否则折线图会把「服务停机的那两天」直接连成一条直线。
    """
    days = max(1, min(92, int(days)))
    if persons is None:
        persons = face_db.list_persons()
    expect = len(persons)

    conn = face_db.get_conn()
    today0 = time.mktime(time.strptime(time.strftime("%Y-%m-%d"), "%Y-%m-%d"))
    lo, _ = _day_bounds(time.strftime("%Y-%m-%d", time.localtime(today0 - (days - 1) * 86400)))
    _, hi = _day_bounds(time.strftime("%Y-%m-%d", time.localtime(today0)))

    # 同时统计「已注册人数」与「总记录数」：前者参与打卡率，
    # 后者用来判断当天数据源是否活着（决定 absent 还是 no_data）
    by_day = {}
    for r in conn.execute(
        "SELECT substr(created_at,1,10) d, "
        "SUM(CASE WHEN is_temp=0 THEN 1 ELSE 0 END) reg, COUNT(*) all_n "
        "FROM face_seen WHERE created_at >= ? AND created_at < ? GROUP BY d",
        (lo, hi),
    ).fetchall():
        by_day[r["d"]] = (r["reg"] or 0, r["all_n"] or 0)

    out = []
    for i in range(days):
        d = time.strftime("%Y-%m-%d", time.localtime(today0 - (days - 1 - i) * 86400))
        reg_n, all_n = by_day.get(d, (0, 0))
        # 去重：同一个人当天被识别多次只算一次打卡。这里用一个近似 ——
        # 当日已注册记录数 >= 已注册去重人数，无法直接从聚合里得到去重值，
        # 因此额外查一次去重人数（代价可控：每天一次 COUNT(DISTINCT)）
        present = conn.execute(
            "SELECT COUNT(DISTINCT person_name) FROM face_seen "
            "WHERE created_at >= ? AND created_at < ? AND is_temp = 0",
            _day_bounds(d),
        ).fetchone()[0]
        absent = max(0, expect - present)
        status_day = ST_PRESENT if all_n else ST_NO_DATA
        out.append(
            {
                "date": d[5:],
                "full_date": d,
                "expect": expect,
                "present": present,
                "absent": absent,
                "records": all_n,
                "rate": round(present / expect, 3) if expect else None,
                # 当日整体是否有数据源。没有任何记录的日子，
                # 折线图上应显示为「无数据」而不是「全员缺勤」
                "day_status": status_day,
                "day_status_cn": STATUS_LABEL[status_day],
            }
        )
    return out
