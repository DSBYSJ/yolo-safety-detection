# -*- coding: utf-8 -*-
"""装备归属：把「检测到的安全帽/口罩」归到「检测到的某个人」头上

要解决的问题
------------
原来人脸识别和装备检测是**两条互不相干的线**：
    - `face_db` 只知道「画面里有张三」
    - `detector` 只知道「画面里有 1 个安全帽、1 个未戴口罩」

于是「张三没戴安全帽」这件事根本无法回答 —— 只能给出
「本帧 2 人 / 1 个未戴安全帽」这种无法落到人头上的统计。
合规统计因此只能按「次」聚合，不能按「人」聚合。

归属规则（为什么是空间重叠）
----------------------------
安全帽、口罩与人脸都是**同一张脸上的装备**，位置强相关：

    ┌──────────┐  ← 安全帽框：在人脸上方，水平方向大致对齐
    │  安全帽  │
    ├──────────┤  ← 人脸框
    │ 口│ 罩   │  ← 口罩框：覆盖人脸下半部
    └──────────┘

所以用 IoU（交并比）判定归属是最可靠的：
    - 对每个装备框，找与它 IoU 最大的人脸框
    - IoU 超过阈值才算「这个装备属于这个人」
    - 安全帽在人脸**上方**、重叠面积可能不大，因此额外放行一种情况：
      水平方向覆盖人脸中点且纵向贴近（帽沿压在人脸上沿附近）

为什么不用「最近的框」或「画面里人数==装备数就一一对应」：
    - 最近邻在多人的画面里会把隔壁人的帽子算过来
    - 数量相等是巧合，不是规则（3 人 2 帽时无法判断是谁没戴，
      而「谁没戴」恰恰是这个功能唯一的价值所在）

安全降级
--------
- 归属失败（人脸框缺失、装备框与人脸完全不相干）→ 记为 `unknown`，
  不硬塞给某个人。**宁可漏报，不可错报** ——
  把「李四没戴」记到「张三」头上，比不记录更糟。
- 一个人可以同时有多件装备，但每类只取最优的一个，
  避免同一顶帽子被算两次。
"""
import numpy as np

# IoU 阈值：人脸框与装备框的交并比超过此值才认为属于同一个人。
# 取 0.10 偏低 —— 安全帽框比人脸框大一圈、且偏上，
# 典型 IoU 只有 0.1~0.3；取太高会导致「明明戴了却判没戴」。
IOU_THRESHOLD = 0.10

# 安全帽专用：帽框上沿压在人脸框上沿附近时也算（纵向不重叠的情况）
# 容差 = 人脸框高度的多少倍。0.6 表示帽底可以在人脸中部以上。
HAT_TOP_TOLERANCE = 0.6

# 帽子离人脸能有多远（按人脸高度的倍数算）。
# ⚠️ 这个上界是必须的：只判断「帽子在人脸上方、水平对得齐」的话，
# 画面顶上任何一顶帽子都会被配给画面下方任意一个人 ——
# 实测过一个离人脸 290px 的帽子被错误配对。安全帽只能紧贴头顶，
# 因此限定帽底不得高于人脸的 1.5 倍脸高之上。
HAT_MAX_GAP_RATIO = 1.5

# 各类装备的「合规」判定：框类别 -> 该类别代表合规
# helmet=戴了安全帽(mask=戴了口罩) 合规；head/face=未戴 违规
EQUIP_COMPLIANT = {"helmet": True, "mask": True, "head": False, "face": False}

# 每类装备的归属分组：安全帽与未戴安全帽互斥，口罩与未戴口罩互斥
EQUIP_GROUPS = {
    "hat": ("helmet", "head"),      # 头部装备（戴帽 / 未戴帽）
    "mask": ("mask", "face"),       # 面部装备（戴口罩 / 未戴）
}


def iou(a, b) -> float:
    """两个框的交并比。任意一个框无效时返回 0。"""
    try:
        ax1, ay1, ax2, ay2 = (float(v) for v in a[:4])
        bx1, by1, bx2, by2 = (float(v) for v in b[:4])
    except (TypeError, ValueError, IndexError):
        return 0.0
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _hat_overlaps_face(face_box, equip_box) -> bool:
    """安全帽框「骑」在人脸上方时的宽松判定。

    安全帽往往比人脸框大、且整体偏上，两者 IoU 可能很低（甚至为 0），
    但显然属于同一个人。判定条件：
        - 水平方向：帽框与脸框有交集，且帽框水平中点落在脸框水平范围内
        - 纵向：帽框下沿在人脸中部以上（即帽子在脸的「头顶以上到中部」）
        - 纵向：帽框下沿不能离人脸太远（不得超过 HAT_MAX_GAP_RATIO 倍脸高）

    最后那条上界必不可少：只判断「在上方 + 水平对齐」的话，
    画面顶部的帽子会被配给任何一个人 —— 帽子必须紧贴头顶。
    """
    try:
        fx1, fy1, fx2, fy2 = (float(v) for v in face_box[:4])
        ex1, ey1, ex2, ey2 = (float(v) for v in equip_box[:4])
    except (TypeError, ValueError, IndexError):
        return False
    fh = fy2 - fy1
    if fh <= 0:
        return False
    # 水平：帽框中心必须落在脸框水平范围内（略放宽 10%）
    pad = (fx2 - fx1) * 0.1
    ecx = (ex1 + ex2) / 2.0
    if not (fx1 - pad <= ecx <= fx2 + pad):
        return False
    # 有水平重叠
    if min(fx2, ex2) - max(fx1, ex1) <= 0:
        return False
    # 纵向：帽框下沿必须在人脸中部以上
    if ey2 > fy1 + fh * HAT_TOP_TOLERANCE:
        return False
    # 纵向：帽框下沿又不能离人脸顶端太远
    if ey2 < fy1 - fh * HAT_MAX_GAP_RATIO:
        return False
    return True


def _equip_key(det: dict) -> str:
    """把检测框映射到装备分组：hat / mask，其余返回空串。"""
    cls = str(det.get("class") or "")
    for key, members in EQUIP_GROUPS.items():
        if cls in members:
            return key
    return ""


def attach(detections: list, faces: list) -> list:
    """把装备检测框归属到人脸上。

    参数
    ----
    detections : [{class, conf, box, ...}]  装备检测框（像素坐标）
    faces      : [{name, is_temp, person_id, score, box}] 人脸框（同一坐标系）

    返回
    ----
    faces 的**副本**，每项新增：
        equip: {"hat": {"class": "helmet", "conf": 0.87, "compliant": True} | None,
                "mask": {...} | None}
        missing: ["hat"] / ["mask"] 等，未佩戴的装备分组
        compliant: bool，所有应检装备都合规（没有检测到装备时视为 None）

    只有人脸框时才会返回归属结果；`faces` 为空时返回空列表 ——
    「谁没戴」必须落到具体的人身上才有意义。
    """
    if not faces:
        return []

    out = []
    # 每件装备只能用一次：一顶帽子不能同时算给两个人
    used = set()

    for face in faces:
        item = dict(face)
        fbox = item.get("box")
        equip = {"hat": None, "mask": None}

        for gkey in EQUIP_GROUPS:
            best_i, best_score = -1, 0.0
            for i, det in enumerate(detections):
                if i in used or _equip_key(det) != gkey:
                    continue
                ebox = det.get("box")
                score = iou(fbox, ebox)
                # 安全帽额外允许「骑在头顶」的宽松命中
                if gkey == "hat" and score < IOU_THRESHOLD:
                    if _hat_overlaps_face(fbox, ebox):
                        score = max(score, IOU_THRESHOLD)
                if score >= IOU_THRESHOLD and score > best_score:
                    best_i, best_score = i, score
            if best_i >= 0:
                det = detections[best_i]
                used.add(best_i)
                equip[gkey] = {
                    "class": det.get("class"),
                    "conf": det.get("conf"),
                    "compliant": bool(EQUIP_COMPLIANT.get(str(det.get("class")), False)),
                    "iou": round(best_score, 3),
                }

        item["equip"] = equip
        # 只把「明确检测到未佩戴」记进 missing。
        # 该组完全没有检测框时是「无信息」，不能当作未佩戴上报，
        # 否则摄像头没对准、模型漏检都会被误判成员工违规。
        missing = [g for g, v in equip.items()
                   if v is not None and not v["compliant"]]
        item["missing"] = missing
        known = [v for v in equip.values() if v is not None]
        item["compliant"] = (all(v["compliant"] for v in known)
                             if known else None)
        out.append(item)

    return out


def summarize(faces_with_equip: list) -> dict:
    """按人汇总佩戴情况，供页面直接展示。"""
    n = len(faces_with_equip)
    res = {
        "persons": n,
        "hat_ok": 0, "hat_bad": 0, "hat_unknown": 0,
        "mask_ok": 0, "mask_bad": 0, "mask_unknown": 0,
        "fully_compliant": 0, "violators": 0,
        "missing_list": [],          # [{name, missing}] 供前端逐条列出
    }
    for f in faces_with_equip:
        equip = f.get("equip") or {}
        missing = f.get("missing") or []
        for key, ok_k, bad_k in (("hat", "hat_ok", "hat_bad"),
                                 ("mask", "mask_ok", "mask_bad")):
            v = equip.get(key)
            if v is None:
                res[key + "_unknown"] += 1
            elif v.get("compliant"):
                res[ok_k] += 1
            else:
                res[bad_k] += 1
        if f.get("compliant") is True:
            res["fully_compliant"] += 1
        if missing:
            res["violators"] += 1
            res["missing_list"].append({
                "name": f.get("name"),
                "is_temp": bool(f.get("is_temp")),
                "missing": missing,
                "missing_cn": ["安全帽" if m == "hat" else "口罩" for m in missing],
            })
    return res
