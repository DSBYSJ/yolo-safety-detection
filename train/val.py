# -*- coding: utf-8 -*-
"""YOLOv8 模型评估脚本

在验证集上输出 Precision / Recall / mAP@0.5 / mAP@0.5:0.95 及各类别指标，
同时生成混淆矩阵、PR 曲线等可视化图（保存在 runs/val/<name>/ 下）。

用法示例：
    python train/val.py --weights models/helmet.pt --data datasets/helmet/data.yaml --name helmet

对外契约（Web 端依赖，改动须保持兼容）：
    runs/val/<name>/metrics.json
        precision / recall / mAP50 / mAP50-95   : 全局汇总（仅统计验证集中出现过的类）
        per_class[]                             : {class_id, class_name, precision, recall,
                                                   mAP50, mAP50-95}
        weights / data                          : 本次评估使用的权重与数据集路径
    另附 eval_meta（新增，纯补充信息，不影响既有读取方）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 指标数组的长度下限：ap_class_index / p / r / ap50 / ap 必须等长
_METRIC_FIELDS = ("p", "r", "ap50", "ap")


def resolve(path: str) -> str:
    """相对路径以项目根目录为基准。"""
    p = Path(path)
    return str(p if p.is_absolute() else ROOT / p)


def _die(msg: str, code: int = 1) -> None:
    """统一错误出口：打印到 stdout（Web 端只捕获 stdout 日志）后非 0 退出。"""
    print(f"[ERROR] {msg}", flush=True)
    sys.exit(code)


def _as_list(v) -> list:
    """把可能为 numpy 数组 / 标量 / None 的指标字段安全转成 Python list。

    ⚠️ 不要写成 `getattr(obj, name, []) or []`：ultralytics 的指标字段是
    numpy 数组，对元素数 >1 的数组求布尔值会抛
        ValueError: The truth value of an array with more than one element is ambiguous
    （历史日志 eval_mask_1791458044.log 就是踩了这个坑）。
    这里统一走 np.ndarray 判断 + 显式 None 检查，不做真值运算。
    """
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    try:
        import numpy as np

        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):        # numpy 标量（np.float32 等）
            return [v.item()]
    except ImportError:
        pass
    # 普通标量（int/float/str）
    if isinstance(v, (int, float, str, bool)):
        return [v]
    try:
        return list(v)
    except TypeError:
        return [v]


def _finite(x, default: float = 0.0) -> float:
    """把 nan / inf / None 归一到有限浮点数。

    ultralytics 在「某类无正样本」等边界情形下会产出 nan；nan 经 json.dumps
    会写成非法的 `NaN` 字面量，浏览器 JSON.parse 直接抛 SyntaxError，
    表现为评估页整体空白。因此所有指标出口都必须过这道闸。
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def normalize_dataset_yaml(data_arg: str) -> Path:
    """把 data.yaml 规范化成一个「path 已写成绝对路径」的临时配置。

    背景（真实缺陷）：
        数据集配置里若写 `path: ./`，该值是 truthy 字符串，会绕过 ultralytics
        `data.get("path") or yaml_file.parent` 的兜底分支，被 `Path("./")` 按
        **当前工作目录** 解析。于是无论 CWD 是项目根还是 webapp，都会去找
        `<CWD>/images/val` 而必然 FileNotFoundError，评估彻底跑不起来。

    做法：
        把 `path` 改写为 data.yaml 所在目录的绝对路径，`train`/`val` 保持原样
        （相对 `path` 解释）。写出的临时文件放在原目录下并加 `.normalized` 后缀，
        不改动用户的原始 data.yaml。
    """
    src = Path(resolve(data_arg))
    if not src.is_file():
        _die(f"未找到数据集配置：{src}")

    try:
        import yaml
    except ImportError:
        # 没有 pyyaml 时退回「原样传递」，由 ultralytics 自行解析
        print(f"[WARN] 未安装 pyyaml，跳过 path 规范化，直接使用 {src}", flush=True)
        return src

    try:
        cfg = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    except Exception as e:
        _die(f"解析数据集配置失败：{src}（{type(e).__name__}: {e}）")
    if not isinstance(cfg, dict):
        _die(f"数据集配置格式异常（期望映射）：{src}")

    declared = cfg.get("path")
    base = src.parent
    # `path` 语义：相对则相对 data.yaml 所在目录；未声明则就是 data.yaml 所在目录
    if declared in (None, "", ".", "./"):
        root_dir = base
    else:
        p = Path(str(declared))
        root_dir = p if p.is_absolute() else (base / p)
    try:
        root_dir = root_dir.resolve()
    except OSError as e:
        _die(f"解析数据集根目录失败：{declared!r}（{e}）")

    for key in ("train", "val"):
        rel = cfg.get(key)
        if not rel:
            _die(f"数据集配置缺少 `{key}` 字段：{src}")
        # 允许 train/val 写成文件列表
        first = rel[0] if isinstance(rel, (list, tuple)) and rel else rel
        probe = Path(str(first))
        probe = probe if probe.is_absolute() else (root_dir / probe)
        if not probe.exists():
            _die(
                f"数据集 `{key}` 路径不存在：{probe}\n"
                f"        请确认 data.yaml 的 path/train/val 配置，或先运行数据集构建脚本。"
            )

    cfg["path"] = str(root_dir)
    # 后缀必须以 .yaml 结尾：ultralytics 的 YAML.load 有
    # `assert str(file).endswith((".yaml", ".yml"))` 的硬校验
    out = src.with_name(src.stem + ".normalized.yaml")
    try:
        out.write_text(
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
    except Exception as e:
        _die(f"写出规范化配置失败：{out}（{e}）")
    print(f"[data] 规范化数据集配置 -> {out}", flush=True)
    return out


def class_names(model) -> dict:
    """取 {id: name}，兼容 ultralytics 各版本 names 为 dict / list / None 的情形。

    原实现直接 `model.names.get(...)`：当 names 是 list（部分自定义权重会出现）
    时抛 AttributeError，整段 per_class 构造崩掉。
    """
    raw = getattr(model, "names", None)
    if isinstance(raw, dict):
        out = {}
        for k, v in raw.items():
            try:
                out[int(k)] = str(v)
            except (TypeError, ValueError):
                continue
        return out
    if isinstance(raw, (list, tuple)):
        return {i: str(v) for i, v in enumerate(raw)}
    return {}


def build_per_class(metrics, names: dict) -> list:
    """从 ultralytics 的 metrics.box 构造逐类指标。

    索引对齐说明（已实测核对）：
        ap_class_index 与 p / r / ap50 / ap 是**等长且同序**的，长度都等于
        「在验证集中出现过正样本的类别数」，因此用同一个 enumerate 索引是安全的。
        但仍做长度防御：任一类字段长度不足时跳过该类并告警，避免 IndexError。
    """
    box = metrics.box
    idx = _as_list(getattr(box, "ap_class_index", None))
    arrays = {f: _as_list(getattr(box, f, None)) for f in _METRIC_FIELDS}
    n = len(idx)
    usable = min([n] + [len(a) for a in arrays.values()])
    if usable < n:
        print(
            f"[WARN] 指标数组长度不一致（ap_class_index={n}，"
            + ", ".join(f"{k}={len(v)}" for k, v in arrays.items())
            + f"），仅取前 {usable} 类",
            flush=True,
        )

    per_class = []
    for i in range(usable):
        try:
            cid = int(idx[i])
        except (TypeError, ValueError):
            continue
        per_class.append(
            {
                "class_id": cid,
                "class_name": names.get(cid, str(cid)),
                "precision": _finite(arrays["p"][i]),
                "recall": _finite(arrays["r"][i]),
                "mAP50": _finite(arrays["ap50"][i]),
                "mAP50-95": _finite(arrays["ap"][i]),
            }
        )
    return per_class


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8 评估脚本")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="val")
    args = parser.parse_args()

    # ---- 输入校验：参数合法性与文件存在性 ----
    if args.imgsz <= 0:
        _die(f"--imgsz 必须为正整数，收到 {args.imgsz}")
    if args.batch == 0 or args.batch < -1:
        _die(f"--batch 必须为正整数或 -1，收到 {args.batch}")
    if not args.name or not args.name.strip():
        _die("--name 不能为空")
    # 防目录穿越：name 会直接拼进输出路径
    if any(sep in args.name for sep in ("/", "\\", "..", ":")):
        _die(f"--name 含非法字符（不得包含路径分隔符）：{args.name!r}")

    weights = Path(resolve(args.weights))
    if not weights.is_file():
        _die(f"未找到权重文件：{weights}")

    data_yaml = normalize_dataset_yaml(args.data)

    out_dir = ROOT / "runs" / "val" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 执行评估 ----
    # 注意：不在此处 try 掉 KeyboardInterrupt，让 Ctrl+C 能正常中断
    try:
        from ultralytics import YOLO
    except ImportError as e:
        _die(f"未安装 ultralytics，无法评估（{e}）")

    t0 = time.time()
    try:
        model = YOLO(str(weights))
        metrics = model.val(
            data=str(data_yaml),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(ROOT / "runs" / "val"),
            name=args.name,
            plots=True,
            exist_ok=True,
            verbose=True,
        )
    except FileNotFoundError as e:
        _die(f"数据集路径无效：{e}")
    except RuntimeError as e:
        # 显存不足、设备不可用等
        _die(f"评估执行失败（运行时错误）：{e}")
    except Exception as e:
        print("[ERROR] 评估执行失败：", type(e).__name__, e, flush=True)
        traceback.print_exc()
        sys.exit(1)
    elapsed = round(time.time() - t0, 2)

    if metrics is None or getattr(metrics, "box", None) is None:
        _die("评估未产出 Box 指标（可能是任务类型不匹配，期望 detect 任务）")

    names = class_names(model)
    per_class = build_per_class(metrics, names)
    box = metrics.box

    summary = {
        # ---- 以下字段为对外契约，语义保持不变 ----
        "weights": args.weights,
        "data": args.data,
        "precision": _finite(getattr(box, "mp", 0.0)),
        "recall": _finite(getattr(box, "mr", 0.0)),
        "mAP50": _finite(getattr(box, "map50", 0.0)),
        "mAP50-95": _finite(getattr(box, "map", 0.0)),
        "per_class": per_class,
        # ---- 以下为新增补充信息，不影响既有读取方 ----
        "eval_meta": {
            "weights_resolved": str(weights),
            "data_resolved": str(Path(resolve(args.data))),
            "imgsz": int(args.imgsz),
            "batch": int(args.batch),
            "device": str(args.device),
            "classes_evaluated": len(per_class),
            "elapsed_sec": elapsed,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }

    if not per_class:
        print(
            "[WARN] 没有任何类别被评估：验证集中可能不存在标注目标。"
            "汇总指标将全部为 0。",
            flush=True,
        )

    out_json = out_dir / "metrics.json"
    try:
        # allow_nan=False：宁可在这行报错，也不产出浏览器读不了的非法 JSON
        out_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    except ValueError as e:
        _die(f"指标含非有限数值，无法写出合法 JSON：{e}")

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[saved] {out_json}", flush=True)


if __name__ == "__main__":
    main()
