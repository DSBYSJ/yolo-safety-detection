# -*- coding: utf-8 -*-
"""在线训练/评估管理：后台子进程执行 train/train.py 与 train/val.py，实时读取进度"""
import csv
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import config


def _subprocess_env() -> dict:
    """子进程环境：使用项目内独立的 matplotlib 缓存，避免多进程锁冲突"""
    env = dict(os.environ)
    cache = config.PROJECT_ROOT / ".mplcache"
    cache.mkdir(exist_ok=True)
    env["MPLCONFIGDIR"] = str(cache)
    return env

_lock = threading.Lock()

_train_proc = None
_train_meta: dict = {}
_eval_proc = None
_eval_meta: dict = {}


def _device() -> str:
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _tail(path: Path, n: int = 40) -> list:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return lines[-n:]
    except Exception:
        return []


# ---------------------------------------------------------------- 训练
def start(kind: str, size: str = "s", epochs: int = 80, imgsz: int = 640, batch: int = 16):
    global _train_proc
    with _lock:
        if _train_proc is not None and _train_proc.poll() is None:
            return None, "已有训练任务正在运行，请等待其结束后再启动"
        data = config.PROJECT_ROOT / "datasets" / kind / "data.yaml"
        if not data.exists():
            return None, f"未找到数据集配置：{data}（请先运行数据集构建脚本）"
        log_path = config.TRAIN_LOG_DIR / f"{kind}_{size}_{int(time.time())}.log"
        cmd = [
            sys.executable,
            str(config.PROJECT_ROOT / "train" / "train.py"),
            "--data", str(data),
            "--model", f"yolov8{size}.pt",
            "--epochs", str(epochs),
            "--imgsz", str(imgsz),
            "--batch", str(batch),
            "--name", f"{kind}_{size}",
            "--deploy-name", kind,
            "--device", _device(),
        ]
        log_f = open(log_path, "w", encoding="utf-8")
        _train_proc = subprocess.Popen(
            cmd, cwd=str(config.PROJECT_ROOT), stdout=log_f, stderr=subprocess.STDOUT,
            env=_subprocess_env(),
        )
        _train_meta.update(
            {
                "kind": kind,
                "size": size,
                "epochs": epochs,
                "imgsz": imgsz,
                "batch": batch,
                "log": str(log_path),
                "started": time.time(),
                "cmd": " ".join(cmd[1:]),
            }
        )
        return _train_proc.pid, None


def status() -> dict:
    with _lock:
        proc = _train_proc
        meta = dict(_train_meta)
    running = proc is not None and proc.poll() is None
    exit_code = proc.poll() if proc is not None else None
    run_name = f"{meta.get('kind', '')}_{meta.get('size', '')}"
    history, best = [], None
    csv_path = config.RUNS_DIR / run_name / "results.csv"
    if csv_path.exists():
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    def num(key):
                        try:
                            return float(r.get(key, "") or 0)
                        except ValueError:
                            return 0.0

                    row = {
                        "epoch": int(num("epoch") or 0) + 1,
                        "box_loss": round(num("train/box_loss"), 4),
                        "cls_loss": round(num("train/cls_loss"), 4),
                        "precision": round(num("metrics/precision(B)"), 4),
                        "recall": round(num("metrics/recall(B)"), 4),
                        "mAP50": round(num("metrics/mAP50(B)"), 4),
                        "mAP50_95": round(num("metrics/mAP50-95(B)"), 4),
                    }
                    history.append(row)
                    best = row["mAP50"] if best is None else max(best, row["mAP50"])
        except Exception:
            pass
    return {
        "running": running,
        "exit_code": exit_code,
        "meta": meta,
        "epoch_done": history[-1]["epoch"] if history else 0,
        "best_mAP50": best,
        "history": history[-80:],
        "log": _tail(Path(meta["log"])) if meta.get("log") else [],
    }


def runs_list() -> list:
    """列出历次训练运行（results.csv + 最佳指标）。"""
    out = []
    if not config.RUNS_DIR.exists():
        return out
    for d in sorted(config.RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        csv_path = d / "results.csv"
        if not csv_path.is_file():
            continue
        best, epochs = None, 0
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    try:
                        v = float(r.get("metrics/mAP50(B)", "") or 0)
                    except ValueError:
                        v = 0.0
                    best = v if best is None else max(best, v)
                    epochs += 1
        except Exception:
            continue
        weights = d / "weights" / "best.pt"
        out.append(
            {
                "name": d.name,
                "epochs": epochs,
                "best_mAP50": round(best, 4) if best else None,
                "weights": str(weights) if weights.exists() else None,
                "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(d.stat().st_mtime)),
            }
        )
    return out


# ---------------------------------------------------------------- 评估
def eval_start(kind: str, imgsz: int = 640, batch: int = 16):
    global _eval_proc
    with _lock:
        if _eval_proc is not None and _eval_proc.poll() is None:
            return None, "已有评估任务正在运行"
        data = config.PROJECT_ROOT / "datasets" / kind / "data.yaml"
        if not data.exists():
            return None, f"未找到数据集配置：{data}"
        try:
            weights = __import__("detector").model_path(kind)
        except Exception as e:
            return None, str(e)
        log_path = config.TRAIN_LOG_DIR / f"eval_{kind}_{int(time.time())}.log"
        cmd = [
            sys.executable,
            str(config.PROJECT_ROOT / "train" / "val.py"),
            "--weights", str(weights),
            "--data", str(data),
            "--imgsz", str(imgsz),
            "--batch", str(batch),
            "--name", kind,
            "--device", _device(),
        ]
        log_f = open(log_path, "w", encoding="utf-8")
        _eval_proc = subprocess.Popen(
            cmd, cwd=str(config.PROJECT_ROOT), stdout=log_f, stderr=subprocess.STDOUT,
            env=_subprocess_env(),
        )
        _eval_meta.update({"kind": kind, "log": str(log_path), "started": time.time()})
        return _eval_proc.pid, None


def eval_status() -> dict:
    with _lock:
        proc = _eval_proc
        meta = dict(_eval_meta)
    running = proc is not None and proc.poll() is None
    return {
        "running": running,
        "kind": meta.get("kind"),
        "log": _tail(Path(meta["log"])) if meta.get("log") else [],
    }


def eval_metrics(kind: str):
    """读取评估生成的 metrics.json 与可视化图片。"""
    import json

    base = config.RUNS_DIR / "val" / kind
    mj = base / "metrics.json"
    data = None
    if mj.exists():
        try:
            data = json.loads(mj.read_text(encoding="utf-8"))
        except Exception:
            data = None
    images = []
    for name in ("confusion_matrix_normalized.png", "confusion_matrix.png",
                 "BoxPR_curve.png", "results.png", "val_batch0_pred.jpg",
                 "val_batch1_pred.jpg", "val_batch2_pred.jpg"):
        p = base / name
        if p.exists():
            images.append({"name": name, "url": f"/runs/val/{kind}/{name}"})
    return {"metrics": data, "images": images}
