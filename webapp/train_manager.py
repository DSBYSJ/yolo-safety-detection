# -*- coding: utf-8 -*-
"""在线训练/评估管理：后台子进程执行 train/train.py 与 train/val.py，实时读取进度"""
import csv
import json
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
_paused = False          # 训练是否处于暂停（进程被挂起）状态
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
    if kind not in config.MODELS:
        return None, f"未知的模型类型：{kind}"
    with _lock:
        if _train_proc is not None and _train_proc.poll() is None:
            return None, "已有训练任务正在运行，请等待其结束后再启动"
        # 评估正在跑时让出显卡，避免训练与评估互相抢显存
        if _eval_proc is not None and _eval_proc.poll() is None:
            return None, "评估任务正在运行，请等评估结束后再开始训练"
        data = config.PROJECT_ROOT / "datasets" / kind / "data.yaml"
        if not data.exists():
            return None, f"未找到数据集配置：{data}（请先运行数据集构建脚本）"
        if not isinstance(epochs, int) or epochs <= 0:
            return None, f"epochs 必须为正整数，收到 {epochs!r}"
        if not isinstance(imgsz, int) or imgsz <= 0:
            return None, f"imgsz 必须为正整数，收到 {imgsz!r}"
        if not isinstance(batch, int) or batch == 0 or batch < -1:
            return None, f"batch 必须为正整数或 -1，收到 {batch!r}"
        config.TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = config.TRAIN_LOG_DIR / f"{kind}_{size}_{int(time.time())}.log"
        # 优先用项目内已有权重，避免 ultralytics 联网下载（国内常超时/失败）
        local_w = config.PRETRAINED_DIR / f"yolov8{size}.pt"
        model_arg = str(local_w) if local_w.exists() else f"yolov8{size}.pt"
        cmd = [
            sys.executable,
            str(config.PROJECT_ROOT / "train" / "train.py"),
            "--data", str(data),
            "--model", model_arg,
            "--epochs", str(epochs),
            "--imgsz", str(imgsz),
            "--batch", str(batch),
            "--name", f"{kind}_{size}",
            "--deploy-name", kind,
            "--device", _device(),
        ]
        log_f = open(log_path, "w", encoding="utf-8")
        try:
            _train_proc = subprocess.Popen(
                cmd, cwd=str(config.PROJECT_ROOT), stdout=log_f, stderr=subprocess.STDOUT,
                env=_subprocess_env(),
            )
        except Exception as e:
            log_f.close()
            _train_proc = None
            return None, f"启动训练进程失败：{e}"
        else:
            # 子进程已继承句柄，父进程这端可以关了：原实现从不关闭，
            # 每启动一次训练就泄漏一个文件描述符
            try:
                log_f.close()
            except Exception:
                pass
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


def _iter_tree(pid: int):
    """返回进程树（自身 + 递归子进程）；psutil 缺失或进程不存在时返回 None。

    训练进程会派生子进程做数据加载，暂停时要把整棵树一起挂起，
    否则父进程停住、子进程仍在空转，效果不干净。
    """
    try:
        import psutil
    except ImportError:
        return None
    try:
        root = psutil.Process(pid)
    except Exception:
        return None
    try:
        return [root] + root.children(recursive=True)
    except Exception:
        return [root]


def pause():
    """暂停训练：挂起训练进程树。进度、优化器状态、显存占用原样保留。

    注意：挂起期间显存不会释放（CUDA 上下文还在），要腾显卡请用 stop()。
    """
    global _paused
    with _lock:
        proc = _train_proc
    if proc is None or proc.poll() is not None:
        return False, "当前没有正在运行的训练任务"
    if _paused:
        return True, None
    procs = _iter_tree(proc.pid)
    if procs is None:
        return False, "缺少 psutil，无法暂停（请先执行 pip install psutil）"
    n = 0
    for p in procs:            # 先挂父进程，避免它在挂起子进程期间继续派发任务
        try:
            p.suspend()
            n += 1
        except Exception:
            pass
    if not n:
        return False, "暂停失败：训练进程可能已结束"
    _paused = True
    return True, None


def resume():
    """继续训练：恢复被挂起的训练进程树。"""
    global _paused
    with _lock:
        proc = _train_proc
    if proc is None or proc.poll() is not None:
        return False, "当前没有正在运行的训练任务"
    procs = _iter_tree(proc.pid)
    if procs is None:
        return False, "缺少 psutil，无法继续（请先执行 pip install psutil）"
    for p in reversed(procs):  # 先恢复子进程再恢复父进程
        try:
            p.resume()
        except Exception:
            pass
    _paused = False
    return True, None


def stop():
    """停止训练：先恢复（避免挂起状态无法结束），再结束整棵进程树。

    已完成的轮次已经写进 runs/<name>/results.csv 与 weights/last.pt，
    停止不会丢失这些产物；如需接着练，可用 last.pt 作为起点续训。
    """
    global _paused
    with _lock:
        proc = _train_proc
    if proc is None or proc.poll() is not None:
        return False, "当前没有正在运行的训练任务"
    procs = _iter_tree(proc.pid) or []
    if _paused:
        for p in reversed(procs):
            try:
                p.resume()
            except Exception:
                pass
        _paused = False
    for p in reversed(procs):
        try:
            p.terminate()
        except Exception:
            pass
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=15)
    except Exception:
        for p in procs:
            try:
                p.kill()
            except Exception:
                pass
        try:
            proc.kill()
        except Exception:
            pass
    return True, None


def _refresh_paused(proc) -> bool:
    """按进程真实状态刷新暂停标志（psutil 不可用时沿用内存里的标志）。"""
    global _paused
    if proc is None or proc.poll() is not None:
        _paused = False
        return False
    try:
        import psutil
    except ImportError:
        return _paused
    procs = _iter_tree(proc.pid)
    if not procs:
        return _paused
    try:
        stopped = sum(1 for p in procs if p.status() == psutil.STATUS_STOPPED)
    except Exception:
        return _paused
    _paused = stopped > 0
    return _paused


def status() -> dict:
    with _lock:
        proc = _train_proc
        meta = dict(_train_meta)
    running = proc is not None and proc.poll() is None
    exit_code = proc.poll() if proc is not None else None
    paused = _refresh_paused(proc) if running else False
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
        "paused": paused,
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
def _eval_state(proc, meta: dict) -> dict:
    """由「进程对象 + 元信息」推导评估任务状态（调用方须已持有 _lock 或已快照）。

    状态语义（前端据此区分提示文案）：
        idle    从未启动过评估（meta 为空）
        running 子进程仍在运行
        done    子进程正常退出（returncode == 0）
        failed  子进程异常退出（returncode != 0），附 exit_code
    """
    if not meta:
        running, exit_code = False, None
    else:
        running = proc is not None and proc.poll() is None
        exit_code = None if running or proc is None else proc.poll()
    if running:
        state = "running"
    elif not meta:
        state = "idle"
    elif exit_code == 0:
        state = "done"
    else:
        state = "failed"
    return {"running": running, "exit_code": exit_code, "state": state}


def eval_start(kind: str, imgsz: int = 640, batch: int = 16):
    """启动一次验证集评估。返回 (pid, error)；失败时 pid 为 None。

    与训练互斥：同一张显卡上并行跑训练与评估会互相抢占显存，
    因此只要训练在跑就拒绝启动评估（反之亦然，见 start()）。
    """
    global _eval_proc
    if kind not in config.MODELS:
        return None, f"未知的模型类型：{kind}"
    if not isinstance(imgsz, int) or imgsz <= 0:
        return None, f"--imgsz 必须为正整数，收到 {imgsz!r}"
    if not isinstance(batch, int) or batch == 0 or batch < -1:
        return None, f"--batch 必须为正整数或 -1，收到 {batch!r}"

    with _lock:
        # 已有评估在跑
        if _eval_proc is not None and _eval_proc.poll() is None:
            return None, "已有评估任务正在运行，请等待其结束后再启动"
        # 训练正在跑 —— 让出显卡
        if _train_proc is not None and _train_proc.poll() is None:
            return None, "训练任务正在运行，请等训练结束后再评估"

        data = config.PROJECT_ROOT / "datasets" / kind / "data.yaml"
        if not data.exists():
            return None, f"未找到数据集配置：{data}（请先运行数据集构建脚本）"
        try:
            weights = __import__("detector").model_path(kind)
        except Exception as e:
            return None, str(e)
        # 权重文件必须真实存在，否则子进程会白跑一趟
        try:
            if not Path(weights).is_file():
                return None, f"权重文件不存在：{weights}"
        except Exception:
            return None, f"权重路径无效：{weights}"

        config.TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)
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
        try:
            log_f = open(log_path, "w", encoding="utf-8")
        except OSError as e:
            return None, f"无法创建评估日志文件：{log_path}（{e}）"
        try:
            _eval_proc = subprocess.Popen(
                cmd, cwd=str(config.PROJECT_ROOT), stdout=log_f, stderr=subprocess.STDOUT,
                env=_subprocess_env(),
            )
        except Exception as e:
            log_f.close()
            _eval_proc = None
            return None, f"启动评估进程失败：{e}"
        finally:
            # 父进程不再需要写端：子进程已继承句柄，关掉可避免文件描述符泄漏
            # （原实现在每次评估后都泄漏一个 fd，长期运行会耗尽）
            try:
                log_f.close()
            except Exception:
                pass

        _eval_meta.clear()
        _eval_meta.update({
            "kind": kind,
            "log": str(log_path),
            "started": time.time(),
            "imgsz": imgsz,
            "batch": batch,
            "weights": str(weights),
        })
        return _eval_proc.pid, None


def eval_status() -> dict:
    with _lock:
        proc = _eval_proc
        meta = dict(_eval_meta)
    st = _eval_state(proc, meta)
    elapsed = None
    if meta.get("started"):
        end = time.time() if st["running"] else None
        if end is None:
            # 已结束：尽量用日志文件的落盘时间近似结束时刻
            try:
                end = Path(meta["log"]).stat().st_mtime
            except Exception:
                end = time.time()
        elapsed = round(end - meta["started"], 1)
    return {
        "running": st["running"],
        "state": st["state"],
        "exit_code": st["exit_code"],
        "kind": meta.get("kind"),
        "elapsed_sec": elapsed,
        "log": _tail(Path(meta["log"])) if meta.get("log") else [],
    }


# 评估产出的可视化白名单（顺序即前端展示顺序）。
# 注意：ultralytics 还会生成 val_batchN_labels.jpg（标注对照图），
# 但对使用者价值低且体积大，此处在白名单外，保持既有行为不变。
EVAL_IMAGE_NAMES = (
    "confusion_matrix_normalized.png",
    "confusion_matrix.png",
    "BoxPR_curve.png",
    "BoxF1_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
    "results.png",
    "val_batch0_pred.jpg",
    "val_batch1_pred.jpg",
    "val_batch2_pred.jpg",
)


def eval_metrics(kind: str, strict: bool = False):
    """读取评估生成的 metrics.json 与可视化图片。

    strict=True 时，若指标文件缺失或损坏则抛 ValueError（供调试/API 需要明确
    报错的场景）；默认 strict=False 保持原有「读不到就返回 None」的宽松语义，
    但会把失败原因以 `error` 字段一并返回，不再静默吞掉。

    返回值（对外契约）：
        metrics : dict | None   —— 与旧版完全一致的指标结构
        images  : list          —— [{name, url}]，旧版已有
        error   : str | None    —— 新增，说明 metrics 不可用的原因
        stale   : bool          —— 新增，图片比指标新（上次评估失败后残留的旧指标）
    """
    base = config.RUNS_DIR / "val" / kind
    mj = base / "metrics.json"
    data = None
    error = None
    if mj.exists():
        try:
            data = json.loads(mj.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                error = f"metrics.json 顶层不是对象：{type(data).__name__}"
                data = None
            elif "per_class" in data and not isinstance(data["per_class"], list):
                error = "metrics.json 的 per_class 不是列表"
                data = None
        except json.JSONDecodeError as e:
            error = f"metrics.json 解析失败（JSON 非法）：{e}"
        except OSError as e:
            error = f"metrics.json 读取失败：{e}"
    else:
        error = "尚未生成评估结果"

    if strict and data is None:
        raise ValueError(error or "评估指标不可用")

    images = []
    newest_img = 0.0
    for name in EVAL_IMAGE_NAMES:
        p = base / name
        try:
            if not p.exists():
                continue
            newest_img = max(newest_img, p.stat().st_mtime)
        except OSError:
            continue
        images.append({"name": name, "url": f"/runs/val/{kind}/{name}"})

    # 一致性检测：若可视化比 metrics.json 新出一截，说明最近一次评估
    # 在「图已落盘、指标未写出」的中间态失败（本次修复前的常见现象），
    # 此时页面上看到的是旧指标 + 新图片，需要显式告知而不是让用户误解。
    stale = False
    if data is not None and newest_img:
        try:
            stale = newest_img - mj.stat().st_mtime > 60
        except OSError:
            stale = False
    return {"metrics": data, "images": images, "error": error, "stale": stale}
