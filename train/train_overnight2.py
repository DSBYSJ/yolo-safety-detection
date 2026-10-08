# -*- coding: utf-8 -*-
"""夜间无人值守训练驱动 v2（GPU）—— 按 8 小时预算调整后的 epoch 数

与 train/train.py 业务一致（ultralytics YOLO 训练 + best/last 检查点），
额外负责：输出到临时目录后回拷、依次训练多个模型、按指标决定是否替换现役权重。
"""
import argparse
import json
import shutil
import time
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
TMP_PROJECT = Path(r"C:/Users/20261/AppData/Local/Temp/yolo_overnight_runs")

# (名称, data.yaml, 预训练权重, epochs, imgsz, batch)
PLAN = [
    ("helmet", "datasets/helmet/data.yaml", "pretrained/yolov8s.pt", 35, 640, 16),
    ("mask",   "datasets/mask/data.yaml",   "pretrained/yolov8s.pt", 80, 640, 16),
    ("ppe",    "datasets/ppe/data.yaml",    "pretrained/yolov8s.pt", 50, 640, 16),
]

SUMMARY: dict = {}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def copy_back(src_dir: Path, name: str) -> Path:
    dst = ROOT / "runs" / name
    if dst.exists():
        dst = ROOT / "runs" / f"{name}_{time.strftime('%m%d%H%M')}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_dir, dst)
    return dst


def read_best_map(run_dir: Path) -> float:
    csv = run_dir / "results.csv"
    if not csv.exists():
        return -1.0
    try:
        lines = csv.read_text(encoding="utf-8").strip().splitlines()
        header = [h.strip() for h in lines[0].split(",")]
        idx = next((i for i, h in enumerate(header) if h.startswith("metrics/mAP50-95")), None)
        if idx is None:
            return -1.0
        vals = []
        for ln in lines[1:]:
            parts = ln.split(",")
            if len(parts) > idx:
                try:
                    vals.append(float(parts[idx]))
                except ValueError:
                    pass
        return max(vals) if vals else -1.0
    except OSError:
        return -1.0


def evaluate_current(model_path: Path, data: str) -> float:
    if not model_path.exists():
        return -1.0
    try:
        m = YOLO(str(model_path))
        r = m.val(data=str(ROOT / data), imgsz=640, split="val", verbose=False)
        return float(r.results_dict.get("metrics/mAP50-95(M)", -1.0))
    except Exception as e:
        log(f"  评估旧权重失败: {e}")
        return -1.0


def train_one(name, data_yaml, weights, epochs, imgsz, batch):
    log(f"===== 开始训练 {name} (epochs={epochs}) =====")
    data_path = ROOT / data_yaml
    if not data_path.exists():
        log(f"跳过 {name}: 缺少 {data_yaml}")
        SUMMARY[name] = {"error": f"缺少 {data_yaml}"}
        return None
    wpath = ROOT / weights
    if not wpath.exists():
        wpath = Path(weights)

    TMP_PROJECT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = YOLO(str(wpath))
    model.train(
        data=str(data_path),
        epochs=epochs, imgsz=imgsz, batch=batch, device=0, workers=4,
        project=str(TMP_PROJECT), name=name, patience=15, plots=True,
        seed=42, exist_ok=True, save=True, save_period=10, val=True,
    )
    dur = time.time() - t0

    tmp_run = TMP_PROJECT / name
    best_map = read_best_map(tmp_run)
    dst = copy_back(tmp_run, name)
    log(f"{name} 完成：{dur/60:.1f} 分钟，最佳 mAP50-95={best_map:.4f}，结果 -> {dst}")

    cur = ROOT / "models" / f"{name}.pt"
    old_map = evaluate_current(cur, data_yaml) if cur.exists() else -1.0
    best_pt = dst / "weights" / "best.pt"
    deployed = False
    if best_pt.exists() and best_map >= old_map:
        shutil.copy2(best_pt, cur)
        deployed = True
        log(f"已部署 {name}: 新 {best_map:.4f} >= 旧 {old_map:.4f}")
    else:
        log(f"未替换 {name}: 新 {best_map:.4f} < 旧 {old_map:.4f}（新权重保留在 {best_pt}）")

    SUMMARY[name] = {
        "minutes": round(dur / 60, 1),
        "best_map50_95": round(best_map, 4),
        "old_map50_95": round(old_map, 4),
        "deployed": deployed,
        "run_dir": str(dst),
    }
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    want = [x.strip() for x in args.only.split(",") if x.strip()]
    t0 = time.time()
    for item in PLAN:
        if want and item[0] not in want:
            continue
        try:
            train_one(*item)
        except Exception as e:
            log(f"{item[0]} 训练异常: {type(e).__name__}: {e}")
            SUMMARY[item[0]] = {"error": f"{type(e).__name__}: {e}"}
    log(f"===== 全部结束，总耗时 {(time.time()-t0)/60:.1f} 分钟 =====")
    out = ROOT / "runs" / "overnight_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(SUMMARY, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(SUMMARY, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
