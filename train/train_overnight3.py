# -*- coding: utf-8 -*-
"""夜间训练驱动 v3 —— 训练口罩(mask)与 PPE 两个模型

与 train/train.py 业务逻辑一致；仅负责：输出到临时目录后回拷 runs/、
按 mAP 比较决定是否替换 models/ 下现役权重。
"""
import json
import shutil
import time
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
TMP_PROJECT = Path(r"C:/Users/20261/AppData/Local/Temp/yolo_overnight_runs")

PLAN = [
    ("mask", "datasets/mask/data_gpu.yaml", "pretrained/yolov8s.pt", 80, 640, 16),
    ("ppe",  "datasets/ppe/data.yaml",     "pretrained/yolov8s.pt", 50, 640, 16),
]

SUMMARY: dict = {}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def copy_back(src: Path, name: str) -> Path:
    dst = ROOT / "runs" / name
    if dst.exists():
        dst = ROOT / "runs" / f"{name}_{time.strftime('%m%d%H%M')}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    return dst


def read_best_map(run: Path) -> float:
    csv = run / "results.csv"
    if not csv.exists():
        return -1.0
    lines = csv.read_text(encoding="utf-8").strip().splitlines()
    head = [h.strip() for h in lines[0].split(",")]
    idx = next((i for i, h in enumerate(head) if h.startswith("metrics/mAP50-95")), None)
    if idx is None:
        return -1.0
    vals = []
    for ln in lines[1:]:
        p = ln.split(",")
        if len(p) > idx:
            try:
                vals.append(float(p[idx]))
            except ValueError:
                pass
    return max(vals) if vals else -1.0


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
    TMP_PROJECT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = YOLO(str(ROOT / weights))
    model.train(
        data=str(data_path), epochs=epochs, imgsz=imgsz, batch=batch,
        device=0, workers=4, project=str(TMP_PROJECT), name=name,
        patience=15, plots=True, seed=42, exist_ok=True, save=True,
        save_period=10, val=True,
    )
    dur = time.time() - t0
    tmp_run = TMP_PROJECT / name
    best_map = read_best_map(tmp_run)
    dst = copy_back(tmp_run, name)
    log(f"{name} 完成：{dur/60:.1f} 分钟，最佳 mAP50-95={best_map:.4f} -> {dst}")

    cur = ROOT / "models" / f"{name}.pt"
    old_map = evaluate_current(cur, data_yaml) if cur.exists() else -1.0
    best_pt = dst / "weights" / "best.pt"
    deployed = False
    if best_pt.exists() and best_map >= old_map:
        shutil.copy2(best_pt, cur)
        deployed = True
        log(f"已部署 {name}: 新 {best_map:.4f} >= 旧 {old_map:.4f}")
    else:
        log(f"未替换 {name}: 新 {best_map:.4f} < 旧 {old_map:.4f}（新权重在 {best_pt}）")
    SUMMARY[name] = {
        "minutes": round(dur / 60, 1), "best_map50_95": round(best_map, 4),
        "old_map50_95": round(old_map, 4), "deployed": deployed, "run_dir": str(dst),
    }
    return dst


def main() -> None:
    t0 = time.time()
    for item in PLAN:
        try:
            train_one(*item)
        except Exception as e:
            log(f"{item[0]} 训练异常: {type(e).__name__}: {e}")
            SUMMARY[item[0]] = {"error": f"{type(e).__name__}: {e}"}
    log(f"===== 全部结束，总耗时 {(time.time()-t0)/60:.1f} 分钟 =====")
    out = ROOT / "runs" / "overnight_summary_mask_ppe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(SUMMARY, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(SUMMARY, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
