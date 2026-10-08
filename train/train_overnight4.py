# -*- coding: utf-8 -*-
"""夜间训练驱动 v4 —— 安全帽第二阶段追加训练

用第一阶段的最优权重 runs/helmet/weights/best.pt 作为起点继续训练，
冲击更高 mAP；结束后同样回拷结果并按指标决定是否替换现役权重。
"""
import json
import shutil
import time
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
TMP_PROJECT = Path(r"C:/Users/20261/AppData/Local/Temp/yolo_overnight_runs")
STAGE1_BEST = ROOT / "runs" / "helmet" / "weights" / "best.pt"

EPOCHS = 30


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


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


def main() -> None:
    log(f"===== 安全帽第二阶段训练 (epochs={EPOCHS}) =====")
    start = STAGE1_BEST if STAGE1_BEST.exists() else ROOT / "pretrained" / "yolov8s.pt"
    log(f"起点权重: {start}")
    TMP_PROJECT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = YOLO(str(start))
    model.train(
        data=str(ROOT / "datasets" / "helmet" / "data.yaml"),
        epochs=EPOCHS, imgsz=640, batch=16, device=0, workers=4,
        project=str(TMP_PROJECT), name="helmet_s2", patience=15, plots=True,
        seed=42, exist_ok=True, save=True, save_period=10, val=True,
    )
    dur = time.time() - t0
    tmp_run = TMP_PROJECT / "helmet_s2"
    new_map = read_best_map(tmp_run)

    dst = ROOT / "runs" / "helmet_s2"
    if dst.exists():
        dst = ROOT / "runs" / f"helmet_s2_{time.strftime('%m%d%H%M')}"
    shutil.copytree(tmp_run, dst)
    log(f"第二阶段完成：{dur/60:.1f} 分钟，最佳 mAP50-95={new_map:.4f} -> {dst}")

    # 与第一阶段最优比较
    old_map = read_best_map(ROOT / "runs" / "helmet")
    cur = ROOT / "models" / "helmet.pt"
    best_pt = dst / "weights" / "best.pt"
    deployed = False
    if best_pt.exists() and new_map >= old_map:
        shutil.copy2(best_pt, cur)
        deployed = True
        log(f"已部署第二阶段: 新 {new_map:.4f} >= 第一阶段 {old_map:.4f}")
    else:
        log(f"未替换: 第二阶段 {new_map:.4f} < 第一阶段 {old_map:.4f}，保留原权重（新权重在 {best_pt}）")

    summary = {
        "helmet_stage2": {
            "minutes": round(dur / 60, 1),
            "best_map50_95": round(new_map, 4),
            "stage1_map50_95": round(old_map, 4),
            "deployed": deployed,
            "run_dir": str(dst),
        }
    }
    out = ROOT / "runs" / "overnight_summary_stage2.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
