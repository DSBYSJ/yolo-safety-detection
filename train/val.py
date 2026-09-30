# -*- coding: utf-8 -*-
"""YOLOv8 模型评估脚本

在验证集上输出 Precision / Recall / mAP@0.5 / mAP@0.5:0.95 及各类别指标，
同时生成混淆矩阵、PR 曲线等可视化图（保存在 runs/val/<name>/ 下）。

用法示例：
    python train/val.py --weights models/helmet.pt --data datasets/helmet/data.yaml --name helmet
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else ROOT / p)


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8 评估脚本")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="val")
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(
        data=resolve(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(ROOT / "runs" / "val"),
        name=args.name,
        plots=True,
        exist_ok=True,
    )

    names = model.names  # {id: name}
    per_class = []
    for i, cid in enumerate(metrics.box.ap_class_index):
        per_class.append(
            {
                "class_id": int(cid),
                "class_name": names.get(int(cid), str(cid)),
                "precision": float(metrics.box.p[i]),
                "recall": float(metrics.box.r[i]),
                "mAP50": float(metrics.box.ap50[i]),
                "mAP50-95": float(metrics.box.ap[i]),
            }
        )

    summary = {
        "weights": args.weights,
        "data": args.data,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "per_class": per_class,
    }

    out_json = ROOT / "runs" / "val" / args.name / "metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()
