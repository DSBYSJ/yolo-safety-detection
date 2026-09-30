# -*- coding: utf-8 -*-
"""YOLOv8 模型训练脚本

用法示例：
    python train/train.py --data datasets/helmet/data.yaml --model yolov8s.pt \
        --epochs 80 --imgsz 640 --batch 16 --name helmet_s --deploy-name helmet

训练完成后若指定 --deploy-name，会把最优权重拷贝到 models/<deploy-name>.pt 供 Web 系统使用。
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str) -> str:
    """相对路径以项目根目录为基准。"""
    p = Path(path)
    return str(p if p.is_absolute() else ROOT / p)


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8 训练脚本")
    parser.add_argument("--data", required=True, help="data.yaml 路径")
    parser.add_argument("--model", default="yolov8n.pt", help="预训练权重: yolov8n/s/m/l/x")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16, help="-1 为自动批量")
    parser.add_argument("--device", default="0", help="0/1/cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="exp", help="本次运行名称（runs/ 下）")
    parser.add_argument("--deploy-name", default=None, help="训练完成后部署到 models/<name>.pt")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=resolve(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(ROOT / "runs"),
        name=args.name,
        patience=20,
        plots=True,
        seed=42,
        exist_ok=True,
    )

    best = ROOT / "runs" / args.name / "weights" / "best.pt"
    if args.deploy_name and best.exists():
        ROOT.joinpath("models").mkdir(exist_ok=True)
        dst = ROOT / "models" / f"{args.deploy_name}.pt"
        shutil.copy(best, dst)
        print(f"[deploy] 已部署最优权重 -> {dst}")

    print(f"[done] 训练完成，结果目录: runs/{args.name}")


if __name__ == "__main__":
    main()
