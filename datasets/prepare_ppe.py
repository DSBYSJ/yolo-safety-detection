# -*- coding: utf-8 -*-
"""PPE（个人防护装备）数据集构建脚本 —— 在安全帽/口罩之外的扩充类别

数据来源：VincentGOURBIN/ppe-detection（HuggingFace，源自 Roboflow Universe）
        Roboflow 导出即 YOLO 格式，5 个类别：
            0 = helmet     安全帽
            1 = no-helmet  未戴安全帽
            2 = no-vest    未穿安全背心
            3 = person     人员
            4 = vest       安全背心

用法：
    python datasets/prepare_ppe.py --download   # 先从 HF 镜像下载再构建
    python datasets/prepare_ppe.py              # 已下载时仅构建
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw" / "ppe"
OUT = ROOT / "datasets" / "ppe"

REPO_ID = "VincentGOURBIN/ppe-detection"

# Roboflow 分包 -> 目标 split
SPLIT_MAP = {"train": "train", "valid": "val", "test": "val"}

NAMES = ["helmet", "no-helmet", "no-vest", "person", "vest"]


def download_raw() -> None:
    """从 HF 镜像下载原始数据。"""
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(RAW),
        endpoint="https://hf-mirror.com",
        max_workers=8,
    )
    print(f"[download] 原始数据已下载到 {RAW}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="先下载数据集")
    args = parser.parse_args()

    if args.download:
        download_raw()

    stats = {"images": 0, "labels": 0, "missing_label": 0}
    for src_split, dst_split in SPLIT_MAP.items():
        img_src = RAW / src_split / "images"
        lbl_src = RAW / src_split / "labels"
        if not img_src.exists():
            print(f"[warn] 缺少目录: {img_src}")
            continue
        img_out = OUT / "images" / dst_split
        lbl_out = OUT / "labels" / dst_split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img in sorted(img_src.iterdir()):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                continue
            lbl = lbl_src / (img.stem + ".txt")
            if not lbl.exists():
                stats["missing_label"] += 1
                continue
            try:
                txt = lbl.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if not txt:
                stats["missing_label"] += 1
                continue
            # 过滤越界类别 id，避免训练时 index error
            keep = []
            for line in txt.splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    cid = int(float(parts[0]))
                except ValueError:
                    continue
                if 0 <= cid < len(NAMES) and len(parts) >= 5:
                    keep.append(line)
            if not keep:
                stats["missing_label"] += 1
                continue
            dst_img = img_out / img.name
            if not dst_img.exists():
                shutil.copy2(img, dst_img)
            (lbl_out / (img.stem + ".txt")).write_text("\n".join(keep), encoding="utf-8")
            stats["images"] += 1
            stats["labels"] += 1

    if stats["images"] == 0:
        raise SystemExit(
            f"未构建出任何样本，请检查 {RAW}\n"
            "请先运行: python datasets/prepare_ppe.py --download"
        )

    yaml_text = (
        f"path: {OUT.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(NAMES)}\n"
        "names:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES))
    )
    (OUT / "data.yaml").write_text(yaml_text, encoding="utf-8")
    print("[done] PPE 数据集构建完成:", stats)
    print(f"[yaml] {OUT / 'data.yaml'}")


if __name__ == "__main__":
    main()
