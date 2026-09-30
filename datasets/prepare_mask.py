# -*- coding: utf-8 -*-
"""口罩检测数据集构建脚本

数据来源：hmnshudhmn24/face-mask-detection（HuggingFace，即经典 maksssksksss 数据集）
        853 张图像 + 853 个 PASCAL VOC XML 标注，需先下载到 datasets/raw/mask/
        （images/ 与 annotations/ 两个子目录）。

类别映射（YOLO）：
    0 = mask  口罩（with_mask）
    1 = face  未戴口罩（without_mask；mask_weared_incorrect 佩戴不规范视同未佩戴）

输出：datasets/mask/{images,labels}/{train,val} 与 data.yaml，按 8:2 随机划分（seed=42）。

用法：
    python datasets/prepare_mask.py
    python datasets/prepare_mask.py --download   # 先从 HF 镜像下载原始数据再转换
"""
import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw" / "mask"
OUT = ROOT / "datasets" / "mask"

CLASS_MAP = {"with_mask": 0, "without_mask": 1, "mask_weared_incorrect": 1}
NAMES = ["mask", "face"]
VAL_RATIO = 0.2
SEED = 42

REPO_ID = "hmnshudhmn24/face-mask-detection"


def download_raw() -> None:
    """从 HF 镜像下载原始数据（国内使用 hf-mirror.com 加速）。"""
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(RAW),
        endpoint="https://hf-mirror.com",
        max_workers=8,
    )
    print(f"[download] 原始数据已下载到 {RAW}")


def convert_box(size, box):
    """VOC (xmin,ymin,xmax,ymax) -> YOLO 归一化 (cx,cy,w,h)，非法框返回 None。"""
    w, h = size
    xmin, ymin, xmax, ymax = box
    xmin, xmax = max(0.0, min(xmin, w)), max(0.0, min(xmax, w))
    ymin, ymax = max(0.0, min(ymin, h)), max(0.0, min(ymax, h))
    bw, bh = xmax - xmin, ymax - ymin
    if bw < 2 or bh < 2:
        return None
    return (xmin + bw / 2) / w, (ymin + bh / 2) / h, bw / w, bh / h


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="先下载数据集")
    args = parser.parse_args()

    if args.download:
        download_raw()

    img_dir, ann_dir = RAW / "images", RAW / "annotations"
    xmls = sorted(ann_dir.glob("*.xml"))
    if not xmls:
        sys_exit = (
            f"未找到标注文件: {ann_dir}\n"
            "请先运行: python datasets/prepare_mask.py --download"
        )
        raise SystemExit(sys_exit)

    random.seed(SEED)
    random.shuffle(xmls)
    n_val = max(1, int(len(xmls) * VAL_RATIO))
    splits = {"val": xmls[:n_val], "train": xmls[n_val:]}

    stats = {"images": 0, "skipped": 0, "boxes": {0: 0, 1: 0}}
    for split, files in splits.items():
        img_out = OUT / "images" / split
        lbl_out = OUT / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for xml in files:
            try:
                root = ET.parse(xml).getroot()
            except ET.ParseError:
                stats["skipped"] += 1
                continue

            size_node = root.find("size")
            if size_node is None:
                stats["skipped"] += 1
                continue
            size = (float(size_node.findtext("width", 0)), float(size_node.findtext("height", 0)))
            if size[0] <= 0 or size[1] <= 0:
                stats["skipped"] += 1
                continue

            img_name = root.findtext("filename") or (xml.stem + ".png")
            src_img = img_dir / img_name
            if not src_img.exists():
                # 有些图片扩展名与标注不一致，尝试常见扩展名
                found = None
                for ext in (".png", ".jpg", ".jpeg"):
                    cand = img_dir / (xml.stem + ext)
                    if cand.exists():
                        found = cand
                        break
                if not found:
                    stats["skipped"] += 1
                    continue
                src_img = found
                img_name = src_img.name

            lines = []
            for obj in root.iter("object"):
                cls = CLASS_MAP.get((obj.findtext("name") or "").strip())
                if cls is None:
                    continue
                bb = obj.find("bndbox")
                if bb is None:
                    continue
                box = [float(bb.findtext(t, 0)) for t in ("xmin", "ymin", "xmax", "ymax")]
                yolo = convert_box(size, box)
                if yolo is None:
                    continue
                lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in yolo))
                stats["boxes"][cls] += 1

            if not lines:  # 无有效目标的图像不参与训练
                stats["skipped"] += 1
                continue

            shutil.copy(src_img, img_out / img_name)
            (lbl_out / (Path(img_name).stem + ".txt")).write_text(
                "\n".join(lines), encoding="utf-8"
            )
            stats["images"] += 1

    yaml_text = (
        f"path: {OUT.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(NAMES)}\n"
        "names:\n"
        "  0: mask\n"
        "  1: face\n"
    )
    (OUT / "data.yaml").write_text(yaml_text, encoding="utf-8")
    print("[done] 口罩数据集构建完成:", stats)
    print(f"[yaml] {OUT / 'data.yaml'}")


if __name__ == "__main__":
    main()
