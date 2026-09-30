# -*- coding: utf-8 -*-
"""安全帽检测数据集构建脚本

数据来源：keremberke/hard-hat-detection（HuggingFace，源自 Roboflow Universe "Hard Hats"）
        共 19,745 张图像（train 13782 / valid 3962 / test 2001），COCO JSON 标注，
        需先下载 train.zip / valid.zip / test.zip 到 datasets/raw/helmet/。

类别映射（YOLO）：
    0 = helmet 安全帽（hardhat）
    1 = head   未戴安全帽（no-hardhat）

划分：train + test 合并为训练集（15,783 张），valid 作为验证集（3,962 张）。
输出：datasets/helmet/{images,labels}/{train,val} 与 data.yaml

用法：python datasets/prepare_helmet.py
"""
import argparse
import json
import shutil
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw" / "helmet"
OUT = ROOT / "datasets" / "helmet"

NAMES = ["helmet", "head"]
# 分包 -> 目标 split（train+test 合并进 train）
SPLITS = {"train.zip": "train", "test.zip": "train", "valid.zip": "val"}


def coco_to_yolo(bbox, w, h):
    x, y, bw, bh = bbox
    cx, cy = x + bw / 2, y + bh / 2
    return cx / w, cy / h, bw / w, bh / h


def convert_split(zip_path: Path, split: str, stats: dict) -> None:
    """解压一个分包并按 COCO 标注生成 YOLO labels。"""
    if not zip_path.exists():
        raise SystemExit(f"缺少数据包: {zip_path}")

    img_out = OUT / "images" / split
    lbl_out = OUT / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW / zip_path.stem)

    extract_dir = RAW / zip_path.stem
    ann_file = extract_dir / "_annotations.coco.json"
    coco = json.loads(ann_file.read_text(encoding="utf-8"))

    # 类别 id -> 我们的 0/1（按名字映射，稳妥）
    cat_map = {}
    for c in coco.get("categories", []):
        name = c["name"].lower()
        if name in ("hardhat", "helmet"):
            cat_map[c["id"]] = 0
        elif name in ("no-hardhat", "no-helmet", "head"):
            cat_map[c["id"]] = 1

    img_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_img: dict = {}
    for a in coco.get("annotations", []):
        if a.get("category_id") not in cat_map:
            continue
        anns_by_img.setdefault(a["image_id"], []).append(a)

    for im in coco["images"]:
        anns = anns_by_img.get(im["id"], [])
        if not anns:
            continue  # 无有效目标的图像不参与训练
        src = extract_dir / im["file_name"]
        if not src.exists():
            stats["missing"] += 1
            continue

        base_name = Path(im["file_name"]).name
        base_stem = Path(im["file_name"]).stem
        # 幂等：已转换过的图像直接跳过（源文件保留在 raw 目录，不做删除）
        if (img_out / base_name).exists() and (lbl_out / (base_stem + ".txt")).exists():
            continue

        w, h = im["width"], im["height"]
        lines = []
        for a in anns:
            cls = cat_map[a["category_id"]]
            yolo = coco_to_yolo(a["bbox"], w, h)
            if not all(0.0 <= v <= 1.0 for v in yolo) or yolo[2] <= 0.001 or yolo[3] <= 0.001:
                continue
            lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in yolo))
            stats["boxes"][cls] += 1
        if not lines:
            continue

        dst = img_out / base_name
        if dst.exists():  # 同名冲突极小概率，加后缀保证不覆盖
            dst = img_out / (base_stem + f"_{im['id']}.jpg")
        # Windows 下刚解压的文件可能被占用，复制比移动更稳，失败则重试
        for attempt in range(3):
            try:
                shutil.copy2(src, dst)
                break
            except PermissionError:
                if attempt == 2:
                    raise
                time.sleep(0.3)
        (lbl_out / (dst.stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
        stats["images"] += 1
        # 注意：不删除 raw 源文件（幂等续跑依赖它们；磁盘充足时无需清理）


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-raw", action="store_true", help="保留解压的原始目录")
    args = parser.parse_args()

    stats = {"images": 0, "missing": 0, "boxes": {0: 0, 1: 0}}
    for zip_name, split in SPLITS.items():
        print(f"[convert] {zip_name} -> {split}")
        convert_split(RAW / zip_name, split, stats)

    yaml_text = (
        f"path: {OUT.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(NAMES)}\n"
        "names:\n"
        "  0: helmet\n"
        "  1: head\n"
    )
    (OUT / "data.yaml").write_text(yaml_text, encoding="utf-8")

    print("[done] 安全帽数据集构建完成:", stats)
    print(f"[yaml] {OUT / 'data.yaml'}")
    print("[note] 原始解压目录保留在 datasets/raw/helmet/ 下，如需清理请手动删除")


if __name__ == "__main__":
    main()
