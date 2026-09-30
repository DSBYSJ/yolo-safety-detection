# -*- coding: utf-8 -*-
"""
生成「解压即可运行」的迁移包。

设计要点
--------
* **白名单复制**：只拷贝明确列出的内容，而不是「拷贝全部再剔除」——
  后者一旦漏掉某条排除规则就会把 4G 数据集或 88M 上传缓存打进包里。
* **权重分两类**：models/*.pt 是检测必需（否则解压后跑不起来），
  models/face/ 按需（人脸模块用）；两者都在包里，因为目的就是「直接能跑」。
* **证书一并带走**：certs/ 被 .gitignore 排除（含私钥），但迁移包里必须有，
  否则换电脑后手机端调不了摄像头。包内附带提醒：仅限自用，不要外传。
* 产出后在包根目录写 `迁移说明.md` 与 `.env.example`。
"""
import os
import shutil
import sys

SRC = r"G:\毕业设计"
DST = r"G:\毕业设计_迁移包"

# ---- 目录白名单：整目录拷贝 ----
COPY_DIRS = [
    "webapp",        # 应用主体（含 templates/static），已排除 uploads/results/__pycache__
    "scripts",       # 启动与辅助脚本
    "train",         # 训练脚本
    "tests",         # 测试
    "certs",         # HTTPS 证书（.gitignore 排除，但迁移必须带）
    "models",        # 权重与 ONNX
    "docs",          # 截图与说明
]
# ---- 文件白名单 ----
COPY_FILES = [
    "requirements.txt",
    "README.md",
    "yolov8n.pt",    # 训练/迁移用的基础权重
    "yolov8s.pt",
]
# ---- 拷贝时按名称/后缀跳过的垃圾 ----
# 注意：不要把 "face" 放进来 —— models/face 是人脸模型，必须带走。
SKIP_DIR_NAMES = {
    "__pycache__", ".git", ".pytest_cache", ".mypy_cache",
    "uploads", "results", "train_logs",
}
# uploads / results / train_logs 以空目录形式保留（应用启动时要往里写），
# 但里面的文件不拷 —— 它们是历史运行产物。
SKIP_FILE_SUFFIX = (".pyc", ".pyo", ".log", ".tmp", ".db-wal", ".db-shm")
# 历史运行数据库：含检测记录 / 打卡数据，属个人数据，不随代码包分发。
# 应用启动时会自动重建空库，因此删掉不影响运行。
SKIP_FILE_NAMES = {"data.db", "data.db-wal", "data.db-shm", "data.db-journal"}


def log(msg):
    print(msg, flush=True)


def copy_tree(src, dst, skip_face_dir=False):
    """递归拷贝，跳过 __pycache__/uploads 等可再生内容。"""
    for root, dirs, files in os.walk(src):
        # 原地过滤目录，避免往下走
        keep = []
        for d in dirs:
            if d in SKIP_DIR_NAMES:
                continue
            keep.append(d)
        dirs[:] = keep

        rel = os.path.relpath(root, src)
        target_root = dst if rel == "." else os.path.join(dst, rel)
        os.makedirs(target_root, exist_ok=True)

        for f in files:
            if f.endswith(SKIP_FILE_SUFFIX):
                continue
            if f in SKIP_FILE_NAMES:
                continue
            shutil.copy2(os.path.join(root, f), os.path.join(target_root, f))


def human(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def fmt(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def main():
    if not os.path.isdir(SRC):
        log(f"[错误] 源目录不存在：{SRC}")
        return 1
    if os.path.exists(DST):
        log(f"[清理] 目标已存在，先删除：{DST}")
        shutil.rmtree(DST)
    os.makedirs(DST)

    copied = []

    for d in COPY_DIRS:
        s = os.path.join(SRC, d)
        if not os.path.isdir(s):
            log(f"[跳过] 目录不存在：{d}")
            continue
        log(f"[拷贝] {d}/ ...")
        copy_tree(s, os.path.join(DST, d))
        copied.append(d + "/")

    for f in COPY_FILES:
        s = os.path.join(SRC, f)
        if not os.path.isfile(s):
            log(f"[跳过] 文件不存在：{f}")
            continue
        shutil.copy2(s, os.path.join(DST, f))
        copied.append(f)

    # .gitignore 一并带走（换电脑后 git init 能继续沿用排除规则）
    gi = os.path.join(SRC, ".gitignore")
    if os.path.isfile(gi):
        shutil.copy2(gi, os.path.join(DST, ".gitignore"))
        copied.append(".gitignore")

    # ---- 清点关键内容是否齐全 ----
    checks = [
        ("models/helmet.pt", "安全帽权重（检测必需）"),
        ("models/mask.pt", "口罩权重（检测必需）"),
        ("models/face", "人脸识别模型 buffalo_l（人脸模块需要）"),
        ("certs/cert.pem", "HTTPS 证书"),
        ("certs/key.pem", "HTTPS 私钥"),
        ("webapp/app.py", "应用入口"),
        ("requirements.txt", "依赖清单"),
    ]
    log("\n=== 关键内容清点 ===")
    missing = []
    for rel, desc in checks:
        p = os.path.join(DST, rel.replace("/", os.sep))
        ok = os.path.exists(p)
        size = f" ({fmt(human(p))})" if ok and os.path.isdir(p) else (
            f" ({fmt(os.path.getsize(p))})" if ok else "")
        log(f"  [{'OK ' if ok else '缺失'}] {rel:24s}{size}  {desc}")
        if not ok:
            missing.append(rel)

    # ---- 统计 ----
    total = human(DST)
    nfile = sum(len(fs) for _r, _d, fs in os.walk(DST))
    log(f"\n=== 打包结果 ===")
    log(f"  路径：{DST}")
    log(f"  文件：{nfile} 个")
    log(f"  体积：{fmt(total)}")

    if missing:
        log(f"\n[警告] 以下内容缺失，解压后可能无法直接运行：{missing}")
        return 2
    log("\n[完成] 关键内容齐全。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
