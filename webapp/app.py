# -*- coding: utf-8 -*-
"""安全帽/口罩佩戴检测系统 - Flask 主应用

启动：python webapp/app.py  （默认 http://127.0.0.1:5000）
"""
import json
import os
import shutil
import time
import uuid
from pathlib import Path

import cv2
from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

import camera
import config
import db
import detector
import imageio_cn
import train_manager
import video_jobs
from detector import ModelMissingError

app = Flask(__name__)
# 生产部署请通过环境变量 SECRET_KEY 注入；未设置时使用开发默认值
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-helmet-mask-detection")
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

db.init_db()

PAGE_SIZE = 10


# ---------------------------------------------------------------- 通用
def _rand_name(ext: str) -> str:
    return f"{int(time.time())}_{uuid.uuid4().hex[:8]}{ext}"


@app.get("/api/env")
def api_env():
    """运行环境信息（顶栏状态徽章）"""
    try:
        import torch

        device = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
        torch_ver = torch.__version__
    except Exception:
        device, torch_ver = "CPU", "未安装"
    return jsonify({"device": device, "torch": torch_ver})


@app.post("/api/train/deploy")
def api_train_deploy():
    """把某次历史训练的最优权重部署为线上模型"""
    body = request.get_json(force=True, silent=True) or {}
    name = body.get("run", "")
    kind = body.get("kind", "")
    if kind not in config.MODELS:
        return jsonify({"ok": False, "error": "kind 必须是 helmet 或 mask"}), 400
    src = config.RUNS_DIR / name / "weights" / "best.pt"
    if not src.exists():
        return jsonify({"ok": False, "error": f"权重不存在: {src}"}), 400
    config.MODEL_DIR.mkdir(exist_ok=True)
    shutil.copy(src, config.MODEL_DIR / config.MODELS[kind]["file"])
    detector._model_cache.pop(kind, None)  # 让检测服务下次重新加载新权重
    return jsonify({"ok": True})


@app.route("/runs/<path:p>")
def runs_files(p):
    """暴露训练产物（混淆矩阵、PR 曲线等）"""
    return send_from_directory(str(config.RUNS_DIR), p)


@app.errorhandler(ModelMissingError)
def handle_model_missing(e):
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": str(e)}), 400
    flash(str(e), "error")
    return redirect(url_for("train_page"))


# ---------------------------------------------------------------- 页面
@app.route("/")
def index():
    summary = db.stats_summary()
    model_info = []
    for kind, meta in config.MODELS.items():
        try:
            p = detector.model_path(kind)
            info = {
                "kind": kind,
                "title": meta["title"],
                "exists": True,
                "file": p.name,
                "size_mb": round(p.stat().st_size / 1048576, 1),
            }
        except ModelMissingError:
            info = {"kind": kind, "title": meta["title"], "exists": False, "file": None, "size_mb": 0}
        model_info.append(info)
    val_info = {}
    for kind in config.MODELS:
        mj = config.RUNS_DIR / "val" / kind / "metrics.json"
        if mj.exists():
            try:
                val_info[kind] = json.loads(mj.read_text(encoding="utf-8"))
            except Exception:
                pass
    recent, _ = db.query_records(page=1, size=6)
    return render_template(
        "index.html", page="home", summary=summary, model_info=model_info,
        val_info=val_info, recent=recent,
    )


@app.route("/detect")
def detect_page():
    return render_template("detect.html", page="detect")


@app.route("/camera")
def camera_page():
    return render_template("camera.html", page="camera")


@app.route("/records")
def records_page():
    return render_template("records.html", page="records")


@app.route("/stats")
def stats_page():
    return render_template("stats.html", page="stats")


@app.route("/train")
def train_page():
    with_flash = request.args.get("flash")
    return render_template("train.html", page="train", runs=train_manager.runs_list(), flash_msg=with_flash)


@app.route("/eval")
def eval_page():
    data = {}
    for kind in config.MODELS:
        data[kind] = train_manager.eval_metrics(kind)
    return render_template("eval.html", page="eval", data=data, titles={k: v["title"] for k, v in config.MODELS.items()})


# ---------------------------------------------------------------- 图片/视频检测 API
@app.post("/api/detect/image")
def api_detect_image():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"ok": False, "error": "请选择图片文件"}), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in config.ALLOWED_IMG_EXT:
        return jsonify({"ok": False, "error": f"不支持的图片格式: {ext}"}), 400

    kinds = request.form.getlist("kinds") or ["helmet"]
    try:
        conf = min(0.9, max(0.05, float(request.form.get("conf", 0.25))))
    except ValueError:
        conf = 0.25

    in_name = _rand_name(ext)
    in_path = config.UPLOAD_DIR / in_name
    f.save(str(in_path))

    img = imageio_cn.imread(in_path)
    if img is None:
        return jsonify({"ok": False, "error": "图片解析失败，请确认文件未损坏且为常见图片格式"}), 400

    try:
        dets, counts, annotated, tms = detector.infer_image(img, kinds, conf=conf)
    except ModelMissingError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    out_name = f"img_{_rand_name('.jpg')}"
    if not imageio_cn.imwrite(config.RESULT_DIR / out_name, annotated):
        return jsonify({"ok": False, "error": "结果图保存失败，请检查磁盘空间与目录权限"}), 500
    cols = detector.counts_to_db(counts)
    rec_id = db.insert_record(
        source_type="image",
        source_name=f.filename,
        kinds="+".join(detector.detect_kinds(kinds)),
        num_objects=sum(counts.values()),
        avg_conf=round(sum(d["conf"] for d in dets) / len(dets), 3) if dets else 0,
        duration_ms=tms,
        image_path=f"results/{out_name}",
        details=dets[:200],
        **cols,
    )
    return jsonify(
        {
            "ok": True,
            "record_id": rec_id,
            "image_url": f"static/results/{out_name}",
            "detections": dets,
            "counts": counts,
            "time_ms": tms,
            "db_cols": cols,
        }
    )


@app.post("/api/detect/video")
def api_detect_video():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"ok": False, "error": "请选择视频文件"}), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in config.ALLOWED_VIDEO_EXT:
        return jsonify({"ok": False, "error": f"不支持的视频格式: {ext}"}), 400
    kinds = request.form.getlist("kinds") or ["helmet"]
    try:
        conf = min(0.9, max(0.05, float(request.form.get("conf", 0.25))))
    except ValueError:
        conf = 0.25
    in_path = config.UPLOAD_DIR / _rand_name(ext)
    f.save(str(in_path))
    job_id = video_jobs.start_job(str(in_path), detector.detect_kinds(kinds), conf)
    return jsonify({"ok": True, "job_id": job_id})


@app.get("/api/detect/video/status/<job_id>")
def api_detect_video_status(job_id):
    return jsonify(video_jobs.get_status(job_id))


# ---------------------------------------------------------------- 摄像头
@app.get("/camera/feed")
def camera_feed():
    kind = request.args.get("kind", "helmet")
    camera.start(kind)
    return app.response_class(camera.mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/camera/kind")
def api_camera_kind():
    kind = request.json.get("kind", "helmet")
    camera.set_kind(kind)
    return jsonify({"ok": True})


@app.get("/api/camera/state")
def api_camera_state():
    return jsonify(camera.get_state())


@app.post("/api/camera/snapshot")
def api_camera_snapshot():
    res = camera.snapshot()
    if not res:
        return jsonify({"ok": False, "error": "当前没有可用画面"}), 400
    rid, img_path = res
    return jsonify({"ok": True, "record_id": rid, "image_url": f"static/{img_path}"})


# ---------------------------------------------------------------- 记录 API
@app.get("/api/records")
def api_records():
    try:
        page = max(1, int(request.args.get("page", 1)))
        size = min(100, max(1, int(request.args.get("size", PAGE_SIZE))))
    except ValueError:
        page, size = 1, PAGE_SIZE
    rows, total = db.query_records(
        page=page,
        size=size,
        kind=request.args.get("kind") or None,
        source=request.args.get("source") or None,
        keyword=request.args.get("q") or None,
    )
    for r in rows:
        r["violations"] = (r.get("head") or 0) + (r.get("face") or 0)
    return jsonify({"ok": True, "records": rows, "total": total, "page": page, "size": size})


@app.get("/api/records/<int:rid>")
def api_record_detail(rid):
    r = db.get_record(rid)
    if not r:
        return jsonify({"ok": False, "error": "记录不存在"}), 404
    if r.get("details"):
        try:
            r["detail_list"] = json.loads(r["details"])
        except Exception:
            r["detail_list"] = []
    else:
        r["detail_list"] = []
    r.pop("details", None)
    return jsonify({"ok": True, "record": r})


@app.post("/api/records/delete")
def api_records_delete():
    ids = request.json.get("ids") or []
    if not ids:
        return jsonify({"ok": False, "error": "请选择要删除的记录"}), 400
    db.delete_records(ids)
    return jsonify({"ok": True})


# ---------------------------------------------------------------- 统计 API
@app.get("/api/stats")
def api_stats():
    return jsonify(
        {
            "ok": True,
            "summary": db.stats_summary(),
            "daily": db.stats_daily(14),
            "classes": db.stats_classes(),
            "sources": db.stats_sources(),
            "conf": db.stats_conf(),
        }
    )


# ---------------------------------------------------------------- 训练/评估 API
@app.post("/api/train/start")
def api_train_start():
    body = request.get_json(force=True, silent=True) or {}
    kind = body.get("kind", "helmet")
    if kind not in config.MODELS:
        return jsonify({"ok": False, "error": "kind 必须是 helmet 或 mask"}), 400
    size = body.get("size", "s")
    if size not in ("n", "s", "m"):
        return jsonify({"ok": False, "error": "模型规模仅支持 n/s/m"}), 400
    try:
        epochs = int(body.get("epochs", 80))
        imgsz = int(body.get("imgsz", 640))
        batch = int(body.get("batch", 16))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "参数类型错误"}), 400
    epochs = min(300, max(10, epochs))
    imgsz = min(1280, max(320, imgsz))
    batch = min(64, max(2, batch))
    pid, err = train_manager.start(kind, size, epochs, imgsz, batch)
    if err:
        return jsonify({"ok": False, "error": err}), 400
    return jsonify({"ok": True, "pid": pid})


@app.get("/api/train/status")
def api_train_status():
    return jsonify(train_manager.status())


@app.get("/api/train/runs")
def api_train_runs():
    return jsonify({"ok": True, "runs": train_manager.runs_list()})


@app.post("/api/eval/start")
def api_eval_start():
    body = request.get_json(force=True, silent=True) or {}
    kind = body.get("kind", "helmet")
    if kind not in config.MODELS:
        return jsonify({"ok": False, "error": "kind 必须是 helmet 或 mask"}), 400
    pid, err = train_manager.eval_start(kind)
    if err:
        return jsonify({"ok": False, "error": err}), 400
    return jsonify({"ok": True, "pid": pid})


@app.get("/api/eval/status")
def api_eval_status():
    return jsonify(train_manager.eval_status())


@app.get("/api/eval/metrics/<kind>")
def api_eval_metrics(kind):
    if kind not in config.MODELS:
        abort(404)
    return jsonify(train_manager.eval_metrics(kind))


# ---------------------------------------------------------------- 启动
if __name__ == "__main__":
    try:
        import torch

        device = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    except Exception:
        device = "CPU"
    print("=" * 60)
    print("  安全帽/口罩佩戴检测系统  |  YOLOv8 + Flask")
    print(f"  计算设备: {device}")
    print("  访问地址: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
