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
import numpy as np
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

import attendance
import camera
import camera_db
import config
import db
import detector
import face_db
import imageio_cn
import phone_face
import train_manager
import video_jobs
from detector import ModelMissingError

app = Flask(__name__)
# 生产部署请通过环境变量 SECRET_KEY 注入；未设置时使用开发默认值
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-helmet-mask-detection")
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

db.init_db()
face_db.init_db()
camera_db.init_db()
# 首次启动把环境变量里那路摄像头导入配置表，页面上就能直接看到并管理
camera_db.seed_from_env()

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


@app.route("/faces")
def faces_page():
    """人脸底库：注册与查看已录入人员"""
    return render_template(
        "faces.html",
        page="faces",
        ready=face_db.face_model_ready(),
        threshold=face_db.MATCH_THRESHOLD,
    )


@app.route("/compliance")
def compliance_page():
    """合规统计：按「人」聚合的识别记录与合规情况"""
    return render_template(
        "compliance.html", page="compliance", ready=face_db.face_model_ready()
    )


@app.route("/attendance")
def attendance_page():
    """人脸打卡：只依据「当日是否识别到」判定打卡 / 缺勤"""
    return render_template(
        "attendance.html",
        page="attendance",
        ready=face_db.face_model_ready(),
        today=time.strftime("%Y-%m-%d"),
        gallery_size=face_db.count_persons(),
        threshold=face_db.MATCH_THRESHOLD,
    )


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


@app.post("/api/camera/face")
def api_camera_face():
    """开关摄像头实时人脸识别。"""
    body = request.get_json(force=True, silent=True) or {}
    on = bool(body.get("on"))
    if on and not face_db.face_model_ready():
        return jsonify({"ok": False, "error": "人脸模型未就绪，无法开启识别"}), 400
    camera.set_face(on)
    return jsonify({"ok": True, "face": on})


@app.get("/api/camera/state")
def api_camera_state():
    return jsonify(camera.get_state())


# ------------------------------------------------- 监控摄像头配置管理
@app.get("/api/cameras")
def api_cameras_list():
    """列出已配置的摄像头（地址已脱敏，绝不下发明文凭据）。"""
    active = camera.get_state().get("active_camera")
    return jsonify({
        "ok": True,
        "items": camera_db.list_cameras(masked=True),
        "types": [{"value": t, "label": camera_db.TYPE_CN[t]}
                  for t in camera_db.SOURCE_TYPES],
        "active": active,
        "count": camera_db.count(),
    })


@app.post("/api/cameras/add")
def api_cameras_add():
    """新增一路摄像头配置。

    校验失败返回 400 并把原因原文带上 —— 这些都是用户填错字段造成的，
    说清楚哪错了比一句「参数错误」有用得多。
    """
    body = request.get_json(force=True, silent=True) or {}
    try:
        cid = camera_db.add(
            body.get("name"), body.get("type"), body.get("url"),
            location=body.get("location"), note=body.get("note"),
        )
    except camera_db.CameraConfigError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    return jsonify({"ok": True, "id": cid, "item": camera_db.get(cid) and
                    {**camera_db.get(cid), "url": camera_db.mask_url(camera_db.get(cid)["url"])}})


@app.post("/api/cameras/update")
def api_cameras_update():
    body = request.get_json(force=True, silent=True) or {}
    cid = body.get("id")
    if cid is None:
        return jsonify({"ok": False, "error": "缺少 id"}), 400
    try:
        ok = camera_db.update(
            cid,
            name=body.get("name"), ctype=body.get("type"), url=body.get("url"),
            location=body.get("location"), note=body.get("note"),
            enabled=body.get("enabled"),
        )
    except camera_db.CameraConfigError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    if not ok:
        return jsonify({"ok": False, "error": "摄像头不存在"}), 404
    return jsonify({"ok": True})


@app.post("/api/cameras/delete")
def api_cameras_delete():
    body = request.get_json(force=True, silent=True) or {}
    cid = body.get("id")
    if cid is None:
        return jsonify({"ok": False, "error": "缺少 id"}), 400
    # 删掉的正是当前在用的那路 → 同时退回环境变量来源，避免画面卡在已删配置上
    if camera.get_state().get("active_camera") == int(cid):
        camera.set_active_camera(None)
    ok = camera_db.delete(cid)
    return jsonify({"ok": bool(ok), "error": "" if ok else "摄像头不存在"})


@app.post("/api/cameras/switch")
def api_cameras_switch():
    """切换当前使用的摄像头。传 id=null 表示回到环境变量配置的那一路。"""
    body = request.get_json(force=True, silent=True) or {}
    cid = body.get("id")
    if cid in ("", "null", 0, "0"):
        cid = None
    if cid is not None:
        try:
            cid = int(cid)
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "id 必须是整数或 null"}), 400
    if not camera.set_active_camera(cid):
        return jsonify({"ok": False, "error": "摄像头不存在"}), 404
    camera.start()          # 确保线程在跑；reload 会让它重连新来源
    return jsonify({"ok": True, "active": cid})


@app.post("/api/cameras/test")
def api_cameras_test():
    """测试某路摄像头是否连得通（不切换当前画面）。

    现场配置 RTSP 最常踩的坑是地址写错、端口不对、密码过期，
    等页面黑屏再排查很费劲。这里单独给一个「先测再切」的入口。
    """
    body = request.get_json(force=True, silent=True) or {}
    cid = body.get("id")
    if cid is None:
        return jsonify({"ok": False, "error": "缺少 id"}), 400

    res = camera_db.resolve_url(int(cid))
    if not res:
        return jsonify({"ok": False, "error": "摄像头不存在或地址非法"}), 404
    ctype, target = res

    cap = None
    try:
        if ctype == "device":
            cap = cv2.VideoCapture(int(target))
        else:
            params = [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, config.CAMERA_STREAM_TIMEOUT * 1000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, config.CAMERA_STREAM_TIMEOUT * 1000,
            ]
            try:
                cap = cv2.VideoCapture(target, cv2.CAP_FFMPEG, params)
            except (cv2.error, TypeError):
                cap = cv2.VideoCapture(target)
        if not cap or not cap.isOpened():
            camera_db.mark_result(int(cid), False, "无法打开视频源")
            return jsonify({"ok": False, "error": "无法打开该视频源，请检查地址 / 端口 / 账号"}), 200
        ok, frame = cap.read()
        if not ok or frame is None:
            camera_db.mark_result(int(cid), False, "连上了但读不到画面")
            return jsonify({"ok": False, "error": "已连接但读不到画面，可能是编码不受支持"}), 200
        h, w = frame.shape[:2]
        camera_db.mark_result(int(cid), True)
        return jsonify({"ok": True, "size": [w, h]})
    except Exception as e:  # noqa: BLE001
        camera_db.mark_result(int(cid), False, str(e))
        return jsonify({"ok": False, "error": f"连接失败：{e}"}), 200
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:  # noqa: BLE001
                pass


@app.post("/api/camera/snapshot")
def api_camera_snapshot():
    res = camera.snapshot()
    if not res:
        return jsonify({"ok": False, "error": "当前没有可用画面"}), 400
    rid, img_path = res
    return jsonify({"ok": True, "record_id": rid, "image_url": f"static/{img_path}"})


# ------------------------------------------- 手机端（浏览器直接调摄像头）
# 手机浏览器通过 getUserMedia 拿到本地画面，逐帧把 JPEG 发到这里推理。
# 与 /camera/feed 的区别：那条链路是「服务端摄像头 → 浏览器」，
# 这条是「浏览器摄像头 → 服务端推理 → 浏览器」，因此手机无需装推流 App。
_PHONE_MAX_FRAME_BYTES = 6 * 1024 * 1024   # 单帧上限 6MB，防止异常大图打满内存


@app.post("/api/phone/detect")
def api_phone_detect():
    """接收手机浏览器传来的一帧图像，返回检测结果与计数。

    默认**不落库** —— 逐帧落库会在几秒内产生上千条记录。
    只有显式传 save=1（用户点「抓拍存档」）时才写入。

    人脸识别由 body.face_mode 控制：off / interval / every。
    识别结果里的框与安全帽框共用同一套坐标系（都是「上传帧」像素），
    因此前端可以用同一套换算逻辑画两种框。人脸结果同样只在 save=1
    时落库，理由与上面一致。
    """
    import base64

    body = request.get_json(force=True, silent=True) or {}
    data_url = body.get("image") or ""
    if not data_url:
        return jsonify({"ok": False, "error": "缺少图像数据"}), 400
    if len(data_url) > _PHONE_MAX_FRAME_BYTES:
        return jsonify({"ok": False, "error": "图像过大"}), 413

    # 接受两种格式：data:image/jpeg;base64,xxx 或纯 base64
    if "," in data_url[:64]:
        data_url = data_url.split(",", 1)[1]
    try:
        raw = base64.b64decode(data_url, validate=False)
    except Exception:
        return jsonify({"ok": False, "error": "图像解码失败"}), 400

    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"ok": False, "error": "图像解析失败"}), 400

    kinds = body.get("kinds") or ["helmet"]
    if isinstance(kinds, str):
        kinds = [kinds]
    try:
        conf = min(0.9, max(0.05, float(body.get("conf", 0.25))))
    except (TypeError, ValueError):
        conf = 0.25

    want_save = str(body.get("save", "")).lower() in ("1", "true", "yes")

    # 逐帧请求必须「宁可丢帧，不可排队」：等锁超时说明推理通道被占满
    # （比如摄像头线程正在跑），直接告诉前端跳过这一帧。
    # 注意服务端为单线程串行，客户端帧间隔 250ms > 单帧推理 60~80ms，
    # 正常情况下不会走到这里；一旦频繁触发说明有别的推理在抢，
    # 超时值取 2 秒即可，不必久等。
    try:
        dets, counts, annotated, tms = detector.infer_image(
            img, kinds, conf=conf, timeout=2.0
        )
    except detector.InferBusy:
        return jsonify({"ok": False, "error": "推理繁忙，本帧已跳过", "skip": True}), 503
    except ModelMissingError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    # ---- 人脸识别（附加能力，失败不影响上面这条主线） ----
    face_mode = phone_face.normalize_mode(body.get("face_mode"))
    client_key = str(body.get("client") or request.remote_addr or "-")[:64]
    faces = []
    if face_mode != "off" or want_save:
        if phone_face.should_run(client_key, face_mode, save=want_save):
            # 只有真的跑了才记时间：识别失败（无脸/模型缺失）不记，
            # 否则接下来两秒都不会再试，看起来像卡住了。
            phone_face.mark_ran(client_key)
            # 把装备检测框一起传进去 → 人脸与装备按空间重叠配对，
            # 于是「谁没戴安全帽」可以落到具体的人头上
            faces = phone_face.recognize(img, save=want_save, detections=dets,
                                         source="phone")

    person_sum = {}
    if faces:
        try:
            import equip
            person_sum = equip.summarize(faces)
        except Exception:  # noqa: BLE001
            person_sum = {}

    resp = {
        "ok": True,
        "counts": counts,
        "db_cols": detector.counts_to_db(counts),
        "num_objects": sum(counts.values()),
        "time_ms": tms,
        "size": [img.shape[1], img.shape[0]],
        # 检测框坐标（原图像素）。前端据 size 换算成显示比例后叠加绘制，
        # 这样画框与画面严格同步 —— 服务端不需要回传标注图（省带宽、免二次编码）。
        "detections": dets,
        "face_mode": face_mode,
        "faces": faces,
        "person_compliance": person_sum,
        "face_ready": face_db.face_model_ready(),
        # 提示前端最快多久后再来人脸请求（interval 模式下可用于主动降频）
        "next_face_sec": phone_face.next_interval(client_key, face_mode),
    }

    # 只在用户点「抓拍存档」时落盘 + 落库
    if want_save:
        out_name = f"phone_{_rand_name('.jpg')}"
        if not imageio_cn.imwrite(config.RESULT_DIR / out_name, annotated):
            return jsonify({"ok": False, "error": "结果图保存失败"}), 500
        rec_id = db.insert_record(
            source_type="camera",
            source_name="手机网页摄像头",
            kinds="+".join(detector.detect_kinds(kinds)),
            num_objects=sum(counts.values()),
            avg_conf=round(sum(d["conf"] for d in dets) / len(dets), 3) if dets else 0,
            duration_ms=tms,
            image_path=f"results/{out_name}",
            details=dets[:200],
            **detector.counts_to_db(counts),
        )
        resp["record_id"] = rec_id
        resp["image_url"] = f"static/results/{out_name}"

    return jsonify(resp)


@app.get("/phone")
def phone_page():
    """手机端检测页：用浏览器自带摄像头实时检测"""
    return render_template("phone.html", page="phone")


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


# ---------------------------------------------------------------- 人脸底库 API
@app.get("/api/faces/status")
def api_faces_status():
    """人脸模块状态：模型是否就绪、底库人数"""
    return jsonify(
        {
            "ok": True,
            "model_ready": face_db.face_model_ready(),
            "threshold": face_db.MATCH_THRESHOLD,
            "gallery_size": face_db.count_persons() if face_db.face_model_ready() else 0,
        }
    )


@app.get("/api/faces/list")
def api_faces_list():
    """已注册人员列表（不含特征向量，向量体积大且毫无展示价值）"""
    try:
        persons = face_db.list_persons()
    except Exception as e:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"读取底库失败: {e}"}), 500
    return jsonify({"ok": True, "persons": persons})


@app.post("/api/faces/register")
def api_faces_register():
    """提交人脸照片并命名此人。

    表单：file（照片）、name（姓名）、note（备注，可选）
    """
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"ok": False, "error": "请选择一张人脸照片"}), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in config.ALLOWED_IMG_EXT:
        return jsonify({"ok": False, "error": f"不支持的图片格式: {ext}"}), 400

    name = (request.form.get("name") or "").strip()
    if not name:
        return jsonify({"ok": False, "error": "请填写姓名"}), 400
    note = (request.form.get("note") or "").strip()

    in_name = _rand_name(ext)
    in_path = config.UPLOAD_DIR / in_name
    f.save(str(in_path))

    img = imageio_cn.imread(in_path)
    if img is None:
        return jsonify({"ok": False, "error": "图片解析失败，请确认文件未损坏"}), 400

    try:
        emb = face_db.extract_single(img)
    except face_db.FaceModelMissingError as e:
        return jsonify({"ok": False, "error": str(e)}), 503
    except face_db.NoFaceFound as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    # 存一份缩略图便于底库里辨认，避免直接引用用户上传的大图
    thumb_rel = ""
    try:
        thumb_name = f"face_{uuid.uuid4().hex[:10]}.jpg"
        scale = 240.0 / max(img.shape[:2])
        if scale < 1:
            thumb = cv2.resize(
                img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        else:
            thumb = img
        if imageio_cn.imwrite(config.FACE_THUMB_DIR / thumb_name, thumb):
            thumb_rel = f"faces/{thumb_name}"
    except Exception:  # noqa: BLE001
        thumb_rel = ""          # 缩略图存不下来不该阻止注册

    try:
        pid = face_db.register(name, emb, note=note, thumb=thumb_rel)
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    return jsonify({"ok": True, "id": pid, "name": name, "thumb": thumb_rel})


@app.post("/api/faces/delete")
def api_faces_delete():
    body = request.get_json(force=True, silent=True) or {}
    ids = body.get("ids") or []
    ids = [int(i) for i in ids if str(i).isdigit()]
    if not ids:
        return jsonify({"ok": False, "error": "请选择要删除的人员"}), 400
    for i in ids:
        face_db.delete_person(i)
    return jsonify({"ok": True, "deleted": len(ids)})


@app.post("/api/faces/identify")
def api_faces_identify():
    """上传一张照片，识别是谁。用于底库自检与调试。"""
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"ok": False, "error": "请选择一张照片"}), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in config.ALLOWED_IMG_EXT:
        return jsonify({"ok": False, "error": f"不支持的图片格式: {ext}"}), 400

    in_name = _rand_name(ext)
    in_path = config.UPLOAD_DIR / in_name
    f.save(str(in_path))
    img = imageio_cn.imread(in_path)
    if img is None:
        return jsonify({"ok": False, "error": "图片解析失败"}), 400

    try:
        faces = face_db.extract_faces(img)
    except face_db.FaceModelMissingError as e:
        return jsonify({"ok": False, "error": str(e)}), 503

    if not faces:
        return jsonify({"ok": False, "error": "未检测到人脸"}), 400

    results = []
    for fc in faces:
        pid, nm, score, is_temp = face_db.identify(fc["embedding"])
        results.append(
            {
                "bbox": [round(v, 1) for v in fc["bbox"]],
                "person_id": pid,
                "name": nm,
                "score": score,
                "is_temp": is_temp,
            }
        )
    return jsonify(
        {"ok": True, "count": len(results), "faces": results, "threshold": face_db.MATCH_THRESHOLD}
    )


# ---------------------------------------------------------------- 合规统计 API
@app.get("/api/compliance")
def api_compliance():
    """合规统计：按人聚合的识别情况 + 按人聚合的装备佩戴情况。

    注意：安全帽与口罩是两套独立模型，这里严格分开统计、互不交叉，
    与「两项分别统计、不算总分」的需求一致。

    `person_compliance` 是本轮新增的核心数据 —— 装备已按空间重叠
    归属到人，因此可以答出「张三 3 次未戴安全帽」这种落到人头上的结论，
    而不是只有「本帧 2 个未戴安全帽」。
    """
    try:
        days = max(1, min(365, int(request.args.get("days", 30))))
    except (TypeError, ValueError):
        days = 30
    return jsonify(
        {
            "ok": True,
            "model_ready": face_db.face_model_ready(),
            "seen_summary": face_db.stats_seen_summary(),
            "seen_daily": face_db.stats_seen_daily(14),
            "persons": face_db.stats_persons(50),
            "detect_summary": db.stats_summary(),
            "detect_classes": db.stats_classes(),
            "compliance_overview": face_db.stats_compliance_overview(days),
            "person_compliance": face_db.stats_person_compliance(limit=50, days=days),
        }
    )


@app.get("/api/compliance/seen")
def api_compliance_seen():
    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1
    try:
        size = min(100, max(1, int(request.args.get("size", PAGE_SIZE))))
    except ValueError:
        size = PAGE_SIZE
    person = request.args.get("person") or None
    only_temp = request.args.get("only_temp")
    only_temp = None if only_temp in (None, "") else (only_temp == "1")

    rows, total = face_db.query_seen(page=page, size=size, person=person, only_temp=only_temp)
    return jsonify({"ok": True, "items": rows, "total": total, "page": page, "size": size})


@app.post("/api/compliance/seen/delete")
def api_compliance_seen_delete():
    body = request.get_json(force=True, silent=True) or {}
    ids = body.get("ids") or []
    ids = [int(i) for i in ids if str(i).isdigit()]
    if not ids:
        return jsonify({"ok": False, "error": "请选择要删除的记录"}), 400
    face_db.delete_seen(ids)
    return jsonify({"ok": True})


@app.post("/api/compliance/seen/clear")
def api_compliance_seen_clear():
    n = face_db.clear_seen()
    face_db.reset_temp_registry()
    return jsonify({"ok": True, "cleared": n})


# ---------------------------------------------------------------- 人脸打卡 API
#
# 判定口径（本轮需求刻意收窄）：
#     当日识别到该人员 → 打卡（present）
#     当日未识别到     → 缺勤（absent）
#     当日无任何流水   → 无法判定（no_data）
# 不考虑戴口罩 / 安全帽，也不引入班次与迟到早退。
# 第三条是必须的：服务没开、摄像头没接的日子若判成缺勤，
# 会把全员冤枉一遍，报表立刻失去可信度。
@app.get("/api/attendance/day")
def api_attendance_day():
    """某日全员打卡状态。默认查今天。"""
    day = request.args.get("date") or time.strftime("%Y-%m-%d")
    try:
        day = attendance.validate_day(day)
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    try:
        rows = attendance.daily_status(day)
        summary = attendance.daily_summary(day, rows=rows)
    except Exception as e:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"核算失败: {e}"}), 500
    return jsonify(
        {
            "ok": True,
            "date": day,
            "model_ready": face_db.face_model_ready(),
            "rows": rows,
            "summary": summary,
            "no_data": not summary.get("has_record"),
        }
    )


@app.get("/api/attendance/range")
def api_attendance_range():
    """区间打卡台账。默认最近 7 天。"""
    today = time.strftime("%Y-%m-%d")
    try:
        days = max(1, min(92, int(request.args.get("days", 7))))
    except (TypeError, ValueError):
        days = 7
    end = request.args.get("end") or today
    try:
        end = attendance.validate_day(end)
        t = time.mktime(time.strptime(end, "%Y-%m-%d"))
        start = attendance.validate_day(
            time.strftime("%Y-%m-%d", time.localtime(t - (days - 1) * 86400))
        )
        data = attendance.range_status(start, end)
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"核算失败: {e}"}), 500
    return jsonify({"ok": True, "days_n": days, **data})


@app.get("/api/attendance/trend")
def api_attendance_trend():
    """近 N 天趋势序列（应到 / 打卡 / 缺勤）。"""
    try:
        days = max(1, min(92, int(request.args.get("days", 14))))
    except (TypeError, ValueError):
        days = 14
    try:
        return jsonify({"ok": True, "days_n": days, "items": attendance.trend(days)})
    except Exception as e:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"核算失败: {e}"}), 500


@app.get("/api/attendance/status")
def api_attendance_status():
    """轻量状态查询：给手机端/大屏轮询用，只回今日汇总。"""
    day = time.strftime("%Y-%m-%d")
    try:
        summary = attendance.daily_summary(day)
    except Exception as e:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"核算失败: {e}"}), 500
    return jsonify({"ok": True, "summary": summary, "model_ready": face_db.face_model_ready()})


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
def _local_ip() -> str:
    """尽力获取本机局域网 IP，仅用于启动时打印可访问地址。"""
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def _run_dev_server(use_ssl, ssl_cert, ssl_key):
    """Flask 自带开发服务器（仅在没有 cheroot/waitress 时兜底）。

    ⚠️ **不要用它跑手机端逐帧检测。** 它的 threaded 模式对 SSL 处理很弱：
    客户端正常关闭连接（FIN）时，正忙于推理的服务端来不及 close()，
    于是 CLOSE_WAIT 不断堆积，最终整个进程无响应（浏览器 ERR_TIMED_OUT）。
    实测 threaded=True 时 CLOSE_WAIT 可堆到 17 个、内存涨到 1.4GB。
    """
    if use_ssl:
        app.run(host="0.0.0.0", port=5000, threaded=True, debug=False,
                ssl_context=(ssl_cert, ssl_key))
    else:
        app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)


def _quiet_tls_handshake_noise(server):
    """压掉 cheroot 的 TLS 握手失败噪音。

    背景：服务用自签证书，手机浏览器首次访问会因「证书不受信任」在握手
    阶段直接断开；浏览器反复重试时，cheroot 会把每一次失败都打成一整段
    traceback（还夹带 WinError 10038「在一个非套接字上尝试了一个操作」
    这类善后异常），把真正有用的日志淹没。

    为什么不能用 logging 拦截：这些消息不走 logging，而是 cheroot 内部
    直接调用 server.error_log() 写出去的，所以只能覆盖该方法。
    error_log 的源码注释明确写着「Override this in subclasses as desired」，
    属于官方预留的扩展点，覆盖它是稳妥做法。

    只过滤「已知的、由客户端引起的」握手噪音；其它错误照常输出，
    避免把真正的服务端故障一起吞掉。
    """
    _NOISE = (
        "peer dropped the TLS connection suddenly",   # 客户端拒绝自签证书
        "attempted to speak plain HTTP",              # 用 http:// 打了 https 端口
        "[WinError 10038]",                           # 上述失败后清理 socket 的善后异常
        "The handshake operation timed out",
        "UNEXPECTED_EOF_WHILE_READING",
    )

    def _filtered(msg="", level=20, traceback=False):  # noqa: A002
        text = str(msg)
        if any(k in text for k in _NOISE):
            return
        server.orig_error_log(msg, level=level, traceback=traceback)

    server.orig_error_log = server.error_log
    server.error_log = _filtered


def _harden_windows_tls(ssl_adapter):
    """让 HTTPS 也能在 Windows 浏览器（schannel）下稳定工作。

    ⚠️ 这是「手机能开、电脑打不开」的真凶。

    现象：同一份服务，手机浏览器一切正常，电脑（Edge/Chrome/curl）时好时坏 ——
    实测 8 次请求成功 4 次、失败 4 次，正好一半。

    根因：cheroot 的 BuiltinSSLAdapter 用
    ``ssl.create_default_context(purpose=CLIENT_AUTH)`` 建上下文，它**默认
    启用两种 TLS 后置消息**：会话票据（session ticket）与后置重新协商
    （renegotiation，表现为 curl 的
    ``schannel: remote party requests renegotiation``）。

    Python/OpenSSL 这些实现会照常处理；但 **Windows 的 schannel
    （Edge、Chrome、curl for Windows 都走它）会把握手后突发的
    renegotiation 视为异常并直接重置连接** —— 于是电脑端有一半请求莫名失败。

    修复：换掉上下文，显式关掉这两项。

    为什么禁用票据是安全的：本项目是单机自用服务，没有大量短连接需要
    会话恢复，禁掉只影响性能微优化，换来的是 Windows 端稳定可访问。
    """
    import ssl as _ssl

    try:
        cert = ssl_adapter.certificate
        key = ssl_adapter.private_key

        ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert, key)
        # 关掉会话票据：TLS1.3 下票据是「握手后」异步发送的，
        # 在 Windows 上会被当成不完整的握手。
        ctx.options |= _ssl.OP_NO_TICKET
        # 关掉后置重新协商：schannel 对此零容忍，会直接重置连接。
        if hasattr(_ssl, "OP_NO_RENEGOTIATION"):
            ctx.options |= _ssl.OP_NO_RENEGOTIATION

        ctx.check_hostname = False
        ctx.verify_mode = _ssl.CERT_NONE
        ssl_adapter.context = ctx
        print("  TLS 加固: 已禁用会话票据与后置重新协商（兼容 Windows 浏览器）")
    except Exception as e:  # noqa: BLE001
        # 加固失败不该导致服务起不来 —— 退回 cheroot 的默认上下文，
        # 手机端仍可用，只是电脑端可能偶发失败。
        print(f"  ⚠️  TLS 加固失败（不影响启动）：{e}")


def _run_prod_server(use_ssl, ssl_cert, ssl_key):
    """cheroot 生产级 WSGI 服务器（默认启动方式）。

    cheroot（CherryPy 的服务器内核）相比 Flask 开发服务器的关键优势：

    1. **SSL 原生支持** —— 通过 BuiltinSSLAdapter 接管，不做奇怪的中间层，
       连接生命周期由它自己管理，不会出现 CLOSE_WAIT 堆积；
    2. **固定线程池** —— 线程数可控，不会因请求堆积无限膨胀；
    3. **优雅关闭连接** —— 有明确的超时与收割机制。

    （备选是 waitress，但 waitress 3.x 的 Adjustments 不接受 ssl_context，
      要接 HTTPS 得自己包一层 SSL socket，反而更绕。）
    """
    try:
        from cheroot.wsgi import Server as WSGIServer
    except ImportError:
        print("  ⚠️  未安装 cheroot，回退到 Flask 开发服务器。")
        print("     建议执行： pip install cheroot")
        return _run_dev_server(use_ssl, ssl_cert, ssl_key)

    server = WSGIServer(
        ("0.0.0.0", 5000),
        app,
        numthreads=8,          # 推理已串行，8 个线程足够接管页面请求
        timeout=60,            # 连接闲置超时，及时回收 socket
        shutdown_timeout=5,
        server_name="yolo-safety-detection",
    )
    _quiet_tls_handshake_noise(server)
    if use_ssl:
        from cheroot.ssl.builtin import BuiltinSSLAdapter

        server.ssl_adapter = BuiltinSSLAdapter(ssl_cert, ssl_key)
        _harden_windows_tls(server.ssl_adapter)

    if use_ssl:
        print("  正在启动 HTTPS 服务（cheroot + SSL）…")
        print("  提示：手机首次访问会提示证书不受信任，点「继续前往」即可。")
    else:
        print("  正在启动 HTTP 服务（cheroot）…")
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n  收到中断信号，正在停止…")
    finally:
        server.stop()


if __name__ == "__main__":
    try:
        import torch

        device = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    except Exception:
        device = "CPU"

    # 可选 HTTPS：手机浏览器访问摄像头要求 HTTPS 或 localhost。
    # 设了这两个环境变量就以 HTTPS 启动，否则维持 HTTP。
    ssl_cert = os.environ.get("FLASK_SSL_CERT", "").strip()
    ssl_key = os.environ.get("FLASK_SSL_KEY", "").strip()
    use_ssl = bool(ssl_cert and ssl_key)
    scheme = "https" if use_ssl else "http"

    # APP_DEV_SERVER=1 可强制用 Flask 开发服务器（仅排查问题时用）
    force_dev = os.environ.get("APP_DEV_SERVER", "").strip().lower() in ("1", "true", "yes")

    print("=" * 60)
    print("  安全帽/口罩佩戴检测系统  |  YOLOv8 + Flask")
    print(f"  计算设备: {device}")
    print(f"  本机访问: {scheme}://127.0.0.1:5000")
    print(f"  手机访问: {scheme}://{_local_ip()}:5000")
    if use_ssl:
        print("  模式: HTTPS（手机可直接调用摄像头）")
    else:
        print("  模式: HTTP —— 手机端调摄像头受浏览器限制，")
        print("        需配 HTTPS 或用 chrome://flags 白名单，详见 README「六之四」")
    print(f"  服务器: {'Flask 开发服务器（仅调试）' if force_dev else 'cheroot（生产级）'}")
    print("=" * 60)

    if force_dev:
        _run_dev_server(use_ssl, ssl_cert, ssl_key)
    else:
        _run_prod_server(use_ssl, ssl_cert, ssl_key)
