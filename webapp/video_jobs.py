# -*- coding: utf-8 -*-
"""视频检测任务：后台线程逐帧推理，输出 H.264 结果视频 + 汇总记录"""
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import cv2

import config
import db
import detector

JOBS: dict = {}
_jobs_lock = threading.Lock()

MAX_SIDE = 1280        # 输出视频最大边长
DETAIL_SAMPLE_SEC = 1  # 每隔多少秒抽一帧保存目标明细


def _ffmpeg() -> str:
    """优先使用 imageio-ffmpeg 自带的 ffmpeg，其次系统 PATH。"""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _safe_unlink(path: Path) -> None:
    """安全删除：删除失败（权限/占用/沙箱保护）时静默跳过，绝不让清理动作中断任务。"""
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


def _to_h264(src: Path, dst: Path) -> None:
    """mp4v -> H.264（浏览器 <video> 仅支持 H.264，需要转码）"""
    cmd = [
        _ffmpeg(), "-y", "-loglevel", "error",
        "-i", str(src),
        "-c:v", "libx264", "-preset", "fast", "-crf", "26",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def start_job(video_path: str, kinds, conf: float = 0.25) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        JOBS[job_id] = {
            "status": "running",
            "progress": 0.0,
            "frames": 0,
            "total": 0,
            "record_id": None,
            "error": None,
        }
    t = threading.Thread(target=_process, args=(job_id, video_path, kinds, conf), daemon=True)
    t.start()
    return job_id


def get_status(job_id: str) -> dict:
    with _jobs_lock:
        return dict(JOBS.get(job_id, {}))


def _process(job_id: str, video_path: str, kinds, conf: float) -> None:
    tmp_path = config.RESULT_DIR / f"tmp_{job_id}.mp4"
    out_path = config.RESULT_DIR / f"video_{job_id}.mp4"
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("无法打开视频文件（请确认格式为 mp4/avi/mov/mkv）")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        scale = min(1.0, config_result_maxside() / max(w, h))
        out_w, out_h = max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2)

        writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
        if not writer.isOpened():
            raise RuntimeError("视频编码器初始化失败")

        with _jobs_lock:
            JOBS[job_id]["total"] = total

        counts_total: dict = {}
        details: list = []
        conf_sum, conf_n, t_sum = 0.0, 0, 0.0
        idx, sample_every = 0, max(1, int(fps * DETAIL_SAMPLE_SEC))

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if scale < 1.0:
                frame = cv2.resize(frame, (out_w, out_h))
            dets, counts, annotated, tms = detector.infer_image(frame, kinds, conf=conf)
            writer.write(annotated)
            idx += 1
            t_sum += tms
            conf_sum += sum(d["conf"] for d in dets)
            conf_n += len(dets)
            for k, v in counts.items():
                counts_total[k] = counts_total.get(k, 0) + v
            if (idx - 1) % sample_every == 0 and dets:
                details.extend(dets[:20])
            with _jobs_lock:
                JOBS[job_id]["frames"] = idx
                JOBS[job_id]["progress"] = idx / total if total else 1.0
        cap.release()
        writer.release()

        # 转码为 H.264 供浏览器播放
        _to_h264(tmp_path, out_path)

        cols = detector.counts_to_db(counts_total)
        num = sum(counts_total.values())
        rec_id = db.insert_record(
            source_type="video",
            source_name=Path(video_path).name,
            kinds="+".join(kinds),
            num_objects=num,
            avg_conf=round(conf_sum / conf_n, 3) if conf_n else 0,
            duration_ms=round(t_sum / max(1, idx), 1),
            image_path=None,
            video_path=f"results/{out_path.name}",
            details=details[:300],
            **cols,
        )
        with _jobs_lock:
            JOBS[job_id].update(
                {
                    "status": "done",
                    "progress": 1.0,
                    "record_id": rec_id,
                    "video_url": f"static/results/{out_path.name}",
                    "counts": counts_total,
                    "num_objects": num,
                    "avg_frame_ms": round(t_sum / max(1, idx), 1),
                }
            )
        # 清理中间文件（放在状态更新之后，且删除失败不影响任务结果）
        _safe_unlink(tmp_path)
    except Exception as e:  # noqa: BLE001
        _safe_unlink(tmp_path)
        _safe_unlink(out_path)
        with _jobs_lock:
            JOBS[job_id].update({"status": "error", "error": str(e)})


def config_result_maxside() -> int:
    return MAX_SIDE
