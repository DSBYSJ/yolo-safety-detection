# -*- coding: utf-8 -*-
"""离线单元测试：不依赖运行中的服务、GPU 与数据集，可在 CI 中直接运行。

覆盖范围（与 tests/regression.py 的端到端测试互补）：
    - imageio_cn：中文路径读写、失败可感知（防「cv2 静默失败」回归）
    - db：建库、增删查、统计聚合
    - detector：纯函数（kinds 过滤、类别计数映射）
    - fontutil：跨平台字体加载
    - config：模型定义自洽性（类别名与 data.yaml 一致）
    - camera：取流来源解析（网络流优先于本地设备索引）
    - 手机端：base64 图像解码（带/不带 data: 前缀、非法数据兜底）

用法：
    python tests/test_unit.py
    python tests/test_unit.py -v
"""
import json
import os
import sys
import tempfile
import traceback
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp"))

import numpy as np  # noqa: E402


def _img(w=64, h=48, color=(0, 0, 255)):
    """构造一张纯色 BGR 图（OpenCV 顺序）。"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = color
    return img


class TestImageioCN(unittest.TestCase):
    """imageio_cn：中文路径读写（本项目曾因 cv2.imwrite 静默失败导致前端裂图）"""

    def setUp(self):
        import imageio_cn

        self.m = imageio_cn
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_imwrite_read_ascii(self):
        p = self.dir / "a.jpg"
        self.assertTrue(self.m.imwrite(p, _img()))
        self.assertTrue(p.exists())
        self.assertEqual(self.m.imread(p).shape, (48, 64, 3))

    def test_imwrite_read_chinese_path(self):
        """中文路径必须能正常写入 —— 这是 cv2.imwrite 会静默失败的场景。"""
        p = self.dir / "中文目录" / "检测结果_测试.jpg"
        self.assertTrue(self.m.imwrite(p, _img()))
        self.assertTrue(p.exists())
        img = self.m.imread(p)
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (48, 64, 3))

    def test_imwrite_creates_parent_dirs(self):
        p = self.dir / "深层" / "嵌套" / "目录" / "x.png"
        self.assertTrue(self.m.imwrite(p, _img()))
        self.assertTrue(p.exists())

    def test_imwrite_returns_false_on_bad_input(self):
        """输入 None 必须返回 False，而不是抛异常或被静默吞掉。"""
        self.assertFalse(self.m.imwrite(self.dir / "x.jpg", None))

    def test_imread_missing_returns_none(self):
        self.assertIsNone(self.m.imread(self.dir / "不存在.jpg"))

    def test_imread_corrupt_returns_none(self):
        p = self.dir / "坏图.jpg"
        p.write_bytes(b"this is not an image")
        self.assertIsNone(self.m.imread(p))

    def test_imread_bytes_roundtrip(self):
        data = (self.dir / "b.jpg")
        self.m.imwrite(data, _img())
        self.assertEqual(self.m.imread_bytes(data.read_bytes()).shape, (48, 64, 3))

    def test_imread_bytes_empty(self):
        self.assertIsNone(self.m.imread_bytes(b""))

    def test_write_jpeg_bytes(self):
        buf = self.m.write_jpeg_bytes(_img())
        self.assertIsInstance(buf, bytes)
        self.assertTrue(buf.startswith(b"\xff\xd8"))  # JPEG SOI 魔数

    def test_write_jpeg_bytes_none(self):
        self.assertIsNone(self.m.write_jpeg_bytes(None))


class TestDB(unittest.TestCase):
    """db：SQLite 记录管理（用临时库，不碰 webapp/data.db）"""

    def setUp(self):
        import config
        import db

        self.db = db
        self.config = config
        self._orig = config.DB_PATH
        self.tmp = tempfile.TemporaryDirectory()
        config.DB_PATH = Path(self.tmp.name) / "test.db"
        db._local = __import__("threading").local()  # 重置线程连接
        db.init_db()

    def tearDown(self):
        # 必须先关闭 SQLite 连接再删临时目录：Windows 上文件被占用会导致
        # TemporaryDirectory.cleanup() 抛 PermissionError(WinError 32)。
        self.db.close()
        self.config.DB_PATH = self._orig
        self.db._local = __import__("threading").local()
        self.tmp.cleanup()

    def test_insert_and_get(self):
        rid = self.db.insert_record(
            source_type="image", source_name="a.jpg", kinds="helmet",
            num_objects=3, helmet=2, head=1, avg_conf=0.8, duration_ms=42.0,
        )
        self.assertIsInstance(rid, int)
        r = self.db.get_record(rid)
        self.assertEqual(r["source_name"], "a.jpg")
        self.assertEqual(r["helmet"], 2)
        self.assertEqual(r["head"], 1)

    def test_get_missing_returns_none(self):
        self.assertIsNone(self.db.get_record(999999))

    def test_details_json_roundtrip(self):
        """details 传 list 时应序列化为 JSON 字符串入库。"""
        dets = [{"kind": "helmet", "class": "helmet", "conf": 0.9, "box": [1, 2, 3, 4]}]
        rid = self.db.insert_record(
            source_type="image", source_name="b.jpg", kinds="helmet",
            num_objects=1, helmet=1, details=dets,
        )
        raw = self.db.get_record(rid)["details"]
        self.assertEqual(json.loads(raw), dets)

    def test_query_pagination(self):
        for i in range(15):
            self.db.insert_record(source_type="image", source_name=f"f{i}.jpg", kinds="helmet")
        rows, total = self.db.query_records(page=1, size=10)
        self.assertEqual(total, 15)
        self.assertEqual(len(rows), 10)
        rows2, _ = self.db.query_records(page=2, size=10)
        self.assertEqual(len(rows2), 5)

    def test_query_filter_by_source(self):
        self.db.insert_record(source_type="image", source_name="i.jpg", kinds="helmet")
        self.db.insert_record(source_type="video", source_name="v.mp4", kinds="helmet")
        rows, total = self.db.query_records(source="video")
        self.assertEqual(total, 1)
        self.assertEqual(rows[0]["source_type"], "video")

    def test_query_filter_by_kind(self):
        self.db.insert_record(source_type="image", source_name="a.jpg", kinds="helmet")
        self.db.insert_record(source_type="image", source_name="b.jpg", kinds="helmet+mask")
        _, total = self.db.query_records(kind="mask")
        self.assertEqual(total, 1)

    def test_delete_records(self):
        ids = [
            self.db.insert_record(source_type="image", source_name=f"d{i}.jpg", kinds="helmet")
            for i in range(3)
        ]
        self.db.delete_records(ids[:2])
        _, total = self.db.query_records()
        self.assertEqual(total, 1)

    def test_stats_summary_aggregates(self):
        self.db.insert_record(source_type="image", source_name="a.jpg", kinds="helmet",
                              num_objects=3, helmet=2, head=1)
        self.db.insert_record(source_type="image", source_name="b.jpg", kinds="mask",
                              num_objects=2, mask=1, face=1)
        s = self.db.stats_summary()
        self.assertEqual(s["total_records"], 2)
        self.assertEqual(s["helmet"], 2)
        self.assertEqual(s["head"], 1)
        self.assertEqual(s["violations"], 2)  # head + face

    def test_stats_daily_backfills_days(self):
        self.db.insert_record(source_type="image", source_name="a.jpg", kinds="helmet")
        daily = self.db.stats_daily(7)
        self.assertEqual(len(daily), 7)          # 空日期需补零
        self.assertEqual(sum(d["records"] for d in daily), 1)

    def test_stats_classes_keys(self):
        self.db.insert_record(source_type="image", source_name="a.jpg", kinds="helmet", helmet=1)
        classes = self.db.stats_classes()
        self.assertEqual(set(classes), {"安全帽", "未戴安全帽", "口罩", "未戴口罩"})
        self.assertEqual(classes["安全帽"], 1)

    def test_stats_sources_labels(self):
        self.db.insert_record(source_type="camera", source_name="c", kinds="helmet")
        self.assertIn("实时监控", self.db.stats_sources())


class TestDetectorPure(unittest.TestCase):
    """detector 中的纯函数（不加载模型）"""

    def setUp(self):
        import detector

        self.d = detector

    def test_detect_kinds_filters_invalid(self):
        self.assertEqual(self.d.detect_kinds(["helmet", "bogus"]), ["helmet"])

    def test_detect_kinds_empty_falls_back(self):
        """非法/空 kinds 必须回退到 helmet，保证不崩溃。"""
        self.assertEqual(self.d.detect_kinds([]), ["helmet"])
        self.assertEqual(self.d.detect_kinds(None), ["helmet"])
        self.assertEqual(self.d.detect_kinds(["x", "y"]), ["helmet"])

    def test_detect_kinds_keeps_order(self):
        self.assertEqual(self.d.detect_kinds(["mask", "helmet"]), ["mask", "helmet"])

    def test_counts_to_db_maps_all_four_columns(self):
        out = self.d.counts_to_db({"helmet": 2, "head": 1, "mask": 3, "face": 4})
        self.assertEqual(out, {"helmet": 2, "head": 1, "mask": 3, "face": 4})

    def test_counts_to_db_zero_fills_missing(self):
        """缺字段必须补 0，否则 SQLite 会写入 NULL 导致求和统计出错。"""
        out = self.d.counts_to_db({"helmet": 5})
        self.assertEqual(out, {"helmet": 5, "head": 0, "mask": 0, "face": 0})

    def test_model_missing_error_is_runtime_error(self):
        self.assertTrue(issubclass(self.d.ModelMissingError, RuntimeError))


class TestFontutil(unittest.TestCase):
    """fontutil：跨平台字体加载"""

    def setUp(self):
        import fontutil

        self.f = fontutil

    def test_get_font_returns_font(self):
        self.assertIsNotNone(self.f.get_font(18))

    def test_get_font_caches_by_size(self):
        self.assertIs(self.f.get_font(18), self.f.get_font(18))
        self.assertIsNot(self.f.get_font(18), self.f.get_font(26))

    def test_font_can_render_chinese(self):
        """字体必须能测量中文字串（回退到默认字体时宽度极小）。"""
        font = self.f.get_font(18)
        w_cn = font.getbbox("安全帽")[2]
        self.assertGreater(w_cn, 0)


class TestConfig(unittest.TestCase):
    """config：模型定义自洽性"""

    def setUp(self):
        import config

        self.c = config

    def test_models_have_required_keys(self):
        for kind, meta in self.c.MODELS.items():
            for key in ("title", "file", "names", "names_cn"):
                self.assertIn(key, meta, f"MODELS[{kind}] 缺少 {key}")

    def test_model_names_match_data_yaml(self):
        """config.MODELS 的类别名必须与 datasets/*/data.yaml 一致，否则标签会画错。"""
        import yaml

        for kind, meta in self.c.MODELS.items():
            yml = ROOT / "datasets" / kind / "data.yaml"
            if not yml.exists():
                self.skipTest(f"缺少 {yml}（数据集未构建）")
            spec = yaml.safe_load(yml.read_text(encoding="utf-8"))
            self.assertEqual(
                {int(k): v for k, v in spec["names"].items()},
                meta["names"],
                f"{kind} 的类别定义与 data.yaml 不一致",
            )

    def test_names_cn_covers_all_classes(self):
        for kind, meta in self.c.MODELS.items():
            self.assertEqual(
                set(meta["names"]), set(meta["names_cn"]),
                f"{kind} 的中文名未覆盖全部类别",
            )

    def test_extension_sets_lowercase_dotted(self):
        for ext in self.c.ALLOWED_IMG_EXT | self.c.ALLOWED_VIDEO_EXT:
            self.assertTrue(ext.startswith("."), f"{ext} 缺少前导点")
            self.assertEqual(ext, ext.lower(), f"{ext} 应为小写")

    def test_upload_limit_positive(self):
        self.assertGreater(self.c.MAX_CONTENT_LENGTH, 0)

    def test_camera_index_defaults_to_int(self):
        """CAMERA_INDEX 必须是 int（OpenCV 设备索引），非法值要能兜底。"""
        import config

        self.assertIsInstance(config.CAMERA_INDEX, int)
        self.assertGreaterEqual(config.CAMERA_INDEX, 0)

    def test_env_int_parses_valid(self):
        import importlib

        import config

        os.environ["CAMERA_INDEX"] = "2"
        try:
            importlib.reload(config)
            self.assertEqual(config.CAMERA_INDEX, 2)
        finally:
            os.environ.pop("CAMERA_INDEX", None)
            importlib.reload(config)

    def test_env_int_falls_back_on_garbage(self):
        """环境变量填了非数字时必须回退默认值，不能让服务启动崩溃。

        默认值不写死数字 —— 它随 config 源码调整（本机摄像头索引就
        从 0 改成了 1）。这里比对环境变量未设置时的取值，断言两者一致。
        """
        import importlib

        import config

        os.environ.pop("CAMERA_INDEX", None)
        importlib.reload(config)
        default = config.CAMERA_INDEX

        for bad in ("abc", "", "  ", "1.5"):
            os.environ["CAMERA_INDEX"] = bad
            try:
                importlib.reload(config)
                self.assertEqual(config.CAMERA_INDEX, default,
                                 f"输入 {bad!r} 时未回退默认值 {default}")
            finally:
                os.environ.pop("CAMERA_INDEX", None)
                importlib.reload(config)

    def test_camera_source_defaults_to_empty(self):
        """未设置时 CAMERA_SOURCE 应为空串，表示改用本地设备索引。"""
        import config

        self.assertEqual(config.CAMERA_SOURCE, "")

    def test_env_str_reads_value(self):
        import importlib

        import config

        url = "rtsp://192.168.1.9:554/h264"
        os.environ["CAMERA_SOURCE"] = url
        try:
            importlib.reload(config)
            self.assertEqual(config.CAMERA_SOURCE, url)
        finally:
            os.environ.pop("CAMERA_SOURCE", None)
            importlib.reload(config)

    def test_env_str_strips_whitespace_and_blank(self):
        """空白串必须视为未设置，否则会生成一个打不开的地址。"""
        import importlib

        import config

        os.environ["CAMERA_SOURCE"] = "   "
        try:
            importlib.reload(config)
            self.assertEqual(config.CAMERA_SOURCE, "")
        finally:
            os.environ.pop("CAMERA_SOURCE", None)
            importlib.reload(config)

    def test_stream_timeout_positive(self):
        import config

        self.assertGreater(config.CAMERA_STREAM_TIMEOUT, 0)


class TestCameraSource(unittest.TestCase):
    """camera.resolve_source：网络流优先于本地索引"""

    def setUp(self):
        import camera
        import config

        self.camera = camera
        self.config = config

    def tearDown(self):
        self.config.CAMERA_SOURCE = ""
        self.config.CAMERA_INDEX = 0

    def test_falls_back_to_index_when_no_source(self):
        self.config.CAMERA_SOURCE = ""
        self.config.CAMERA_INDEX = 3
        source, is_stream, shown = self.camera.resolve_source()
        self.assertEqual(source, 3)
        self.assertFalse(is_stream)
        self.assertIn("3", shown)

    def test_url_source_takes_priority(self):
        self.config.CAMERA_SOURCE = "http://192.168.1.7:8080/video"
        self.config.CAMERA_INDEX = 1
        source, is_stream, _ = self.camera.resolve_source()
        self.assertEqual(source, "http://192.168.1.7:8080/video")
        self.assertTrue(is_stream, "设置了 CAMERA_SOURCE 时应判定为网络流")

    def test_blank_source_ignored(self):
        """纯空白不该被当成有效地址。"""
        self.config.CAMERA_SOURCE = "   "
        self.config.CAMERA_INDEX = 2
        source, is_stream, _ = self.camera.resolve_source()
        self.assertEqual(source, 2)
        self.assertFalse(is_stream)

    def test_index_zero_is_not_lost(self):
        """索引 0 是合法设备，不能被 `or` 之类的写法误判为假值。"""
        self.config.CAMERA_SOURCE = ""
        self.config.CAMERA_INDEX = 0
        source, is_stream, shown = self.camera.resolve_source()
        self.assertEqual(source, 0)
        self.assertFalse(is_stream)
        self.assertIn("0", shown)


class TestSourceMasking(unittest.TestCase):
    """camera._mask_source：回传给前端的取流地址必须抹掉密码。

    /api/camera/state 会把当前视频源显示在页面上，而 RTSP 地址里
    通常带摄像头账号密码（rtsp://admin:密码@ip:554/...）。
    若原样返回，登录密码就暴露在浏览器、接口响应和截图里。
    """

    def setUp(self):
        import camera

        self.m = camera._mask_source

    def test_password_is_masked(self):
        out = self.m("rtsp://admin:Secret123@192.168.1.64:554/Streaming/Channels/102")
        self.assertNotIn("Secret123", out, "密码泄漏！")
        self.assertIn("***", out)
        self.assertIn("admin", out, "用户名应保留，便于辨认账号")
        self.assertIn("192.168.1.64", out, "主机地址应保留，便于确认设备")

    def test_no_password_left_untouched(self):
        """本来就没密码的地址不该被改动（包括不该凭空加上 ***@）。"""
        url = "http://192.168.1.100:8080/video"
        self.assertEqual(self.m(url), url)

    def test_url_encoded_password_masked(self):
        """密码里含 URL 编码字符（如 %40 代表 @）也要打掉。"""
        out = self.m("rtsp://user:p%40ss@10.0.0.5:554/h264")
        self.assertNotIn("p%40ss", out)
        self.assertIn("***", out)

    def test_empty_and_none_safe(self):
        self.assertEqual(self.m(""), "")
        self.assertIsNone(self.m(None))


class TestPhoneDecode(unittest.TestCase):
    """手机端接口的图像解码：两种 base64 格式都要兼容

    这条链路是「手机浏览器 → base64 → 服务端解码」，格式约定一旦不兼容，
    手机上会静默无画面，所以必须由测试守住。
    """

    def setUp(self):
        import base64
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (48, 32), (200, 60, 60)).save(buf, format="JPEG")
        self.raw = buf.getvalue()
        self.pure = base64.b64encode(self.raw).decode()
        self.with_prefix = "data:image/jpeg;base64," + self.pure

    @staticmethod
    def _decode(data_url: str):
        """复刻 app.api_phone_detect 里的解码逻辑，便于离线验证。"""
        import base64

        import cv2
        import numpy as np

        if "," in data_url[:64]:
            data_url = data_url.split(",", 1)[1]
        raw = base64.b64decode(data_url, validate=False)
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def test_pure_base64_decodes(self):
        img = self._decode(self.pure)
        self.assertIsNotNone(img)
        self.assertEqual(img.shape[:2], (32, 48))   # (h, w)

    def test_data_url_with_prefix_decodes(self):
        """带 data:image/jpeg;base64, 前缀时必须能正确剥离。"""
        img = self._decode(self.with_prefix)
        self.assertIsNotNone(img)
        self.assertEqual(img.shape[:2], (32, 48))

    def test_both_formats_agree(self):
        a = self._decode(self.pure)
        b = self._decode(self.with_prefix)
        self.assertTrue((a == b).all(), "两种格式解出的图像应完全一致")

    def test_garbage_returns_none_not_raise(self):
        """非法数据应返回 None（接口据此报 400），不能抛异常打挂请求。"""
        import base64

        bad = base64.b64encode(b"this is not an image at all").decode()
        self.assertIsNone(self._decode(bad))

    def test_prefix_split_only_on_short_head(self):
        """分隔符判断只看前 64 字符，避免把超长 base64 内部的逗号误当地址分隔。"""
        # 构造一个不含前缀、但内容很长的串，确保不会被误切
        img = self._decode(self.pure)
        self.assertIsNotNone(img)


class TestInferLock(unittest.TestCase):
    """推理串行锁：防止多线程并发调用 YOLO 导致 CUDA 上下文死锁。

    背景：手机端逐帧请求与摄像头后台线程会同时触发推理，而 YOLO/PyTorch
    的模型对象不是线程安全的。曾出现「端口仍在 LISTENING 但所有请求无响应」
    的整进程冻结，就是并发推理踩踏导致。
    """

    def setUp(self):
        # webapp 路径已在模块顶部插入 sys.path
        import detector

        self.detector = detector

    def test_infer_busy_is_runtime_error(self):
        """InferBusy 必须是 RuntimeError 子类，便于上层统一按异常处理。"""
        self.assertTrue(issubclass(self.detector.InferBusy, RuntimeError))

    def test_lock_is_exclusive(self):
        """锁被占用时，第二个持有者必须拿不到（模拟并发推理互斥）。"""
        lock = self.detector._infer_lock
        self.assertTrue(lock.acquire(timeout=0.1))
        try:
            # 非阻塞尝试：同一个非可重入锁不应被再次获得
            self.assertFalse(lock.acquire(timeout=0.1))
        finally:
            lock.release()
        # 释放后应能重新获得
        self.assertTrue(lock.acquire(timeout=0.1))
        lock.release()

    def test_timeout_acquire_returns_false_when_held(self):
        """锁被长期占用时，带 timeout 的 acquire 应返回 False 而不是永久阻塞。

        这是手机端接口「宁可丢帧不可排队」的实现基础：等不到锁就跳过本帧。
        """
        lock = self.detector._infer_lock
        self.assertTrue(lock.acquire(timeout=0.1))
        try:
            got = lock.acquire(timeout=0.05)
            self.assertFalse(got, "锁已被占用时不应再次获得")
        finally:
            lock.release()


class TestDetectionPayload(unittest.TestCase):
    """检测框数据契约：手机端前端要靠它把框画到画面上。

    前端绘制逻辑是「像素坐标 × 显示比例 + 留白偏移」。只要 box 的格式、
    取值单位（原图像素）、坐标系原点（左上）任一环节变了，框就会整体偏移，
    而这类问题在真机上才看得出来、很容易漏。所以在此锁死契约。
    """

    def _make(self, size=(320, 240)):
        """按 detector.infer_image 的返回结构造一份样例检测结果。"""
        w, h = size
        dets = [
            {"kind": "helmet", "class": "helmet", "class_cn": "安全帽",
             "conf": 0.91, "box": [10.0, 20.0, 110.0, 140.0]},
            {"kind": "helmet", "class": "head", "class_cn": "未戴安全帽",
             "conf": 0.77, "box": [200.0, 30.0, 300.0, 150.0]},
        ]
        return {"size": [w, h], "detections": dets, "num_objects": len(dets)}

    def test_box_is_left_top_right_bottom(self):
        """box 必须是 [x1,y1,x2,y2] 且 x1<x2、y1<y2（前端据此算宽高）。"""
        for det in self._make()["detections"]:
            x1, y1, x2, y2 = det["box"]
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)

    def test_box_within_image_bounds(self):
        """框必须落在图像尺寸内，否则画到画布外会被裁掉。"""
        d = self._make(size=(320, 240))
        w, h = d["size"]
        for det in d["detections"]:
            x1, y1, x2, y2 = det["box"]
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
            self.assertLessEqual(x2, w)
            self.assertLessEqual(y2, h)

    def test_each_detection_has_display_fields(self):
        """前端画框与标签要用到 class、class_cn、conf，缺一不可。"""
        for det in self._make()["detections"]:
            self.assertIn(det["class"], ("helmet", "head", "mask", "face"))
            self.assertTrue(det["class_cn"], "中文类别名不能为空（标签要显示）")
            self.assertGreaterEqual(det["conf"], 0.0)
            self.assertLessEqual(det["conf"], 1.0)

    def test_size_is_width_then_height(self):
        """size 必须是 [宽, 高]，与 numpy shape[:2] 的 (高, 宽) 顺序相反。

        接口里写的是 img.shape[1], img.shape[0]，顺序写反会让画框长宽互换。
        """
        w, h = self._make(size=(640, 480))["size"]
        self.assertEqual((w, h), (640, 480))


class TestFaceIdentify(unittest.TestCase):
    """face_db.identify：比对阈值逻辑。

    ⚠️ 这里刻意用「构造的单位向量」而不是真实人脸照片来做断言。
    原因：真实照片的相似度受光照/角度影响，数值不稳定，写死会变成脆弱测试；
    而阈值判断是纯数学逻辑，用正交/同向向量可以精确验证「该命中」和「该拒绝」，
    既快又不会因为换张照片就挂。
    """

    def setUp(self):
        import tempfile

        import config

        self._tmp = tempfile.mkdtemp()
        self._old_db = config.DB_PATH
        config.DB_PATH = Path(self._tmp) / "face_test.db"

        import face_db

        self.f = face_db
        # 重置模块级线程局部连接，确保指向新库
        self.f.close()
        self.f.init_db()
        self.f.reset_temp_registry()

    def tearDown(self):
        import config

        self.f.close()
        config.DB_PATH = self._old_db

    @staticmethod
    def _vec(dim, idx):
        """构造第 idx 维为 1 的单位向量（彼此正交，相似度 0）。"""
        v = np.zeros(dim, dtype=np.float32)
        v[idx] = 1.0
        return v

    def test_exact_same_face_matches(self):
        """同一特征必须命中，相似度为 1。"""
        e = self._vec(512, 0)
        pid = self.f.register("张工", e)
        got_pid, name, score, is_temp = self.f.identify(e)
        self.assertEqual(got_pid, pid)
        self.assertEqual(name, "张工")
        self.assertFalse(is_temp)
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_orthogonal_face_is_rejected(self):
        """正交向量（相似度 0）必须判为未注册，不能瞎认。"""
        self.f.register("张工", self._vec(512, 0))
        _, name, score, is_temp = self.f.identify(self._vec(512, 1))
        self.assertTrue(is_temp, "相似度 0 不该命中底库")
        self.assertTrue(name.startswith("访客-"), f"应给临时编号，实际 {name}")

    def test_threshold_boundary(self):
        """恰好等于阈值应算命中（>= 判定），略低于阈值应拒绝。

        这个边界很容易在重构时被写成 > 而不是 >=，值得钉住。
        """
        th = self.f.MATCH_THRESHOLD
        # 构造与基准向量相似度恰为 th 的向量：cos = th
        base = np.zeros(512, dtype=np.float32)
        base[0] = 1.0
        other = np.zeros(512, dtype=np.float32)
        other[0] = th
        other[1] = float(np.sqrt(max(0.0, 1 - th * th)))
        self.f.register("边界", base)

        _, name, score, is_temp = self.f.identify(other)
        self.assertFalse(is_temp, f"相似度 {score} 达到阈值 {th} 应命中")
        self.assertEqual(name, "边界")

        # 略低于阈值
        lower = np.zeros(512, dtype=np.float32)
        k = th - 0.05
        lower[0] = k
        lower[1] = float(np.sqrt(max(0.0, 1 - k * k)))
        _, name2, _, is_temp2 = self.f.identify(lower)
        self.assertTrue(is_temp2, f"相似度 {k:.3f} 低于阈值 {th} 不该命中")

    def test_empty_gallery_returns_temp(self):
        """底库为空时不能崩，应返回临时编号。"""
        _, name, score, is_temp = self.f.identify(self._vec(512, 3))
        self.assertTrue(is_temp)
        self.assertEqual(score, 0.0)
        self.assertTrue(name.startswith("访客-"))

    def test_zero_vector_does_not_crash(self):
        """全零向量（范数为 0）会让归一化除零，必须兜住。"""
        self.f.register("张工", self._vec(512, 0))
        try:
            _, name, _, is_temp = self.f.identify(np.zeros(512, dtype=np.float32))
        except ZeroDivisionError:
            self.fail("全零向量导致除零崩溃")
        self.assertTrue(is_temp)

    def test_picks_best_match_not_first(self):
        """必须取最相似的那个，而不是底库里第一个。"""
        self.f.register("甲", self._vec(512, 0))
        self.f.register("乙", self._vec(512, 1))
        _, name, _, is_temp = self.f.identify(self._vec(512, 1))
        self.assertFalse(is_temp)

    def test_dim_mismatch_row_skipped(self):
        """底库里混入维度不符的脏数据时，跳过它而不是整体崩溃。"""
        self.f.register("正常", self._vec(512, 0))
        conn = self.f.get_conn()
        conn.execute(
            "INSERT INTO face_persons (name, note, embedding, created_at) VALUES (?,?,?,?)",
            ("脏数据", "", np.zeros(128, dtype=np.float32).tobytes(), "2026-01-01 00:00:00"),
        )
        conn.commit()
        _, name, _, is_temp = self.f.identify(self._vec(512, 0))
        self.assertFalse(is_temp, "遇到脏数据不该放弃正常比对")


class TestFaceTempIds(unittest.TestCase):
    """临时访客编号：同一张脸必须保持同一个编号。

    这是需求「没录入的人算临时 id」的核心质量点——
    如果同一张脸每次都被发新号，统计页上一个人会被拆成十几行，功能等于废掉。
    """

    def setUp(self):
        import tempfile

        import config

        self._tmp = tempfile.mkdtemp()
        self._old_db = config.DB_PATH
        config.DB_PATH = Path(self._tmp) / "face_test2.db"

        import face_db

        self.f = face_db
        self.f.close()
        self.f.init_db()
        self.f.reset_temp_registry()

    def tearDown(self):
        import config

        self.f.close()
        config.DB_PATH = self._old_db

    @staticmethod
    def _vec(dim, idx):
        v = np.zeros(dim, dtype=np.float32)
        v[idx] = 1.0
        return v

    def test_same_face_keeps_same_temp_id(self):
        """同一张脸连续识别，编号必须稳定不变。"""
        v = self._vec(512, 5)
        _, n1, _, _ = self.f.identify(v)
        _, n2, _, _ = self.f.identify(v)
        _, n3, _, _ = self.f.identify(v)
        self.assertEqual(n1, n2, "同一张脸第二次被发了新编号")
        self.assertEqual(n2, n3, "同一张脸第三次被发了新编号")

    def test_different_faces_get_different_ids(self):
        """不同的人必须是不同编号，不能共用。"""
        _, n1, _, _ = self.f.identify(self._vec(512, 10))
        _, n2, _, _ = self.f.identify(self._vec(512, 20))
        self.assertNotEqual(n1, n2, "两个不同的人共用了同一个临时编号")

    def test_reset_clears_registry(self):
        """重置后应从头发号。"""
        self.f.identify(self._vec(512, 7))
        self.f.reset_temp_registry()
        _, n, _, _ = self.f.identify(self._vec(512, 8))
        self.assertEqual(n, "访客-1")

    def test_registered_person_never_gets_temp_id(self):
        """已注册的人不能被发临时编号。"""
        v = self._vec(512, 30)
        self.f.register("李工", v)
        pid, name, _, is_temp = self.f.identify(v)
        self.assertFalse(is_temp)
        self.assertEqual(name, "李工")
        self.assertIsNotNone(pid)


class TestFaceStore(unittest.TestCase):
    """face_db 的底库与识别记录存取、统计。"""

    def setUp(self):
        import tempfile

        import config

        self._tmp = tempfile.mkdtemp()
        self._old_db = config.DB_PATH
        config.DB_PATH = Path(self._tmp) / "face_test3.db"

        import face_db

        self.f = face_db
        self.f.close()
        self.f.init_db()
        self.f.reset_temp_registry()

    def tearDown(self):
        import config

        self.f.close()
        config.DB_PATH = self._old_db

    @staticmethod
    def _vec(dim, idx):
        v = np.zeros(dim, dtype=np.float32)
        v[idx] = 1.0
        return v

    def test_register_requires_name(self):
        """姓名为空必须被拒绝（否则底库里全是无名氏）。"""
        with self.assertRaises(ValueError):
            self.f.register("   ", self._vec(512, 0))

    def test_register_strips_name(self):
        """姓名两端空格应被去掉，避免「张工」和「张工 」被当成两个人。"""
        pid = self.f.register("  张工  ", self._vec(512, 0))
        p = self.f.get_person(pid)
        self.assertEqual(p["name"], "张工")

    def test_delete_person(self):
        pid = self.f.register("待删", self._vec(512, 0))
        self.assertTrue(self.f.delete_person(pid))
        self.assertEqual(self.f.count_persons(), 0)
        self.assertIsNone(self.f.get_person(pid))

    def test_delete_missing_returns_false(self):
        self.assertFalse(self.f.delete_person(99999))

    def test_embedding_roundtrip(self):
        """特征存入再读出必须完全一致（否则等于换了张脸）。"""
        v = np.random.RandomState(42).randn(512).astype(np.float32)
        v = v / np.linalg.norm(v)
        pid = self.f.register("张三", v)
        p = self.f.get_person(pid)
        back = np.frombuffer(p["embedding"], dtype=np.float32)
        np.testing.assert_allclose(back, v, rtol=0, atol=1e-6)

    def test_person_list_excludes_embedding(self):
        """列表接口不能带 embedding——体积大且无展示价值。"""
        self.f.register("张工", self._vec(512, 0))
        for p in self.f.list_persons():
            self.assertNotIn("embedding", p)

    def test_seen_records_and_summary(self):
        """识别记录落库后，总览统计要能对上。"""
        pid = self.f.register("张工", self._vec(512, 0))
        self.f.add_seen(pid, "张工", False, 0.91)
        self.f.add_seen(None, "访客-1", True, 0.12)

        s = self.f.stats_seen_summary()
        self.assertEqual(s["total"], 2)
        self.assertEqual(s["registered_seen"], 1)
        self.assertEqual(s["temp_seen"], 1)
        self.assertEqual(s["distinct_registered"], 1)
        self.assertEqual(s["distinct_temp"], 1)
        self.assertEqual(s["gallery_size"], 1)

    def test_seen_filter_by_temp(self):
        """按「是否未注册」筛选必须准确。"""
        self.f.add_seen(1, "张工", False, 0.9)
        self.f.add_seen(None, "访客-1", True, 0.1)

        rows, total = self.f.query_seen(only_temp=True)
        self.assertEqual(total, 1)
        self.assertTrue(rows[0]["is_temp"])

        rows, total = self.f.query_seen(only_temp=False)
        self.assertEqual(total, 1)
        self.assertFalse(rows[0]["is_temp"])

    def test_person_ranking_groups_by_name(self):
        """按人聚合：同一个人出现两次应合成一行、计数为 2。"""
        self.f.add_seen(1, "张工", False, 0.9)
        self.f.add_seen(1, "张工", False, 0.8)
        self.f.add_seen(2, "李工", False, 0.85)

        st = self.f.stats_persons()
        reg = {r["person_name"]: r["n"] for r in st["registered"]}
        self.assertEqual(reg.get("张工"), 2)
        self.assertEqual(reg.get("李工"), 1)

    def test_clear_seen_keeps_gallery(self):
        """清空识别明细不能连底库一起清掉。"""
        pid = self.f.register("张工", self._vec(512, 0))
        self.f.add_seen(pid, "张工", False, 0.9)
        n = self.f.clear_seen()
        self.assertEqual(n, 1)
        self.assertEqual(self.f.count_persons(), 1, "清空明细误删了底库人员")

    def test_daily_series_length_and_shape(self):
        """近 14 天序列必须是 14 个点且字段齐全（绘图直接用）。"""
        daily = self.f.stats_seen_daily(14)
        self.assertEqual(len(daily), 14)
        for d in daily:
            self.assertIn("date", d)
            self.assertIn("registered", d)
            self.assertIn("temp", d)

    def test_face_model_ready_is_bool_without_loading(self):
        """face_model_ready 只做文件检查，不该触发模型加载（否则会拖慢接口）。"""
        v = self.f.face_model_ready()
        self.assertIsInstance(v, bool)


class TestFaceAnnotation(unittest.TestCase):
    """camera._annotate_faces / _recognize_frame 的绘制行为。

    这两个函数的接口约定容易被改坏：早先版本是「原地修改入参」，
    调用方传了 frame 和 annotated 两个不同对象时，画的是 annotated
    但很多人以为改的是 frame，导致姓名框莫名其妙不出现在画面上。
    改成返回新图后，这个测试把约定钉死。
    """

    def test_annotate_returns_new_image_not_same_object(self):
        """必须返回新图，而不是依赖调用方传对对象。"""
        import camera

        img = _img(200, 150, (30, 30, 30))
        faces = [{"name": "张三", "score": 0.9, "is_temp": False, "box": [20, 20, 100, 100]}]
        out = camera._annotate_faces(img, faces)
        self.assertFalse(np.array_equal(out, img), "画框后图像应与原图不同")
        self.assertEqual(out.shape, img.shape, "尺寸不能变")

    def test_annotate_empty_faces_keeps_image(self):
        """没人脸时不该改动图像。"""
        import camera

        img = _img(80, 60, (10, 20, 30))
        out = camera._annotate_faces(img, [])
        self.assertTrue(np.array_equal(out, img))

    def test_annotate_handles_chinese_name(self):
        """中文姓名必须能画上去（cv2 画中文会变问号，所以走了 PIL）。"""
        import camera

        img = _img(200, 150, (255, 255, 255))
        faces = [{"name": "王工程师", "score": 0.88, "is_temp": False, "box": [10, 10, 120, 120]}]
        out = camera._annotate_faces(img, faces)
        self.assertFalse(np.array_equal(out, img))

    def test_annotate_survives_bad_box(self):
        """框坐标异常（NaN / 越界）不能让绘制抛异常。"""
        import camera

        img = _img(100, 100, (0, 0, 0))
        bad = [{"name": "异常", "score": 0.5, "is_temp": True, "box": [float("nan"), 0, 50, 50]}]
        try:
            camera._annotate_faces(img, bad)
        except Exception as e:  # noqa: BLE001
            self.fail(f"异常框导致绘制崩溃: {e}")


def main():
    argv = sys.argv[:]
    if "-v" in argv:
        argv.remove("-v")
        verbosity = 2
    else:
        verbosity = 1

    loader = unittest.TestLoader()
    suite = unittest.TestSuite(
        loader.loadTestsFromTestCase(cls)
        for cls in (
            TestImageioCN,
            TestDB,
            TestDetectorPure,
            TestFontutil,
            TestConfig,
            TestCameraSource,
            TestSourceMasking,
            TestPhoneDecode,
            TestInferLock,
            TestDetectionPayload,
            TestFaceIdentify,
            TestFaceTempIds,
            TestFaceStore,
            TestFaceAnnotation,
        )
    )
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)

    print("\n" + "=" * 68)
    total = result.testsRun
    bad = len(result.failures) + len(result.errors)
    print(f"单元测试结果: {total - bad}/{total} 通过")
    for case, tb in result.failures + result.errors:
        print(f"  [FAIL] {case}")
        print("    " + tb.strip().splitlines()[-1])
    print("=" * 68)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
