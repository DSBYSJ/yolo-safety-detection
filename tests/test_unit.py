# -*- coding: utf-8 -*-
"""离线单元测试：不依赖运行中的服务、GPU 与数据集，可在 CI 中直接运行。

覆盖范围（与 tests/regression.py 的端到端测试互补）：
    - imageio_cn：中文路径读写、失败可感知（防「cv2 静默失败」回归）
    - db：建库、增删查、统计聚合
    - detector：纯函数（kinds 过滤、类别计数映射）
    - fontutil：跨平台字体加载
    - config：模型定义自洽性（类别名与 data.yaml 一致）
    - camera：取流来源解析（网络流优先于本地设备索引）

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
        """环境变量填了非数字时必须回退默认值，不能让服务启动崩溃。"""
        import importlib

        import config

        for bad in ("abc", "", "  ", "1.5"):
            os.environ["CAMERA_INDEX"] = bad
            try:
                importlib.reload(config)
                self.assertEqual(config.CAMERA_INDEX, 0, f"输入 {bad!r} 时未回退默认值")
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
