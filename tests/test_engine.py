from __future__ import annotations

import base64
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".deps"))
sys.path.insert(0, str(ROOT / "desktop"))

import cv2
import numpy as np

from backend import engine
from backend.app import load_inline_html
from backend.bridge import DesktopBridge


def save_image(path: Path, image: np.ndarray) -> None:
    ok, data = cv2.imencode(path.suffix, image)
    if not ok:
        raise AssertionError("test image encoding failed")
    data.tofile(str(path))


class EngineTests(unittest.TestCase):
    def test_default_template_matches_verified_account_with_outline_padding(self) -> None:
        template = cv2.imdecode(np.fromfile(str(engine.asset_path('watermark_mask_11578066187.png')), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        expected = cv2.resize(template, (1280, 1707), interpolation=cv2.INTER_NEAREST)
        expected = cv2.dilate(expected, np.ones((5, 5), np.uint8))
        actual = engine.build_mask(1280, 1707, {'mode': 'template'})
        np.testing.assert_array_equal(actual, expected)
        legacy = engine.build_mask(1280, 1707, {'mode': 'template', 'templateId': 'legacy'})
        self.assertGreater(np.count_nonzero((actual > 0) & (legacy == 0)), 100)

    def test_unknown_template_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            engine.build_mask(960, 1280, {'mode': 'template', 'templateId': 'unknown'})

    def test_template_previews_follow_selected_account(self) -> None:
        self.assertNotEqual(engine.mask_preview(960, 1280), engine.mask_preview(960, 1280, 'legacy'))

    def test_template_output_keeps_original_and_outside_pixels(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "desktop") as folder:
            root = Path(folder)
            original = np.full((1280, 960, 3), (150, 185, 202), dtype=np.uint8)
            cv2.rectangle(original, (810, 1190), (932, 1260), (255, 255, 255), -1)
            source = root / "照片.png"
            save_image(source, original)
            source_bytes = source.read_bytes()
            output = engine.repair_to_file(source, root / "results", {"mode": "template"})
            actual = engine.read_image(Path(output["path"]))
            mask = engine.build_mask(960, 1280, {"mode": "template"})
            self.assertEqual(actual.shape, original.shape)
            self.assertEqual(np.count_nonzero(actual[mask == 0] != original[mask == 0]), 0)
            self.assertEqual(source.read_bytes(), source_bytes)
            self.assertNotEqual(np.count_nonzero(actual[mask > 0] != original[mask > 0]), 0)
            second = engine.repair_to_file(source, root / "results", {"mode": "template"})
            self.assertNotEqual(output["path"], second["path"])

    def test_landscape_requires_explicit_region(self) -> None:
        with self.assertRaises(ValueError):
            engine.build_mask(800, 600, {"mode": "template"})
        mask = engine.build_mask(800, 600, {"mode": "regions", "regions": [{"x": 0.8, "y": 0.8, "w": 0.15, "h": 0.15}]})
        self.assertEqual(mask[520, 700], 255)
        self.assertEqual(mask[100, 100], 0)

    def test_corrupt_image_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "desktop") as folder:
            path = Path(folder) / "bad.jpg"
            path.write_bytes(b"not an image")
            with self.assertRaises(ValueError):
                engine.inspect(path)

    def test_bridge_batch_finishes_and_preserves_results(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "desktop") as folder:
            root = Path(folder)
            image = np.full((160, 220, 3), 170, dtype=np.uint8)
            source = root / "sample.png"
            save_image(source, image)
            bridge = DesktopBridge()
            bridge._config_path = root / "settings.json"
            payload = bridge.add_paths([str(source), str(source)])
            self.assertEqual(len(payload["items"]), 1)
            image_id = payload["items"][0]["id"]
            job_id = bridge.start_job([{"id": image_id, "selection": {"mode": "regions", "regions": [{"x": 0.75, "y": 0.75, "w": 0.2, "h": 0.2}]}}], str(root / "output"))
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                status = bridge.get_job_status(job_id)
                if status["state"] != "running":
                    break
                time.sleep(0.05)
            self.assertEqual(status["state"], "finished")
            self.assertEqual(status["results"][image_id]["state"], "success")
            self.assertTrue(Path(status["results"][image_id]["path"]).is_file())
            bridge._dispose()

    def test_dragged_file_uses_original_name_for_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "desktop") as folder:
            root = Path(folder)
            image = np.full((128, 96, 3), 180, dtype=np.uint8)
            ok, encoded = cv2.imencode(".png", image)
            self.assertTrue(ok)
            with patch.dict(os.environ, {"CORNER_CLEANER_DATA_DIR": str(root / "data")}):
                bridge = DesktopBridge()
                payload = bridge.import_dropped_file("测试图片.png", base64.b64encode(encoded.tobytes()).decode("ascii"))
                self.assertEqual(payload["items"][0]["name"], "测试图片.png")
                image_id = payload["items"][0]["id"]
                job_id = bridge.start_job(
                    [{"id": image_id, "selection": {"mode": "template"}}],
                    str(root / "output"),
                )
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    status = bridge.get_job_status(job_id)
                    if status["state"] != "running":
                        break
                    time.sleep(0.05)
                self.assertEqual(status["results"][image_id]["state"], "success")
                self.assertEqual(status["results"][image_id]["name"], "测试图片_clean.png")
                bridge._dispose()

    def test_frontend_can_be_loaded_without_http_server(self) -> None:
        document = load_inline_html(ROOT / "desktop" / "dist" / "index.html")
        self.assertIn('id="root"', document)
        self.assertIn('<script type="module">', document)
        self.assertNotIn('src="./assets/', document)


if __name__ == "__main__":
    unittest.main()
