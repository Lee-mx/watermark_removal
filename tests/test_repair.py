"""Edge continuity, conservative fallback, and pixel-preservation regressions."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / ".deps"), str(ROOT / "desktop")]
import cv2
import numpy as np
from backend.repair import repair_pixels, _directional_values


class RepairTests(unittest.TestCase):
    def test_diagonal_edge_error_is_lower_than_telea(self):
        ys, xs = np.indices((240, 320))
        gray = np.clip(125 + (xs + ys - 280) * 15, 25, 225).astype(np.uint8)
        clean = np.repeat(gray[:, :, None], 3, axis=2)
        mask = np.zeros(gray.shape, np.uint8)
        mask[100:140, 125:195] = 255
        damaged = clean.copy()
        damaged[mask > 0] = 255
        old = repair_pixels(damaged, mask, method="telea")
        new = repair_pixels(damaged, mask)
        old_error = np.abs(old.astype(float) - clean)[mask > 0].mean()
        new_error = np.abs(new.astype(float) - clean)[mask > 0].mean()
        self.assertLess(new_error, old_error * .4)
        np.testing.assert_array_equal(new[mask == 0], damaged[mask == 0])

    def test_flat_surface_falls_back_and_keeps_alpha(self):
        image = np.full((120, 160, 4), 135, np.uint8)
        image[:, :, 3] = np.arange(160, dtype=np.uint8)[None, :]
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[35:65, 55:95] = 255
        image[mask > 0, :3] = 240
        new = repair_pixels(image, mask)
        old = repair_pixels(image, mask, method="telea")
        np.testing.assert_array_equal(new, old)
        np.testing.assert_array_equal(new[:, :, 3], image[:, :, 3])

    def test_large_sparse_sampling_has_no_opencv_dimension_limit(self):
        image = np.full((400, 400), 125, np.uint8)
        mask = np.zeros(image.shape, np.uint8)
        mask[90:290, 90:290] = 255
        ys, xs = np.where(mask > 0)
        valid, values = _directional_values(image, mask, ys, xs, np.array([1., 0.]), 210)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(values, 125)

    def test_border_mask_and_grayscale_preserve_outside(self):
        image = np.tile(np.arange(160, dtype=np.uint8), (120, 1))
        mask = np.zeros(image.shape, np.uint8)
        mask[-30:, -40:] = 255
        result = repair_pixels(image, mask)
        self.assertEqual(result.shape, image.shape)
        np.testing.assert_array_equal(result[mask == 0], image[mask == 0])

    def test_adaptive_rectangle_rebuilds_smooth_gradient_without_a_color_block(self):
        yy, xx = np.indices((180, 260))
        clean = np.stack((20 + .1 * xx + .08 * yy,
                          12 + .04 * xx + .06 * yy,
                          30 + .09 * xx + .05 * yy), axis=2).astype(np.uint8)
        mask = np.zeros(clean.shape[:2], np.uint8)
        mask[115:155, 100:240] = 255
        damaged = clean.copy()
        damaged[mask > 0] = 240
        result = repair_pixels(damaged, mask, method="adaptive")
        np.testing.assert_array_equal(result[mask == 0], damaged[mask == 0])
        self.assertLess(np.abs(result.astype(np.int16) - clean.astype(np.int16))[mask > 0].mean(), 3)

    def test_adaptive_uses_previous_algorithm_on_textured_background(self):
        rng = np.random.default_rng(8)
        damaged = rng.integers(0, 256, (130, 180, 3), dtype=np.uint8)
        mask = np.zeros(damaged.shape[:2], np.uint8)
        mask[45:80, 60:135] = 255
        adaptive = repair_pixels(damaged, mask, method="adaptive")
        previous = repair_pixels(damaged, mask, method="edge-aware")
        np.testing.assert_array_equal(adaptive, previous)


if __name__ == "__main__":
    unittest.main()
