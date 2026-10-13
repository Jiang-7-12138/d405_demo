import unittest

import cv2
import numpy as np

from rail_stereo import RailIntrinsics, classify_painted, depth_from_disparity, measure_classes, reconstruct_pair, unrotate_ccw


def _hsv_patch(image_bgr: np.ndarray, x: int, y: int, w: int, h: int, hsv: tuple[int, int, int]) -> None:
    patch = np.zeros((h, w, 3), np.uint8)
    patch[:, :] = hsv
    image_bgr[y : y + h, x : x + w] = cv2.cvtColor(patch, cv2.COLOR_HSV2BGR)


def _texture(height: int, width: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    image = np.full((height, width), 30, np.uint8)
    for _ in range(120):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        w = int(rng.integers(6, 36))
        h = int(rng.integers(6, 36))
        image[y : y + h, x : x + w] = int(rng.integers(0, 255))
    noise = rng.integers(-12, 13, image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


class RailStereoTests(unittest.TestCase):
    def test_ccw_rotation_roundtrip_matches_opencv(self):
        image = np.zeros((4, 6), np.uint8)
        image[1, 2] = 9
        rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.assertEqual(int(rotated[3, 1]), 9)
        restored = unrotate_ccw(rotated, width=6)
        np.testing.assert_array_equal(restored, image)

    def test_depth_formula(self):
        disparity = np.array([[0.0, 64.0]], dtype=np.float32)
        depth = depth_from_disparity(disparity, focal_px=800.0, baseline_m=0.02)
        self.assertAlmostEqual(float(depth[0, 1]), 0.25, places=5)
        self.assertEqual(float(depth[0, 0]), 0.0)

    def test_vertical_shift_recovers_plane_depth(self):
        height, width = 240, 320
        shift = 32
        focal = 640.0
        baseline_m = 0.02
        expected_z = focal * baseline_m / shift
        image_a = _texture(height, width)
        image_b = np.zeros_like(image_a)
        image_b[:-shift] = image_a[shift:]
        intr = RailIntrinsics(width, height, focal, focal, (width - 1) / 2, (height - 1) / 2, np.zeros(5))
        result = reconstruct_pair(
            image_a,
            image_b,
            intr,
            baseline_m=baseline_m,
            axis="vertical",
            num_disparities=64,
            block_size=5,
            min_depth_m=0.05,
            max_depth_m=2.0,
            stride=2,
        )
        center = result.depth_m[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
        valid = center[center > 0]
        self.assertGreater(valid.size, 0.25 * center.size)
        self.assertAlmostEqual(float(np.median(valid)), expected_z, delta=expected_z * 0.08)

    def test_horizontal_shift_recovers_plane_depth(self):
        height, width = 180, 320
        shift = 32
        focal = 500.0
        baseline_m = 0.03
        expected_z = focal * baseline_m / shift
        image_a = _texture(height, width)
        image_b = np.zeros_like(image_a)
        image_b[:, :-shift] = image_a[:, shift:]
        intr = RailIntrinsics(width, height, focal, focal, (width - 1) / 2, (height - 1) / 2, np.zeros(5))
        result = reconstruct_pair(
            image_a,
            image_b,
            intr,
            baseline_m=baseline_m,
            axis="horizontal",
            num_disparities=64,
            block_size=5,
            min_depth_m=0.05,
            max_depth_m=2.0,
            stride=2,
        )
        center = result.depth_m[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
        valid = center[center > 0]
        self.assertGreater(valid.size, 0.25 * center.size)
        self.assertAlmostEqual(float(np.median(valid)), expected_z, delta=expected_z * 0.08)
        self.assertGreater(result.points_m.shape[0], 100)

    def test_painted_thresholds_label_organs_and_lesion_depth(self):
        image = np.zeros((480, 640, 3), np.uint8)
        _hsv_patch(image, 300, 80, 280, 280, (5, 180, 60))
        _hsv_patch(image, 400, 160, 60, 60, (5, 180, 124))
        _hsv_patch(image, 40, 80, 180, 280, (7, 150, 165))
        _hsv_patch(image, 230, 180, 40, 90, (19, 240, 175))
        _hsv_patch(image, 520, 390, 70, 70, (80, 160, 48))

        masks = classify_painted(image)
        self.assertGreater(int(np.count_nonzero(masks["liver"])), 4000)
        self.assertGreater(int(np.count_nonzero(masks["bowel"])), 1200)
        self.assertGreater(int(np.count_nonzero(masks["omentum"])), 300)
        self.assertGreater(int(np.count_nonzero(masks["gallbladder"])), 250)

        depth = np.zeros(image.shape[:2], np.float32)
        depth[masks["liver"]] = 0.20
        intr = RailIntrinsics(640, 480, 800, 800, 320, 240, np.zeros(5))
        stats, _centers = measure_classes(masks, depth, intr)
        self.assertAlmostEqual(stats["liver"]["median_depth_mm"], 200.0, delta=1.0)
        self.assertEqual(len(stats["liver"]["center_camera_mm"]), 3)
        self.assertIsNone(stats["kidney"]["center_camera_mm"])


if __name__ == "__main__":
    unittest.main()
