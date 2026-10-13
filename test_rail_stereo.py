import unittest

import cv2
import numpy as np

from rail_stereo import RailIntrinsics, depth_from_disparity, reconstruct_pair, unrotate_ccw


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


if __name__ == "__main__":
    unittest.main()
