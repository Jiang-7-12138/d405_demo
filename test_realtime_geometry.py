import unittest

import numpy as np

from realtime_geometry import build_vision_result


class Intrinsics:
    fx = 500.0
    fy = 500.0
    cx = 50.0
    cy = 50.0


class RealtimeGeometryTests(unittest.TestCase):
    def test_lesion_mask_is_pixel_exact_and_excluded_from_obstacles(self):
        depth = np.full((100, 100), 0.5, dtype=np.float32)
        yy, xx = np.ogrid[:100, :100]
        lesion = (xx - 50) ** 2 + (yy - 50) ** 2 <= 10 ** 2
        depth[lesion] = 0.48
        organ = np.zeros_like(lesion)
        organ[20:40, 20:40] = True
        masks = {"lesion": lesion, "liver": organ}
        transform = np.eye(4)
        transform[:3, 3] = [0.1, -0.2, 0.3]

        result = build_vision_result(123.0, masks, depth, Intrinsics(), 0.3, 0.7, transform)
        self.assertTrue(result["lesion"]["detected"])
        self.assertNotIn("lesion", result["obstacles"])
        lesion_state = result["lesion"]["instances"][0]
        self.assertEqual(lesion_state["center_px"], [50.0, 50.0])
        self.assertTrue(lesion_state["valid_3d"])
        self.assertEqual(lesion_state["bbox_px"], {"x": 40, "y": 40, "width": 21, "height": 21})
        rle = lesion_state["mask_rle"]
        decoded = np.repeat(np.arange(len(rle["counts"])) % 2, rle["counts"]).astype(bool)
        decoded = decoded.reshape(rle["size"])
        np.testing.assert_array_equal(decoded, lesion[40:61, 40:61])
        self.assertTrue(result["obstacles"]["liver"]["valid"])
        self.assertIn("base", result["obstacles"]["liver"]["bounds"])
        self.assertNotIn("cut_pose_camera", str(result))

    def test_invalid_depth_is_not_reported_as_missing_obstacle(self):
        mask = np.ones((10, 10), dtype=bool)
        result = build_vision_result(1.0, {"liver": mask}, np.zeros((10, 10), np.float32), Intrinsics(), .1, 1.0, None)
        self.assertFalse(result["obstacles"]["liver"]["valid"])
        self.assertFalse(result["lesion"]["detected"])


if __name__ == "__main__":
    unittest.main()
