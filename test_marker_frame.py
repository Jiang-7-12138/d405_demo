import unittest
from pathlib import Path

import cv2
import numpy as np

from marker_frame import _image_step, detect_marker_frame
from segmenter import PhantomMaterialSegmenter


class MarkerFrameTests(unittest.TestCase):
    def test_top_tag_axes_point_left_down_and_up(self):
        cap = cv2.VideoCapture(str(Path("captures/endoscope_20261016_180212.avi")))
        self.assertTrue(cap.isOpened())
        try:
            ok, frame = cap.read()
        finally:
            cap.release()
        self.assertTrue(ok)
        pose = detect_marker_frame(frame)
        self.assertTrue(pose.valid)
        self.assertEqual(pose.marker_id, 5)
        self.assertIsNotNone(pose.origin_px)
        x, y = pose.origin_px
        self.assertGreater(x, 540)
        self.assertLess(x, 680)
        self.assertGreater(y, 240)
        self.assertLess(y, 340)

        step_x = _image_step(pose.origin_camera_m, pose.x_axis, 700.0)
        step_y = _image_step(pose.origin_camera_m, pose.y_axis, 700.0)
        step_z = _image_step(pose.origin_camera_m, pose.z_axis, 700.0)
        self.assertLess(step_z[1], 0.0)
        self.assertLess(step_x[0], 0.0)
        self.assertGreater(abs(step_x[0]), abs(step_x[1]) * 3.0)
        self.assertGreater(step_y[1], 0.0)
        self.assertGreater(step_y[1], abs(step_y[0]))
        self.assertAlmostEqual(float(np.cross(pose.x_axis, pose.y_axis) @ pose.z_axis), 1.0, places=3)

        masks = PhantomMaterialSegmenter(profile="endoscope-deep").predict_all(frame)
        gallbladder = masks["gallbladder"]
        self.assertGreater(int(np.count_nonzero(gallbladder)), 200)
        ys, xs = np.nonzero(gallbladder)
        distance = np.hypot(xs - x, ys - y)
        self.assertGreater(float(distance.min()), 40.0)


if __name__ == "__main__":
    unittest.main()
