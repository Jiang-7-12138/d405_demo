import unittest
from pathlib import Path

import cv2
import numpy as np

from endoscope_viewer import LesionCenterHold, MaskHold, recognize_frame
from visualization import mask_interior_point
from segmenter import PhantomMaterialSegmenter


class EndoscopeVesselTests(unittest.TestCase):
    def test_blue_vessel_beside_gallbladder_is_kept(self):
        path = Path("captures/endoscope_20261015_234113.png")
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        masks = PhantomMaterialSegmenter(profile="endoscope").predict_all(image)
        self.assertGreater(int(np.count_nonzero(masks["gallbladder"])), 2000)
        self.assertGreater(int(np.count_nonzero(masks["vessel"])), 200)

    def test_lesion_center_ignores_small_jitter_and_follows_a_real_move(self):
        hold = LesionCenterHold(deadband_px=4.0, history=5, hold_frames=3)
        self.assertEqual(hold.update((100, 80)), (100, 80))
        for _ in range(4):
            self.assertEqual(hold.update((102, 81)), (100, 80))
        self.assertEqual(hold.update(None), (100, 80))
        for _ in range(5):
            moved = hold.update((130, 80))
        self.assertEqual(moved, (130, 80))

    def test_bowel_outline_ignores_one_frame_flicker_and_label_stays_on_the_organ(self):
        base = np.zeros((80, 120), dtype=bool)
        base[20:60, 10:110] = True
        hold = MaskHold(history=5, hold_frames=2)
        stable = None
        for i in range(5):
            flicker = base.copy()
            if i == 4:
                flicker[20:60, 100:110] = False
            if i == 2:
                flicker[0:12, 40:70] = True
            stable = hold.update(flicker)
        self.assertIsNotNone(stable)
        self.assertGreater(int(np.count_nonzero(stable[:, 100:110])), 20)
        self.assertEqual(int(np.count_nonzero(stable[:12, 40:70])), 0)

        cropped = np.zeros((100, 160), dtype=bool)
        cropped[30:70, :80] = True
        point = mask_interior_point(cropped, edge_margin=36)
        self.assertIsNotNone(point)
        self.assertGreater(point[0], 36)

    def test_liver_and_bowel_masks_do_not_share_pixels(self):
        path = Path("captures/endoscope_20261015_234113.png")
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        masks = PhantomMaterialSegmenter(profile="endoscope-bright").predict_all(image)
        self.assertEqual(int(np.count_nonzero(masks["liver"] & masks["bowel"])), 0)
        self.assertGreater(int(np.count_nonzero(masks["liver"])), 4000)
        self.assertGreater(int(np.count_nonzero(masks["bowel"])), 1500)
        self.assertGreater(int(np.count_nonzero(masks["lesion"])), 200)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        red = (hue <= 16) | (hue >= 170)
        pale_colon = red & (sat <= 125) & (val >= 158)
        liver_core = red & (sat >= 130) & (val >= 40) & (val <= 210)
        near_liver = cv2.dilate(liver_core.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
        # The pale square inside the liver stays with the liver. The colon
        # outside that organ must not.
        self.assertLess(int(np.count_nonzero(masks["liver"] & pale_colon & ~near_liver)), 500)

    def test_dark_lesion_center_stays_on_the_dome(self):
        cap = cv2.VideoCapture(str(Path("captures/endoscope_20261016_124621.avi")))
        self.assertTrue(cap.isOpened())
        segmenter = PhantomMaterialSegmenter(profile="endoscope")
        hold = LesionCenterHold(deadband_px=12.0)
        centers = []
        try:
            for index in range(60, 301, 40):
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                self.assertTrue(ok)
                _, _, center = recognize_frame(frame, segmenter, hold)
                self.assertIsNotNone(center)
                centers.append(center)
        finally:
            cap.release()
        self.assertEqual(len(set(centers)), 1)
        x, y = centers[0]
        self.assertGreater(x, 180)
        self.assertLess(x, 230)
        self.assertGreater(y, 200)
        self.assertLess(y, 255)

    def test_dark_saturated_dome_center_stays_put(self):
        cap = cv2.VideoCapture(str(Path("captures/endoscope_20261016_141145.avi")))
        self.assertTrue(cap.isOpened())
        segmenter = PhantomMaterialSegmenter(profile="endoscope-dark")
        hold = LesionCenterHold(deadband_px=12.0)
        centers = []
        try:
            for index in range(0, 90, 15):
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                self.assertTrue(ok)
                _, found, center = recognize_frame(frame, segmenter, hold)
                self.assertIn("lesion", found)
                self.assertIsNotNone(center)
                centers.append(center)
        finally:
            cap.release()
        self.assertEqual(len(set(centers)), 1)
        x, y = centers[0]
        self.assertGreater(x, 150)
        self.assertLess(x, 210)
        self.assertGreater(y, 200)
        self.assertLess(y, 270)


if __name__ == "__main__":
    unittest.main()
