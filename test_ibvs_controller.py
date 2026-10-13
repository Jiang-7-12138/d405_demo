import unittest

import numpy as np

from ibvs_controller import PointIBVSController


class PointIBVSTests(unittest.TestCase):
    def setUp(self):
        self.controller = PointIBVSController()
        self.identity = np.eye(4)
        self.args = dict(fx=600.0, fy=600.0, cx=320.0, cy=240.0, t_base_camera=self.identity)

    def test_error_moves_tool_feature_toward_target(self):
        command = self.controller.compute((300, 240), (320, 240), 0.2, **self.args)
        self.assertTrue(command.valid)
        self.assertGreater(command.camera_linear_m_s[0], 0)
        self.assertAlmostEqual(command.camera_linear_m_s[1], 0)

    def test_velocity_is_bounded(self):
        controller = PointIBVSController(max_linear_speed_m_s=0.001)
        command = controller.compute((200, 150), (300, 250), 0.2, **self.args)
        self.assertLessEqual(np.linalg.norm(command.base_linear_m_s), 0.001 + 1e-12)

    def test_deadband_returns_zero(self):
        command = self.controller.compute((320, 240), (321, 240), 0.2, **self.args)
        self.assertEqual(command.reason, "within_deadband")
        self.assertEqual(command.base_linear_m_s, (0.0, 0.0, 0.0))

    def test_invalid_depth_fails_closed(self):
        command = self.controller.compute((300, 240), (320, 240), float("nan"), **self.args)
        self.assertFalse(command.valid)
        self.assertEqual(command.reason, "invalid_tool_depth")
        self.assertEqual(command.base_linear_m_s, (0.0, 0.0, 0.0))

    def test_camera_velocity_is_rotated_to_base(self):
        transform = np.eye(4)
        transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        command = self.controller.compute((300, 240), (320, 240), 0.2, **{**self.args, "t_base_camera": transform})
        self.assertAlmostEqual(command.base_linear_m_s[0], 0.0, places=9)
        self.assertGreater(command.base_linear_m_s[1], 0.0)


if __name__ == "__main__":
    unittest.main()
