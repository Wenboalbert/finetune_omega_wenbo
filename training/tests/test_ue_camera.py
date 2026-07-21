import math
import unittest

import numpy as np

from ftlib.ue_camera import (
    camera_center_from_extrinsic,
    intrinsics_from_ue_hfov,
    ue_pose_to_opencv_extrinsic,
)


class UECameraTest(unittest.TestCase):
    def test_identity_rotation_and_center_basis(self):
        E = ue_pose_to_opencv_extrinsic((100.0, 200.0, 300.0), 0.0, 0.0, 0.0)
        np.testing.assert_allclose(E[:3, :3], np.eye(3), atol=1e-8)
        np.testing.assert_allclose(camera_center_from_extrinsic(E), [2.0, -3.0, 1.0], atol=1e-8)

    def test_general_rotation_is_valid(self):
        E = ue_pose_to_opencv_extrinsic((0.0, 0.0, 0.0), -31.0, 72.0, 4.0)
        np.testing.assert_allclose(E[:3, :3].T @ E[:3, :3], np.eye(3), atol=1e-8)
        self.assertAlmostEqual(float(np.linalg.det(E[:3, :3])), 1.0, places=8)

    def test_hfov_intrinsics(self):
        K = intrinsics_from_ue_hfov(1920, 1080, 90.0)
        recovered = math.degrees(2.0 * math.atan(1920 / (2.0 * K[0, 0])))
        self.assertAlmostEqual(recovered, 90.0, places=8)
        self.assertEqual(K[0, 2], 960.0)
        self.assertEqual(K[1, 2], 540.0)


if __name__ == "__main__":
    unittest.main()
