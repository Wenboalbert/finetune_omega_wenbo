import math
import unittest

import torch

from ftlib.heterocam_loss import camera_geometry_loss
from ftlib.heterocam_metrics import packet_metrics


def rotation_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def make_extrinsics(centers, rotations):
    output = []
    for center, rotation in zip(centers, rotations):
        E = torch.eye(4)
        E[:3, :3] = rotation
        E[:3, 3] = -(rotation @ center)
        output.append(E)
    return torch.stack(output)


class HeterocamGeometryTest(unittest.TestCase):
    def setUp(self):
        self.centers = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [4.0, 0.0, 0.5],
                [0.0, 5.0, 1.0],
                [4.0, 5.0, 2.0],
                [15.0, 8.0, 12.0],
            ]
        )
        self.rotations = torch.stack([rotation_z(0.1 * i) for i in range(5)])
        self.gt_E = make_extrinsics(self.centers, self.rotations)
        self.K = torch.tensor(
            [[[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]]
        ).repeat(5, 1, 1)
        self.anchor_mask = torch.tensor([True, True, True, True, False])

    def test_global_similarity_is_invariant(self):
        scale = 3.2
        Q = rotation_z(0.7)
        translation = torch.tensor([13.0, -7.0, 2.5])
        predicted_centers = ((self.centers - translation) @ Q) / scale
        predicted_rotations = self.rotations @ Q
        pred_E = make_extrinsics(predicted_centers, predicted_rotations)
        total, components = camera_geometry_loss(
            pred_E, self.K, self.gt_E, self.K, self.anchor_mask, (480, 640)
        )
        # float32 acos/log operations leave a tiny numerical residual.
        self.assertLess(float(total), 1e-4)
        metrics = packet_metrics(
            pred_E, self.K, self.gt_E, self.K, self.anchor_mask, (480, 640)
        )
        self.assertLess(metrics["relative_rotation_deg"], 0.03)
        self.assertLess(metrics["baseline_direction_deg"], 0.03)
        self.assertLess(metrics["cross_relative_rotation_deg"], 0.03)
        self.assertLess(metrics["cross_baseline_mape"], 1e-4)

    def test_bad_target_increases_cross_camera_error(self):
        bad_centers = self.centers.clone()
        bad_centers[-1] += torch.tensor([20.0, -10.0, 5.0])
        bad_E = make_extrinsics(bad_centers, self.rotations)
        good = packet_metrics(
            self.gt_E, self.K, self.gt_E, self.K, self.anchor_mask, (480, 640)
        )
        bad = packet_metrics(
            bad_E, self.K, self.gt_E, self.K, self.anchor_mask, (480, 640)
        )
        self.assertGreater(bad["cross_baseline_mape"], good["cross_baseline_mape"] + 20.0)
        self.assertGreater(bad["baseline_direction_deg"], good["baseline_direction_deg"] + 5.0)


if __name__ == "__main__":
    unittest.main()
