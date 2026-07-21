import os
import tempfile
import unittest
from pathlib import Path

import torch

from ftlib.heterocam_train import (
    _hardlink_checkpoint,
    _passes_gate,
    _save_training_state,
)


class TrainingStateTest(unittest.TestCase):
    def test_resume_state_only_contains_trainable_delta(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
        for parameter in model[0].parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad]
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        config = {"model": {"init_checkpoint": "/base/model.pt"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            _save_training_state(path, model, optimizer, scheduler, 7, config, {}, {})
            state = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(state["format"], "omega_trainable_delta_v1")
        self.assertEqual(set(state["trainable_model"]), {"1.bias", "1.weight"})
        self.assertNotIn("model", state)

    def test_checkpoint_alias_is_a_hardlink(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "candidate.pt"
            destination = Path(directory) / "passed.pt"
            source.write_bytes(b"immutable checkpoint")
            _hardlink_checkpoint(source, destination)
            self.assertEqual(source.read_bytes(), destination.read_bytes())
            self.assertEqual(os.stat(source).st_ino, os.stat(destination).st_ino)

    def test_gate_uses_target_and_cross_camera_metrics(self):
        baseline = {
            "cross_baseline_mape": 10.0,
            "cross_baseline_direction_deg": 8.0,
            "cross_relative_rotation_deg": 5.0,
            "target_hfov_mae_deg": 4.0,
        }
        improved = {
            "cross_baseline_mape": 8.0,
            "cross_baseline_direction_deg": 7.0,
            "cross_relative_rotation_deg": 5.1,
            "target_hfov_mae_deg": 4.1,
        }
        self.assertTrue(_passes_gate(improved, baseline, 1.05))
        improved["target_hfov_mae_deg"] = 4.3
        self.assertFalse(_passes_gate(improved, baseline, 1.05))


if __name__ == "__main__":
    unittest.main()
