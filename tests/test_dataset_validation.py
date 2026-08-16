from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from inference.run_inference import discover_cases as discover_inference_cases
from training.train_nnunet2d import (
    checkpoint_exists,
    discover_cases as discover_training_cases,
    staged_dataset_exists,
)


class DatasetValidationTests(unittest.TestCase):
    CONFIG = {"dataset": {"id": 1, "name": "PerthesMetrics"}}

    def test_training_requires_three_channels_and_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "imagesTr").mkdir(); (root / "labelsTr").mkdir()
            for channel in range(3):
                (root / "imagesTr" / f"synthetic_case_{channel:04d}.nii.gz").touch()
            (root / "labelsTr" / "synthetic_case.nii.gz").touch()
            self.assertEqual(discover_training_cases(root), ["synthetic_case"])

    def test_inference_rejects_missing_rgb_channel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            images = Path(temporary)
            (images / "case_0000.nii.gz").touch(); (images / "case_0001.nii.gz").touch()
            with self.assertRaisesRegex(ValueError, "RGB channels"):
                discover_inference_cases(images)

    def test_completed_staging_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "nnUNet_raw" / "Dataset001_PerthesMetrics"
            (dataset / "imagesTr").mkdir(parents=True)
            (dataset / "labelsTr").mkdir()
            (dataset / "dataset.json").touch()
            self.assertTrue(staged_dataset_exists(root, self.CONFIG))

    def test_resume_requires_an_actual_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fold = root / "nnUNet_results" / "Dataset001_PerthesMetrics" / "nnUNetTrainer__nnUNetPlans__2d" / "fold_0"
            fold.mkdir(parents=True)
            self.assertFalse(checkpoint_exists(root, self.CONFIG, "nnUNetTrainer", 0))
            (fold / "checkpoint_latest.pth").touch()
            self.assertTrue(checkpoint_exists(root, self.CONFIG, "nnUNetTrainer", 0))


if __name__ == "__main__":
    unittest.main()
