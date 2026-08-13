from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from inference.run_inference import discover_cases as discover_inference_cases
from training.train_nnunet2d import discover_cases as discover_training_cases


class DatasetValidationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
