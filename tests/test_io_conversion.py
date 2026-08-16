from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from perthesmetrics_io import (
    LABELS,
    export_prediction_bmps,
    stage_bmp_inference_data,
    stage_bmp_training_data,
)


class IoConversionTests(unittest.TestCase):
    def test_training_bmps_become_rgb_channels_and_multiclass_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / "source"; target = root / "target"
            source.mkdir()
            image = np.full((4, 5, 3), [10, 20, 30], dtype=np.uint8)
            Image.fromarray(image).save(source / "Patient_1_AP_L.bmp")
            for name, value in LABELS.items():
                if value == 0:
                    continue
                folder = source / "masks" / name; folder.mkdir(parents=True)
                mask = np.zeros((4, 5), dtype=np.uint8)
                if value == 1:
                    mask[1, 1] = 255
                if value == 8:  # Later class wins at an overlapping pixel.
                    mask[1, 1:3] = 255
                Image.fromarray(mask).save(folder / "Patient_1_AP_L.bmp")
            cases = stage_bmp_training_data(source, target)
            self.assertEqual(len(cases), 1)
            case = cases[0]
            for channel, expected in enumerate((10, 20, 30)):
                data = np.asarray(nib.load(target / "imagesTr" / f"{case}_{channel:04d}.nii.gz").dataobj)
                self.assertTrue(np.all(data == expected))
            label = np.squeeze(np.asarray(nib.load(target / "labelsTr" / f"{case}.nii.gz").dataobj))
            self.assertEqual(int(label[1, 1]), 8)
            self.assertEqual(int(label[1, 2]), 8)

    def test_inference_prediction_becomes_eight_binary_bmps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / "source"; images = root / "imagesTs"
            predictions = root / "predictions"; output = root / "output"
            source.mkdir(); predictions.mkdir()
            Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(source / "case.bmp")
            mapping = stage_bmp_inference_data(source, images)
            labels = np.arange(12, dtype=np.uint8).reshape(1, 3, 4) % 9
            nib.save(nib.Nifti1Image(labels, np.eye(4)), predictions / "case.nii.gz")
            rows = export_prediction_bmps(predictions, output, mapping)
            self.assertEqual(rows[0]["source_filename"], "case.bmp")
            for name, value in LABELS.items():
                if value:
                    mask = np.asarray(Image.open(output / name / "case.bmp"))
                    self.assertTrue(set(np.unique(mask)).issubset({0, 255}))


if __name__ == "__main__":
    unittest.main()
