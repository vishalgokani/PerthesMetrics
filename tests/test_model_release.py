from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from model_release.prepare_model_package import CHANNELS, DATASET, LABELS, MODEL, validate_archive
from tools.build_publication_results import canonical_dataset_metadata


class ModelReleaseTests(unittest.TestCase):
    def test_publication_metadata_uses_canonical_dataset_identity(self) -> None:
        metadata = canonical_dataset_metadata(
            {"name": "legacy", "description": "legacy", "region_class_order": {"background": 0}}
        )
        self.assertEqual(metadata["name"], DATASET)
        self.assertNotIn("region_class_order", metadata)

    def test_canonical_archive_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "model.zip"
            root = f"{DATASET}/{MODEL}/"
            dataset = {
                "name": DATASET,
                "channel_names": CHANNELS,
                "labels": LABELS,
                "file_ending": ".nii.gz",
            }
            plans = {"dataset_name": DATASET, "configurations": {"2d": {}}}
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(root + "dataset.json", json.dumps(dataset))
                archive.writestr(root + "plans.json", json.dumps(plans))
                for fold in range(5):
                    archive.writestr(root + f"fold_{fold}/checkpoint_final.pth", b"test")
            details = validate_archive(archive_path)
            self.assertEqual(details["folds"], [0, 1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
