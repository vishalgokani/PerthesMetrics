from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from inference.run_inference import (
    cached_input_mapping,
    predictions_complete,
    prepare_inputs,
    prepare_scratch,
    resolve_output,
)


class InferenceCacheTests(unittest.TestCase):
    def test_scratch_and_default_outputs_preserve_existing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "keep.txt"
            marker.write_text("keep")
            first, second = prepare_scratch(root), prepare_scratch(root)
            self.assertEqual(first, second)
            self.assertEqual(first.parent, root.resolve())
            self.assertEqual(first.name, "perthesmetrics")
            self.assertEqual(marker.read_text(), "keep")
            initial = resolve_output(root, None)
            self.assertTrue(initial.name.startswith("nnunet_masks_"))
            initial.mkdir()
            self.assertNotEqual(resolve_output(root, None), initial)
            with self.assertRaises(FileExistsError):
                resolve_output(root, initial)

    def test_converted_inputs_are_reused_until_a_source_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            Image.fromarray(np.zeros((5, 7, 3), dtype=np.uint8)).save(source / "case.bmp")
            scratch = prepare_scratch(root / "scratch")

            images, mapping, reused = prepare_inputs(source, scratch)
            self.assertFalse(reused)
            self.assertEqual(mapping, {"case": "case.bmp"})
            self.assertEqual(cached_input_mapping(source, images), mapping)

            same_images, same_mapping, reused = prepare_inputs(source, scratch)
            self.assertTrue(reused)
            self.assertEqual(same_images, images)
            self.assertEqual(same_mapping, mapping)

            Image.fromarray(np.ones((6, 7, 3), dtype=np.uint8)).save(source / "case.bmp")
            _, _, reused = prepare_inputs(source, scratch)
            self.assertFalse(reused)

    def test_complete_predictions_require_exact_case_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            predictions = Path(temporary)
            mapping = {"case_a": "a.bmp", "case_b": "b.bmp"}
            (predictions / "case_a.nii.gz").touch()
            self.assertFalse(predictions_complete(predictions, mapping))
            (predictions / "case_b.nii.gz").touch()
            self.assertTrue(predictions_complete(predictions, mapping))
            (predictions / "unexpected.nii.gz").touch()
            self.assertFalse(predictions_complete(predictions, mapping))


if __name__ == "__main__":
    unittest.main()
