from __future__ import annotations

import tempfile
import unittest
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn

from inference.run_inference import prepare_scratch, resolve_output
from src.gradcam import SegGradCAM, aggregate_case, export_cam, restore_native_map


class TinySegmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(2, 2, 1, bias=False)
        self.head = nn.Conv2d(2, 3, 1, bias=False)
        with torch.no_grad():
            self.features.weight.copy_(torch.eye(2).reshape(2, 2, 1, 1))
            self.head.weight.copy_(torch.tensor([[0., 0.], [1., 0.], [0., 1.]]).reshape(3, 2, 1, 1))

    def forward(self, x):
        return self.head(self.features(x))


class GradCAMTests(unittest.TestCase):
    def setUp(self):
        self.network = TinySegmentation()
        self.image = torch.zeros(1, 2, 6, 10)
        self.image[0, 0, 1:3, 1:4] = 1
        self.image[0, 1, 3:5, 6:9] = 1

    def test_class_targets_and_parameter_randomization(self):
        weights = torch.ones(6, 10) / 60
        baseline = self.network(self.image).detach().clone()
        original = {key: value.clone() for key, value in self.network.state_dict().items()}
        with SegGradCAM(self.network, self.network.features) as cam:
            maps, norms = cam.compute(self.image, {1: weights, 2: weights})
        self.assertGreater(norms[1], 0)
        self.assertGreater(maps[1][1, 1], 0)
        self.assertEqual(maps[1][3, 6], 0)
        self.assertGreater(maps[2][3, 6], 0)
        self.assertFalse(np.array_equal(maps[1], maps[2]))
        torch.testing.assert_close(self.network(self.image), baseline)
        for key, value in self.network.state_dict().items():
            torch.testing.assert_close(value, original[key])
        self.assertTrue(self.network.training)
        self.assertTrue(all(p.requires_grad and p.grad is None for p in self.network.parameters()))
        self.assertEqual(len(self.network.features._forward_hooks), 0)
        with torch.no_grad():
            self.network.head.weight[1].zero_()
        with SegGradCAM(self.network, self.network.features) as cam:
            randomized, _ = cam.compute(self.image, {1: weights})
        self.assertTrue(np.all(randomized[1] == 0))

    def test_hook_cleanup_on_error(self):
        with self.assertRaises(ValueError):
            with SegGradCAM(self.network, self.network.features) as cam:
                cam.compute(self.image, {8: torch.ones(6, 10)})
        self.assertEqual(len(self.network.features._forward_hooks), 0)
        self.assertTrue(all(p.requires_grad for p in self.network.parameters()))

    def test_tiling_mirroring_padding_and_empty_class(self):
        data = self.image[0].numpy()[:, None]
        labels = np.ones((1, 6, 10), dtype=np.int8)
        for patch in ((4, 6), (8, 12)):
            plain, norms = aggregate_case(self.network, self.network.features, data, labels, [1, 2], patch, (), "cpu")
            mirrored, _ = aggregate_case(self.network, self.network.features, data, labels, [1, 2], patch, (0, 1), "cpu")
            self.assertEqual(plain[1].shape, (1, 6, 10))
            self.assertNotIn(2, plain)
            self.assertGreater(norms[1], 0)
            np.testing.assert_allclose(plain[1], mirrored[1], rtol=1e-5, atol=1e-8)
            self.assertTrue(np.isfinite(plain[1]).all())
            self.assertGreater(plain[1][0, 1, 1], 0)
            self.assertEqual(plain[1][0, 3, 6], 0)

    def test_native_geometry_and_exports(self):
        from functools import partial
        from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
        properties = {
            "spacing": [1, 1, 999], "shape_before_cropping": [1, 6, 10],
            "shape_after_cropping_and_before_resampling": [1, 4, 6],
            "bbox_used_for_cropping": [[0, 1], [1, 5], [2, 8]],
        }
        plans = SimpleNamespace(transpose_forward=[2, 0, 1], transpose_backward=[1, 2, 0])
        configuration = SimpleNamespace(spacing=[2, 2], resampling_fn_probabilities=partial(
            resample_data_or_seg_to_shape, is_seg=False, order=1, order_z=0, force_separate_z=False))
        native = restore_native_map(np.ones((1, 2, 3), dtype=np.float32), properties, plans, configuration)
        # SimpleITK geometry is reversed relative to the BMP-backed NIfTI array.
        expected = np.zeros((10, 6), dtype=np.float32)
        expected[2:8, 1:5] = 1
        np.testing.assert_allclose(native, expected)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            export_cam(native, np.zeros((10, 6, 3), dtype=np.uint8), native > 0, folder, "case")
            np.testing.assert_array_equal(np.load(folder / "case_raw.npy"), native)
            with Image.open(folder / "case_overlay.png") as image:
                self.assertEqual(image.size, (6, 10))

    def test_scratch_and_default_outputs_preserve_existing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "keep.txt"
            marker.write_text("keep")
            first, second = prepare_scratch(root), prepare_scratch(root)
            self.assertNotEqual(first, second)
            self.assertEqual(first.parent, root.resolve())
            self.assertEqual(marker.read_text(), "keep")
            initial = resolve_output(root, None)
            self.assertEqual(initial, root / "nnunet_masks")
            initial.mkdir()
            self.assertNotEqual(resolve_output(root, None), initial)
            with self.assertRaises(FileExistsError):
                resolve_output(root, initial)

    def test_worker_with_real_nnunet_loading_and_preprocessing(self):
        import nibabel as nib
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        from inference.gradcam_worker import main
        from perthesmetrics_io import AFFINE, LABELS, write_rgb_nifti

        architecture = {
            "network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
            "arch_kwargs": {
                "n_stages": 3, "features_per_stage": [4, 8, 16],
                "conv_op": "torch.nn.modules.conv.Conv2d", "kernel_sizes": [[3, 3]] * 3,
                "strides": [[1, 1], [2, 2], [2, 2]], "n_conv_per_stage": [1, 1, 1],
                "n_conv_per_stage_decoder": [1, 1], "conv_bias": True,
                "norm_op": "torch.nn.modules.instancenorm.InstanceNorm2d",
                "norm_op_kwargs": {"eps": 1e-5, "affine": True},
                "dropout_op": None, "dropout_op_kwargs": None,
                "nonlin": "torch.nn.LeakyReLU", "nonlin_kwargs": {"inplace": True},
            },
            "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
        }
        configuration = {
            "data_identifier": "test_2d", "preprocessor_name": "DefaultPreprocessor",
            "batch_size": 1, "patch_size": [16, 24], "spacing": [1, 1],
            "normalization_schemes": ["ZScoreNormalization"] * 3, "use_mask_for_norm": [False] * 3,
            "architecture": architecture,
        }
        for kind in ("data", "seg", "probabilities"):
            configuration[f"resampling_fn_{kind}"] = "resample_data_or_seg_to_shape"
            configuration[f"resampling_fn_{kind}_kwargs"] = {"is_seg": kind == "seg", "order": 1,
                                                              "order_z": 0, "force_separate_z": False}
        plans = {
            "dataset_name": "Dataset999_Test", "plans_name": "nnUNetPlans", "image_reader_writer": "SimpleITKIO",
            "transpose_forward": [2, 0, 1], "transpose_backward": [1, 2, 0],
            "foreground_intensity_properties_per_channel": {str(c): {} for c in range(3)},
            "configurations": {"2d": configuration},
        }
        torch.manual_seed(7)
        model = nnUNetTrainer.build_network_architecture(
            architecture["network_class_name"], architecture["arch_kwargs"], architecture["_kw_requires_import"],
            3, 9, enable_deep_supervision=False,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_folder = root / "model"
            model_folder.mkdir()
            for fold in (0, 1):
                folder = model_folder / f"fold_{fold}"
                folder.mkdir()
                torch.save({"network_weights": model.state_dict(), "trainer_name": "nnUNetTrainer",
                            "init_args": {"configuration": "2d"}, "inference_allowed_mirroring_axes": (0, 1)},
                           folder / "checkpoint_final.pth")
            (model_folder / "plans.json").write_text(json.dumps(plans))
            (model_folder / "dataset.json").write_text(json.dumps({"labels": LABELS,
                "channel_names": {"0": "R", "1": "G", "2": "B"}, "file_ending": ".nii.gz"}))
            source = root / "source"
            source.mkdir()
            # Non-square image with black borders exercises transpose, crop, padding, and tiled inference geometry.
            rgb = np.zeros((29, 19, 3), dtype=np.uint8)
            rgb[2:27, 3:17] = np.random.default_rng(7).integers(1, 255, (25, 14, 3), dtype=np.uint8)
            Image.fromarray(rgb).save(source / "synthetic.bmp")
            write_rgb_nifti(source / "synthetic.bmp", root / "images", "synthetic")
            prediction = np.zeros((1, 29, 19), dtype=np.uint8)
            prediction[:, 4:12, 6:10] = 1
            prediction[:, 16:24, 11:15] = 3
            predictions = root / "predictions"
            predictions.mkdir()
            nib.save(nib.Nifti1Image(prediction, AFFINE), predictions / "synthetic.nii.gz")
            argv = ["gradcam_worker", "--model-folder", str(model_folder), "--images-dir", str(root / "images"),
                    "--predictions-dir", str(predictions), "--source-dir", str(source),
                    "--output-dir", str(root / "cam"), "--device", "cpu", "--folds", "0", "1",
                    "--layer", "decoder.stages.0", "--model-sha256", "synthetic"]
            with patch.object(sys, "argv", argv):
                main()
            for name in ("acetabulum", "head"):
                raw = np.load(root / "cam" / name / "synthetic_raw.npy")
                self.assertEqual(raw.shape, (29, 19))
                self.assertTrue(np.isfinite(raw).all())
                self.assertTrue(np.all(raw[:2] == 0))
            self.assertFalse((root / "cam" / "sourcil" / "synthetic_raw.npy").exists())
            manifest = json.loads((root / "cam" / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["folds"], [0, 1])
            np.testing.assert_array_equal(np.asarray(nib.load(predictions / "synthetic.nii.gz").dataobj), prediction)


if __name__ == "__main__":
    unittest.main()
