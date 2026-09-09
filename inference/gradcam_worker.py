"""Gradient-enabled worker launched by run_inference.py after segmentation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image

from perthesmetrics_io import LABELS, discover_radiographs, load_label_map, safe_case_id
from src.gradcam import aggregate_case, export_cam, resolve_layer, restore_native_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-folder", "images-dir", "predictions-dir", "source-dir", "output-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cuda", "cpu", "mps"))
    parser.add_argument("--folds", nargs="+", required=True, type=int)
    parser.add_argument("--classes", nargs="+", type=int, choices=range(1, 9), default=list(range(1, 9)))
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--layer", default="auto")
    parser.add_argument("--disable-tta", action="store_true")
    parser.add_argument("--model-sha256", required=True)
    args = parser.parse_args()
    os.environ["nnUNet_compile"] = "false"
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    torch.manual_seed(0)
    np.random.seed(0)
    predictor = nnUNetPredictor(device=torch.device(args.device), use_mirroring=not args.disable_tta,
                                allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(args.model_folder), tuple(args.folds))
    if predictor.dataset_json["labels"] != LABELS or predictor.label_manager.has_regions:
        raise ValueError("Checkpoint labels do not match the eight PerthesMetrics foreground classes.")
    if len(predictor.dataset_json["channel_names"]) != 3:
        raise ValueError("Expected the PerthesMetrics three-channel RGB model.")
    plans, configuration = predictor.plans_manager, predictor.configuration_manager
    if plans.image_reader_writer_class.__name__ != "SimpleITKIO" or len(configuration.patch_size) != 2:
        raise ValueError("Expected a 2D PerthesMetrics model using SimpleITKIO.")
    network = predictor.network.to(args.device).float().eval()
    layer_name, layer = resolve_layer(network, args.layer)
    mirror_axes = tuple(predictor.allowed_mirroring_axes or ()) if not args.disable_tta else ()
    preprocessor = configuration.preprocessor_class(verbose=False)
    classes = list(dict.fromkeys(args.classes))
    sources = {safe_case_id(path): path for path in discover_radiographs(args.source_dir)}
    if args.cases:
        unknown = set(args.cases) - sources.keys()
        if unknown:
            raise ValueError(f"Unknown Grad-CAM cases: {sorted(unknown)}")
        sources = {case: sources[case] for case in dict.fromkeys(args.cases)}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, class_id in LABELS.items():
        if class_id in classes:
            (args.output_dir / name).mkdir()
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True)
    except OSError:
        git = None
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "method": "2D Seg-Grad-CAM",
        "model_sha256": args.model_sha256, "checkpoint": "checkpoint_final.pth", "folds": args.folds,
        "layer": layer_name, "classes": classes, "labels": LABELS, "mirror_axes": mirror_axes,
        "patch_size": list(configuration.patch_size), "tile_step_size": 0.5,
        "target": "Fixed exported prediction mask, resampled to model space; mean pre-softmax class logit",
        "target_window_weights": "mask * Gaussian / overlap_coverage / global_class_pixel_count",
        "aggregation": "ReLU per window/mirror/fold; undo flips; mean mirrors; Gaussian blend windows; mean folds",
        "interpretation": "Aggregated branch attribution, not an exact attribution of the ensemble or a confidence map",
        "normalization": "Raw float32 maps retained; display divided by each case/class maximum after aggregation",
        "precision": "float32 attribution (standard CUDA segmentation uses nnU-Net autocast)",
        "seed": 0, "device": args.device, "python": platform.python_version(),
        "torch": torch.__version__, "nnunetv2": version("nnunetv2"), "numpy": np.__version__,
        "git_commit": git.stdout.strip() if git and git.returncode == 0 else None,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "case_ids": list(sources), "status": "running",
    }
    manifest = args.output_dir / "manifest.json"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "case_status.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["case_id", "class_id", "class_name", "status", "native_pixels", "target_pixels", "gradient_max", "raw_max"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, (case_id, source) in enumerate(sources.items(), 1):
            print(f"Grad-CAM {index}/{len(sources)}: {case_id}; {len(args.folds)} folds, layer {layer_name}", flush=True)
            try:
                prediction = args.predictions_dir / f"{case_id}.nii.gz"
                images = [str(args.images_dir / f"{case_id}_{channel:04d}.nii.gz") for channel in range(3)]
                data, target, properties = preprocessor.run_case(
                    images, str(prediction), plans, configuration, predictor.dataset_json,
                )
                with Image.open(source) as image:
                    rgb = np.asarray(image.convert("RGB"))
                native_labels = load_label_map(prediction)
                accumulated, gradient_max = {}, {c: 0.0 for c in classes}
                for parameters in predictor.list_of_parameters:
                    network.load_state_dict(parameters)
                    maps, norms = aggregate_case(network, layer, data, target[0], classes,
                                                  configuration.patch_size, mirror_axes, args.device)
                    for c, value in maps.items():
                        if c not in accumulated:
                            accumulated[c] = np.zeros_like(value)
                        accumulated[c] += value / len(args.folds)
                        gradient_max[c] = max(gradient_max[c], norms[c])
                for name, c in LABELS.items():
                    if c not in classes:
                        continue
                    pixels = int(np.count_nonzero(native_labels == c))
                    target_pixels = int(np.count_nonzero(target == c))
                    maximum = 0.0
                    if pixels == 0:
                        status = "absent_class"
                    elif c not in accumulated:
                        status = "empty_after_resampling"
                    else:
                        cam = restore_native_map(accumulated[c], properties, plans, configuration)
                        maximum = export_cam(cam, rgb, native_labels == c, args.output_dir / name, case_id)
                        status = "ok" if maximum > 0 else "zero_positive_attribution"
                        if gradient_max[c] == 0:
                            status = "zero_gradient"
                    writer.writerow(dict(zip(fields, [case_id, c, name, status, pixels, target_pixels,
                                                       gradient_max[c], maximum])))
                stream.flush()
            except Exception:
                writer.writerow({"case_id": case_id, "status": "failed"})
                stream.flush()
                metadata["status"] = "failed"
                metadata["failed_case"] = case_id
                manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
                raise
    metadata["status"] = "complete"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Grad-CAM complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
