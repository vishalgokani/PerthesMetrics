"""Run PerthesMetrics 2D nnU-Net inference using local SSD scratch space."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perthesmetrics_io import LABELS, export_prediction_bmps, stage_bmp_inference_data


DEFAULT_MODEL_FILENAME = "perthesmetrics_nnunet_model.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PerthesMetrics 2D nnU-Net inference.")
    parser.add_argument("--data-dir", required=True, type=Path, help="Folder containing root radiograph BMP files.")
    parser.add_argument("--scratch-dir", required=True, type=Path, help="Disposable local SSD folder.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-zip", type=Path)
    source.add_argument("--model-repo", help="Hugging Face model repository ID.")
    parser.add_argument("--model-filename", default=DEFAULT_MODEL_FILENAME)
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--gpu", help="Optional CUDA_VISIBLE_DEVICES value.")
    parser.add_argument(
        "--output-dir", type=Path,
        help="Output folder (default: <data-dir>/nnunet_masks). Must not already exist.",
    )
    parser.add_argument("--keep-scratch", action="store_true")
    return parser.parse_args()


def remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def prepare_scratch(path: Path) -> Path:
    path = path.resolve()
    if path.anchor == str(path):
        raise ValueError(f"Refusing to use filesystem root as scratch: {path}")
    if path.exists():
        remove(path)
    path.mkdir(parents=True)
    return path


def executable(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    candidate = Path(sys.executable).resolve().parent / "Scripts" / f"{name}.exe"
    return str(candidate) if candidate.is_file() else name


def run(command: list[str], env: dict[str, str]) -> None:
    command = [executable(command[0]), *command[1:]]
    print("\n" + " ".join(f'"{part}"' if " " in part else part for part in command))
    subprocess.run(command, env=env, check=True)


def discover_cases(images_ts: Path) -> dict[str, list[Path]]:
    if not images_ts.is_dir():
        raise FileNotFoundError(f"Required folder not found: {images_ts}")
    pattern = re.compile(r"^(.*)_(\d{4})\.nii\.gz$")
    cases: dict[str, list[Path]] = {}
    channel_sets: dict[str, set[int]] = {}
    for path in sorted(images_ts.glob("*.nii.gz")):
        match = pattern.match(path.name)
        if not match:
            raise ValueError(f"Invalid nnU-Net filename: {path.name}")
        case, channel = match.group(1), int(match.group(2))
        cases.setdefault(case, []).append(path)
        channel_sets.setdefault(case, set()).add(channel)
    if not cases:
        raise FileNotFoundError(f"No NIfTI inputs found in {images_ts}")
    invalid = {case: sorted(channels) for case, channels in channel_sets.items() if channels != {0, 1, 2}}
    if invalid:
        raise ValueError(f"Every case requires RGB channels 0000, 0001, 0002. Invalid: {list(invalid.items())[:3]}")
    return cases


def get_model(args: argparse.Namespace, scratch: Path) -> Path:
    destination = scratch / "model" / DEFAULT_MODEL_FILENAME
    destination.parent.mkdir(parents=True)
    if args.model_zip:
        source = args.model_zip.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
    else:
        from huggingface_hub import hf_hub_download
        source = Path(hf_hub_download(repo_id=args.model_repo, filename=args.model_filename))
    shutil.copy2(source, destination)
    return destination


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install_model(model_zip: Path, env: dict[str, str]) -> tuple[str, Path, list[str]]:
    run(["nnUNetv2_install_pretrained_model_from_zip", str(model_zip)], env)
    datasets = [path for path in Path(env["nnUNet_results"]).glob("Dataset*") if path.is_dir()]
    if len(datasets) != 1:
        raise ValueError(f"Expected one installed DatasetXXX_Name folder; found {[path.name for path in datasets]}")
    dataset = datasets[0]
    match = re.match(r"^Dataset(\d+)_", dataset.name)
    if not match:
        raise ValueError(f"Release model has noncanonical dataset folder: {dataset.name}")
    models = [path for path in dataset.iterdir() if path.is_dir() and path.name.count("__") == 2]
    if len(models) != 1:
        raise ValueError(f"Expected one trainer model folder in {dataset}")
    model = models[0]
    trainer, plans, configuration = model.name.split("__")
    if configuration != "2d":
        raise ValueError(f"Expected a 2d model, found {configuration}")
    folds = sorted(
        path.name.removeprefix("fold_")
        for path in model.glob("fold_*")
        if path.is_dir() and path.name.removeprefix("fold_").isdigit()
    )
    if not folds:
        raise FileNotFoundError("Installed model contains no fold directories.")
    return str(int(match.group(1))), model, folds


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    output = args.output_dir.resolve() if args.output_dir else data_dir / "nnunet_masks"
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}. Choose a new --output-dir.")
    scratch = prepare_scratch(args.scratch_dir)
    completed = False
    try:
        env = os.environ.copy()
        env.update({
            "nnUNet_raw": str(scratch / "nnUNet_raw"),
            "nnUNet_preprocessed": str(scratch / "nnUNet_preprocessed"),
            "nnUNet_results": str(scratch / "nnUNet_results"),
        })
        for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
            Path(env[key]).mkdir(parents=True)
        if args.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = args.gpu
        local_input = scratch / "input" / "imagesTs"
        case_to_filename = stage_bmp_inference_data(data_dir, local_input)
        model_zip = get_model(args, scratch)
        model_hash = sha256(model_zip)
        dataset_id, model_folder, folds = install_model(model_zip, env)
        trainer, plans, configuration = model_folder.name.split("__")
        predictions = scratch / "predictions"
        run([
            "nnUNetv2_predict", "-d", dataset_id, "-i", str(local_input), "-o", str(predictions),
            "-f", *folds, "-tr", trainer, "-c", configuration, "-p", plans, "-device", args.device,
        ], env)
        rendered = scratch / "rendered_masks"
        rows = export_prediction_bmps(predictions, rendered, case_to_filename, LABELS)
        with (rendered / "inference_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["case_id", "source_filename"])
            writer.writeheader(); writer.writerows(rows)
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(), "num_cases": len(case_to_filename),
            "model_sha256": model_hash, "installed_model": model_folder.name, "folds": folds,
            "configuration": configuration, "device": args.device,
            "labels": {str(value): name for name, value in LABELS.items()},
            "output_format": "one binary BMP per foreground class and source radiograph",
        }
        (rendered / "inference_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        shutil.copytree(rendered, output)
        print(f"Copied {len(rows)} predictions as class BMP masks to: {output}")
        completed = True
    finally:
        if scratch.exists() and completed and not args.keep_scratch:
            remove(scratch)
            print(f"Deleted scratch directory: {scratch}")
        elif scratch.exists():
            print(f"Kept scratch directory for inspection: {scratch}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
