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
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perthesmetrics_io import LABELS, export_prediction_bmps, stage_bmp_inference_data


DEFAULT_MODEL_FILENAME = "perthesmetrics_nnunet_model.zip"
DEFAULT_MODEL_REPO = "vishalgokani/perthesmetrics-nnunet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PerthesMetrics 2D nnU-Net inference.")
    parser.add_argument("--data-dir", required=True, type=Path, help="Folder containing root radiograph BMP files.")
    parser.add_argument("--scratch-dir", required=True, type=Path, help="Local SSD parent for an isolated run folder.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-zip", type=Path, help="Path to a local nnU-Net model export zip.")
    source.add_argument(
        "--model-repo",
        help=f"Hugging Face model repository ID (released model: {DEFAULT_MODEL_REPO}).",
    )
    parser.add_argument("--model-filename", default=DEFAULT_MODEL_FILENAME)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu", "mps"),
        default="auto",
        help="Inference device (default: auto selects CUDA when usable, otherwise CPU).",
    )
    parser.add_argument("--gpu", help="Optional CUDA_VISIBLE_DEVICES value.")
    parser.add_argument(
        "--output-dir", type=Path,
        help="Optional output folder. Default: <data-dir>/nnunet_masks, with a timestamp if it exists.",
    )
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--folds", nargs="+", type=int, help="Folds for prediction and attribution (default: all available).")
    parser.add_argument("--disable-tta", action="store_true", help="Disable mirroring for prediction and attribution.")
    parser.add_argument("--gradcam", action="store_true", help="Export class-specific 2D Grad-CAM maps and overlays.")
    parser.add_argument("--gradcam-classes", nargs="+", type=int, choices=range(1, 9),
                        help="Foreground class IDs to explain (default: 1 through 8).")
    parser.add_argument("--gradcam-cases", nargs="+", help="Normalized case IDs to explain (default: all cases).")
    parser.add_argument("--gradcam-layer", default="auto",
                        help="Network module name (default: quarter-resolution decoder stage).")
    args = parser.parse_args()
    if not args.gradcam and (args.gradcam_classes or args.gradcam_cases or args.gradcam_layer != "auto"):
        parser.error("Grad-CAM options require --gradcam.")
    return args


def remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def prepare_scratch(path: Path) -> Path:
    path = path.resolve()
    if path.anchor == str(path):
        raise ValueError(f"Refusing to use filesystem root as scratch: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="perthesmetrics_", dir=path)).resolve()


def resolve_output(data_dir: Path, requested: Path | None) -> Path:
    output = requested.resolve() if requested else data_dir / "nnunet_masks"
    if output.exists() and requested is None:
        output = data_dir / f"nnunet_masks_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}.")
    return output


def executable(name: str) -> str:
    # Always prefer entry points installed beside the Python interpreter that
    # is running this script. PATH may contain nnU-Net from another Conda/venv
    # whose PyTorch build is CPU-only.
    prefix = Path(sys.prefix).resolve()
    candidates = (
        prefix / "Scripts" / f"{name}.exe",  # Windows Conda/venv
        prefix / "Scripts" / name,
        prefix / "bin" / name,               # POSIX Conda/venv
        Path(sys.executable).resolve().parent / f"{name}.exe",
        Path(sys.executable).resolve().parent / name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return shutil.which(name) or name


def run(command: list[str], env: dict[str, str]) -> None:
    command = [executable(command[0]), *command[1:]]
    print("\n" + " ".join(f'"{part}"' if " " in part else part for part in command), flush=True)
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


def resolve_device(device: str, gpu: str | None) -> str:
    if device in ("cpu", "mps"):
        print(f"Inference device: {device}", flush=True)
        return device
    if gpu is not None:
        # Set this before importing torch so the validation and nnU-Net child
        # process see the same physical GPU selection.
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    try:
        import torch
    except ImportError as error:
        if device == "auto":
            print("CUDA unavailable because PyTorch is not installed; using CPU inference.", flush=True)
            return "cpu"
        raise RuntimeError(
            "CUDA inference requires PyTorch. Install a CUDA-compatible PyTorch build before "
            "installing inference/requirements.txt."
        ) from error
    if not torch.cuda.is_available():
        if device == "auto":
            print("CUDA is not available to PyTorch; using CPU inference.", flush=True)
            return "cpu"
        raise RuntimeError(
            "--device cuda was requested, but this PyTorch installation cannot access CUDA. "
            "Install the CUDA-compatible PyTorch build for this system or use --device cpu."
        )
    try:
        probe = torch.empty(1, device="cuda")
        torch.cuda.synchronize()
        del probe
    except Exception as error:
        if device == "auto":
            print(f"CUDA could not be initialized ({error}); using CPU inference.", flush=True)
            return "cpu"
        raise RuntimeError(
            "PyTorch reports CUDA support, but a CUDA tensor could not be allocated. "
            "Check the NVIDIA driver, CUDA/PyTorch compatibility, and --gpu selection."
        ) from error
    index = torch.cuda.current_device()
    selected = f"; CUDA_VISIBLE_DEVICES={gpu}" if gpu is not None else ""
    print(
        f"CUDA inference verified: {torch.cuda.get_device_name(index)} "
        f"(logical cuda:{index}; PyTorch {torch.__version__}; CUDA {torch.version.cuda}{selected})",
        flush=True,
    )
    print(
        "nnU-Net preprocessing and mask export use CPU by design; neural-network "
        "prediction and sliding-window accumulation will use CUDA.",
        flush=True,
    )
    return "cuda"


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
    device = resolve_device(args.device, args.gpu)
    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    output = resolve_output(data_dir, args.output_dir)
    print(f"Results will be copied to: {output}", flush=True)
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
        if args.gradcam_cases:
            unknown = set(args.gradcam_cases) - set(case_to_filename)
            if unknown:
                raise ValueError(f"Unknown --gradcam-cases: {sorted(unknown)}")
        model_zip = get_model(args, scratch)
        model_hash = sha256(model_zip)
        dataset_id, model_folder, folds = install_model(model_zip, env)
        if args.folds:
            selected = list(dict.fromkeys(str(fold) for fold in args.folds))
            if not set(selected).issubset(folds):
                raise ValueError(f"Requested folds {selected}; available folds: {folds}")
            folds = selected
        trainer, plans, configuration = model_folder.name.split("__")
        predictions = scratch / "predictions"
        run([
            "nnUNetv2_predict", "-d", dataset_id, "-i", str(local_input), "-o", str(predictions),
            "-f", *folds, "-tr", trainer, "-c", configuration, "-p", plans, "-device", device,
            *(["--disable_tta"] if args.disable_tta else []),
        ], env)
        rendered = scratch / "rendered_masks"
        rows = export_prediction_bmps(predictions, rendered, case_to_filename, LABELS)
        with (rendered / "inference_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["case_id", "source_filename"])
            writer.writeheader(); writer.writerows(rows)
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(), "num_cases": len(case_to_filename),
            "model_sha256": model_hash, "installed_model": model_folder.name, "folds": folds,
            "configuration": configuration, "device": device, "requested_device": args.device,
            "mirroring": not args.disable_tta, "gradcam": args.gradcam,
            "labels": {str(value): name for name, value in LABELS.items()},
            "output_format": "one binary BMP per foreground class and source radiograph",
        }
        (rendered / "inference_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if args.gradcam:
            worker_env = env.copy()
            worker_env["nnUNet_compile"] = "false"
            command = [
                sys.executable, str(Path(__file__).with_name("gradcam_worker.py")),
                "--model-folder", str(model_folder), "--images-dir", str(local_input),
                "--predictions-dir", str(predictions), "--source-dir", str(data_dir),
                "--output-dir", str(rendered / "gradcam"), "--device", device,
                "--folds", *folds, "--layer", args.gradcam_layer, "--model-sha256", model_hash,
            ]
            if args.disable_tta:
                command.append("--disable-tta")
            if args.gradcam_classes:
                command.extend(["--classes", *map(str, args.gradcam_classes)])
            if args.gradcam_cases:
                command.extend(["--cases", *args.gradcam_cases])
            subprocess.run(command, env=worker_env, check=True)
        shutil.copytree(rendered, output)
        print(f"Copied {len(rows)} predictions as class BMP masks to: {output}")
        completed = True
    finally:
        if scratch.exists() and completed and not args.keep_scratch:
            if scratch.parent != args.scratch_dir.resolve() or not scratch.name.startswith("perthesmetrics_"):
                raise RuntimeError(f"Refusing to clean unexpected scratch path: {scratch}")
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
