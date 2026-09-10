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

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perthesmetrics_io import (
    LABELS,
    discover_radiographs,
    export_prediction_bmps,
    safe_case_id,
    stage_bmp_inference_data,
    write_conversion_manifest,
)


DEFAULT_MODEL_FILENAME = "perthesmetrics_nnunet_model.zip"
DEFAULT_MODEL_REPO = "vishalgokani/perthesmetrics-nnunet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PerthesMetrics 2D nnU-Net inference.")
    parser.add_argument("--data-dir", required=True, type=Path, help="Folder containing root radiograph BMP files.")
    parser.add_argument(
        "--scratch-dir", required=True, type=Path,
        help="Local SSD parent for the reusable perthesmetrics workspace.",
    )
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
        help="Optional output folder. Default: <data-dir>/nnunet_masks_<UTC timestamp>.",
    )
    parser.add_argument("--folds", nargs="+", type=int, help="Folds for prediction (default: all available).")
    parser.add_argument("--disable-tta", action="store_true", help="Disable mirroring for prediction.")
    args = parser.parse_args()
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
    scratch = (path / "perthesmetrics").resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    return scratch


def resolve_output(data_dir: Path, requested: Path | None) -> Path:
    if requested:
        output = requested.resolve()
    else:
        name = f"nnunet_masks_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        output = data_dir / name
        sequence = 1
        while output.exists():
            output = data_dir / f"{name}_{sequence:02d}"
            sequence += 1
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}.")
    return output


def cached_input_mapping(data_dir: Path, images_ts: Path) -> dict[str, str] | None:
    """Return the cached case mapping only when it matches all current inputs."""
    manifest_path = images_ts.parent / "input_conversion_manifest.csv"
    if not manifest_path.is_file() or not images_ts.is_dir():
        return None
    try:
        with manifest_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        sources = discover_radiographs(data_dir)
        if len(rows) != len(sources):
            return None
        by_filename = {row["source_filename"]: row for row in rows}
        if set(by_filename) != {source.name for source in sources}:
            return None
        mapping: dict[str, str] = {}
        for source in sources:
            row = by_filename[source.name]
            case_id = safe_case_id(source)
            if row.get("case_id") != case_id:
                return None
            stat = source.stat()
            fingerprinted = bool(row.get("source_size") and row.get("source_mtime_ns"))
            if fingerprinted:
                if int(row["source_size"]) != stat.st_size or int(row["source_mtime_ns"]) != stat.st_mtime_ns:
                    return None
            else:
                with Image.open(source) as image:
                    if int(row["width"]) != image.width or int(row["height"]) != image.height:
                        return None
            expected = {images_ts / f"{case_id}_{channel:04d}.nii.gz" for channel in range(3)}
            if not all(path.is_file() for path in expected):
                return None
            mapping[case_id] = source.name
        if len(list(images_ts.glob("*.nii.gz"))) != 3 * len(mapping):
            return None
        return mapping
    except (KeyError, OSError, TypeError, ValueError):
        return None


def prepare_inputs(data_dir: Path, scratch: Path) -> tuple[Path, dict[str, str], bool]:
    images_ts = scratch / "input" / "imagesTs"
    mapping = cached_input_mapping(data_dir, images_ts)
    if mapping is not None:
        manifest_path = images_ts.parent / "input_conversion_manifest.csv"
        with manifest_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if rows and not all(row.get("source_size") and row.get("source_mtime_ns") for row in rows):
            sources = {source.name: source for source in discover_radiographs(data_dir)}
            for row in rows:
                stat = sources[row["source_filename"]].stat()
                row["source_size"] = stat.st_size
                row["source_mtime_ns"] = stat.st_mtime_ns
            write_conversion_manifest(manifest_path, rows)
        return images_ts, mapping, True

    staging = scratch / "input_staging"
    if staging.exists():
        remove(staging)
    mapping = stage_bmp_inference_data(data_dir, staging / "imagesTs")
    cached = scratch / "input"
    if cached.exists():
        remove(cached)
    staging.rename(cached)
    return cached / "imagesTs", mapping, False


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
    destination.parent.mkdir(parents=True, exist_ok=True)
    if args.model_zip:
        source = args.model_zip.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
    else:
        from huggingface_hub import hf_hub_download
        source = Path(hf_hub_download(repo_id=args.model_repo, filename=args.model_filename))
    if destination.is_file() and sha256(destination) == sha256(source):
        print(f"Reusing cached model archive: {destination}", flush=True)
        return destination
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


def install_model(
    model_zip: Path, env: dict[str, str], *, reuse_existing: bool = False,
) -> tuple[str, Path, list[str]]:
    datasets = [path for path in Path(env["nnUNet_results"]).glob("Dataset*") if path.is_dir()]
    if not (reuse_existing and datasets):
        run(["nnUNetv2_install_pretrained_model_from_zip", str(model_zip)], env)
        datasets = [path for path in Path(env["nnUNet_results"]).glob("Dataset*") if path.is_dir()]
    else:
        print("Reusing the installed nnU-Net model from the matching prior run.", flush=True)
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


def predictions_complete(predictions: Path, case_to_filename: dict[str, str]) -> bool:
    found = {path.name.removesuffix(".nii.gz") for path in predictions.glob("*.nii.gz")}
    return found == set(case_to_filename)


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device, args.gpu)
    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    output = resolve_output(data_dir, args.output_dir)
    print(f"Results will be copied to: {output}", flush=True)
    scratch = prepare_scratch(args.scratch_dir)
    print(f"Reusable scratch workspace: {scratch}", flush=True)
    runtime = scratch / "runtime"
    runtime.mkdir(exist_ok=True)
    try:
        env = os.environ.copy()
        env.update({
            "nnUNet_raw": str(runtime / "nnUNet_raw"),
            "nnUNet_preprocessed": str(runtime / "nnUNet_preprocessed"),
            "nnUNet_results": str(runtime / "nnUNet_results"),
        })
        for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
            Path(env[key]).mkdir(parents=True, exist_ok=True)
        if args.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = args.gpu
        local_input, case_to_filename, reused_inputs = prepare_inputs(data_dir, scratch)
        if reused_inputs:
            print(f"Reusing {len(case_to_filename)} cached NIfTI inputs from: {local_input}", flush=True)
        else:
            print(f"Converted and cached {len(case_to_filename)} radiographs in: {local_input}", flush=True)
        model_zip = get_model(args, runtime)
        model_hash = sha256(model_zip)
        prior_manifest_path = runtime / "rendered_masks" / "inference_manifest.json"
        prior_model_hash = None
        if prior_manifest_path.is_file():
            try:
                prior_model_hash = json.loads(prior_manifest_path.read_text(encoding="utf-8")).get("model_sha256")
            except (OSError, ValueError):
                pass
        reuse_runtime = prior_model_hash == model_hash
        if not reuse_runtime:
            for generated in (runtime / "predictions", runtime / "rendered_masks", Path(env["nnUNet_results"])):
                if generated.exists():
                    remove(generated)
            Path(env["nnUNet_results"]).mkdir(parents=True)
        dataset_id, model_folder, folds = install_model(model_zip, env, reuse_existing=reuse_runtime)
        if args.folds:
            selected = list(dict.fromkeys(str(fold) for fold in args.folds))
            if not set(selected).issubset(folds):
                raise ValueError(f"Requested folds {selected}; available folds: {folds}")
            folds = selected
        trainer, plans, configuration = model_folder.name.split("__")
        predictions = runtime / "predictions"
        if predictions_complete(predictions, case_to_filename):
            print(f"Reusing {len(case_to_filename)} complete segmentation predictions: {predictions}", flush=True)
        else:
            run([
                "nnUNetv2_predict", "-d", dataset_id, "-i", str(local_input), "-o", str(predictions),
                "-f", *folds, "-tr", trainer, "-c", configuration, "-p", plans, "-device", device,
                *(["--disable_tta"] if args.disable_tta else []),
            ], env)
        rendered = runtime / "rendered_masks"
        if rendered.exists():
            remove(rendered)
        rows = export_prediction_bmps(predictions, rendered, case_to_filename, LABELS)
        with (rendered / "inference_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["case_id", "source_filename"])
            writer.writeheader(); writer.writerows(rows)
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(), "num_cases": len(case_to_filename),
            "model_sha256": model_hash, "installed_model": model_folder.name, "folds": folds,
            "configuration": configuration, "device": device, "requested_device": args.device,
            "mirroring": not args.disable_tta,
            "labels": {str(value): name for name, value in LABELS.items()},
            "output_format": "one binary BMP per foreground class and source radiograph",
        }
        (rendered / "inference_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        shutil.copytree(rendered, output)
        print(f"Copied {len(rows)} predictions as class BMP masks to: {output}")
    finally:
        if runtime.exists():
            print(f"Kept scratch workspace: {scratch}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
