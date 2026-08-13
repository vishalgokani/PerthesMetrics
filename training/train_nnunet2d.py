"""Stage data locally and train patient-grouped PerthesMetrics 2D nnU-Net models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml

try:
    from .patient_splits import load_patient_map, make_grouped_splits, resolve_groups, write_split_artifacts
except ImportError:  # Direct execution: python training/train_nnunet2d.py
    from patient_splits import load_patient_map, make_grouped_splits, resolve_groups, write_split_artifacts


EPOCH_TRAINERS = {
    1: "nnUNetTrainer_1epoch", 5: "nnUNetTrainer_5epochs", 10: "nnUNetTrainer_10epochs",
    20: "nnUNetTrainer_20epochs", 50: "nnUNetTrainer_50epochs", 100: "nnUNetTrainer_100epochs",
    250: "nnUNetTrainer_250epochs", 500: "nnUNetTrainer_500epochs", 750: "nnUNetTrainer_750epochs",
    1000: "nnUNetTrainer", 2000: "nnUNetTrainer_2000epochs", 4000: "nnUNetTrainer_4000epochs",
    8000: "nnUNetTrainer_8000epochs",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PerthesMetrics 2D nnU-Net with patient-grouped folds.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("perthesmetrics_nnunet2d.yaml"))
    parser.add_argument("--data-dir", required=True, type=Path, help="Network folder containing imagesTr and labelsTr.")
    parser.add_argument("--scratch-dir", required=True, type=Path, help="Disposable local SSD folder.")
    parser.add_argument("--patient-map", type=Path, help="Optional CSV with case_id,patient_id columns.")
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"))
    parser.add_argument("--gpu", help="Optional CUDA_VISIBLE_DEVICES value, for example 0.")
    parser.add_argument("--resume", action="store_true", help="Resume checkpoints in the same scratch directory.")
    parser.add_argument("--skip-planning", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-training", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--keep-scratch", action="store_true", help="Keep scratch after successful completion.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for section in ("dataset", "channels", "labels", "training"):
        if section not in config:
            raise ValueError(f"Missing config section: {section}")
    labels = {str(key): int(value) for key, value in config["labels"].items()}
    if labels.get("background") != 0 or sorted(labels.values()) != list(range(9)):
        raise ValueError("PerthesMetrics labels must be unique integers 0 through 8 with background=0.")
    channels = {int(key): str(value) for key, value in config["channels"].items()}
    if channels != {0: "R", 1: "G", 2: "B"}:
        raise ValueError("PerthesMetrics requires RGB channels 0:R, 1:G, 2:B.")
    if config["training"].get("configuration") != "2d":
        raise ValueError("PerthesMetrics training configuration must be 2d.")
    return config


def dataset_folder(config: dict) -> str:
    return f"Dataset{int(config['dataset']['id']):03d}_{config['dataset']['name']}"


def remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def prepare_scratch(path: Path, resume: bool) -> Path:
    path = path.resolve()
    if path.anchor == str(path):
        raise ValueError(f"Refusing to use filesystem root as scratch: {path}")
    if path.exists() and not resume:
        remove(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def discover_cases(data_dir: Path) -> list[str]:
    images, labels = data_dir / "imagesTr", data_dir / "labelsTr"
    if not images.is_dir() or not labels.is_dir():
        raise FileNotFoundError("The data directory must contain imagesTr and labelsTr.")
    channel_pattern = re.compile(r"^(.*)_(\d{4})\.nii\.gz$")
    channels: dict[str, set[int]] = {}
    malformed = []
    for path in images.glob("*.nii.gz"):
        match = channel_pattern.match(path.name)
        if not match:
            malformed.append(path.name)
            continue
        channels.setdefault(match.group(1), set()).add(int(match.group(2)))
    if malformed:
        raise ValueError(f"Malformed image filenames, including: {malformed[:3]}")
    cases = sorted(channels)
    if not cases:
        raise ValueError("No training cases found.")
    bad_channels = {case: sorted(values) for case, values in channels.items() if values != {0, 1, 2}}
    if bad_channels:
        sample = list(bad_channels.items())[:3]
        raise ValueError(f"Every case must have exactly RGB channels 0000, 0001, 0002. Invalid: {sample}")
    label_cases = {path.name.removesuffix(".nii.gz") for path in labels.glob("*.nii.gz")}
    if set(cases) != label_cases:
        raise ValueError(
            f"Image/label mismatch: {len(set(cases)-label_cases)} missing labels and "
            f"{len(label_cases-set(cases))} missing images."
        )
    return cases


def stage_data(data_dir: Path, scratch: Path, config: dict, resume: bool) -> Path:
    target = scratch / "nnUNet_raw" / dataset_folder(config)
    target.mkdir(parents=True, exist_ok=True)
    for name in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        source, destination = data_dir / name, target / name
        if source.is_dir() and not destination.exists():
            print(f"Copying {source} -> {destination}")
            shutil.copytree(source, destination)
        elif name in ("imagesTr", "labelsTr") and not destination.is_dir():
            raise FileNotFoundError(f"Required folder not found: {source}")
    payload = {
        "channel_names": {str(key): value for key, value in config["channels"].items()},
        "labels": config["labels"],
        "numTraining": len(list((target / "labelsTr").glob("*.nii.gz"))),
        "file_ending": config["dataset"].get("file_ending", ".nii.gz"),
        "name": config["dataset"]["name"],
        "description": config["dataset"]["description"],
    }
    (target / "dataset.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def copy_outputs(data_dir: Path, scratch: Path, config: dict, trainer: str, folds: list[int]) -> Path:
    destination = data_dir / "perthesmetrics_nnunet_model"
    if destination.exists():
        remove(destination)
    destination.mkdir(parents=True)
    model_source = scratch / "nnUNet_results" / dataset_folder(config) / f"{trainer}__nnUNetPlans__2d"
    model_destination = destination / "nnUNet_results" / dataset_folder(config) / model_source.name
    model_destination.parent.mkdir(parents=True)
    shutil.copytree(model_source, model_destination)
    shutil.copytree(scratch / "export", destination / "export")
    split_dir = scratch / "nnUNet_preprocessed" / dataset_folder(config)
    for name in ("splits_final.json", "patient_split_audit.json", "patient_fold_assignments.csv"):
        shutil.copy2(split_dir / name, destination / name)
    model_zip = destination / "export" / "perthesmetrics_nnunet_model.zip"
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_folder(config), "configuration": "2d", "trainer": trainer,
        "plans": "nnUNetPlans", "folds": folds, "patient_grouped_cross_validation": True,
        "model_zip_sha256": sha256(model_zip),
        "software": {"python": sys.version.split()[0], "nnunetv2": package_version("nnunetv2"), "torch": package_version("torch")},
    }
    (destination / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return destination


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    training = config["training"]
    folds = args.folds or [int(value) for value in training.get("folds", range(5))]
    num_splits = int(training.get("num_splits", 5))
    if sorted(set(folds)) != sorted(folds) or not set(folds).issubset(range(num_splits)):
        raise ValueError(f"Folds must be unique values from 0 through {num_splits-1}.")
    epochs = args.epochs or int(training.get("epochs", 1000))
    if epochs not in EPOCH_TRAINERS:
        raise ValueError(f"Unsupported epoch count. Choose one of: {sorted(EPOCH_TRAINERS)}")
    trainer = EPOCH_TRAINERS[epochs]
    device = args.device or training.get("device", "cuda")
    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    cases = discover_cases(data_dir)
    map_path = args.patient_map.resolve() if args.patient_map else data_dir / training.get("patient_map", "patient_groups.csv")
    case_to_patient = resolve_groups(cases, load_patient_map(map_path if map_path.is_file() else None))
    splits, patient_to_fold = make_grouped_splits(case_to_patient, num_splits, int(training.get("split_seed", 2026)))
    scratch = prepare_scratch(args.scratch_dir, args.resume)
    completed = False
    try:
        stage_data(data_dir, scratch, config, args.resume)
        env = os.environ.copy()
        env.update({
            "nnUNet_raw": str(scratch / "nnUNet_raw"),
            "nnUNet_preprocessed": str(scratch / "nnUNet_preprocessed"),
            "nnUNet_results": str(scratch / "nnUNet_results"),
        })
        if args.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = args.gpu
        dataset_id = str(int(config["dataset"]["id"]))
        preprocessed = scratch / "nnUNet_preprocessed" / dataset_folder(config)
        plans_file = preprocessed / "nnUNetPlans.json"
        if not args.skip_planning and not (args.resume and plans_file.is_file()):
            run(["nnUNetv2_plan_and_preprocess", "-d", dataset_id, "-c", "2d", "--verify_dataset_integrity"], env)
        if not plans_file.is_file() and not args.skip_planning:
            raise FileNotFoundError(f"Preprocessing did not produce {plans_file}")
        write_split_artifacts(preprocessed, splits, case_to_patient, patient_to_fold, int(training.get("split_seed", 2026)))
        print(f"Patient-grouped split audit passed: {len(cases)} cases, {len(set(case_to_patient.values()))} patients.")
        if not args.skip_training:
            for fold in folds:
                command = ["nnUNetv2_train", dataset_id, "2d", str(fold), "-tr", trainer, "-p", "nnUNetPlans", "-device", device]
                if bool(training.get("save_npz", True)):
                    command.append("--npz")
                if args.resume:
                    command.append("--c")
                run(command, env)
            model_dir = scratch / "nnUNet_results" / dataset_folder(config) / f"{trainer}__nnUNetPlans__2d"
            metrics_script = Path(__file__).with_name("analyze_training_metrics.py")
            run([sys.executable, str(metrics_script), "--model-dir", str(model_dir)], env)
        export_script = Path(__file__).with_name("export_nnunet_model_2d.py")
        run([sys.executable, str(export_script), "--config", str(config_path), "--work-dir", str(scratch), "--trainer", trainer], env)
        destination = copy_outputs(data_dir, scratch, config, trainer, folds)
        print(f"Copied training outputs to: {destination}")
        completed = True
    finally:
        if scratch.exists() and completed and not args.keep_scratch:
            remove(scratch)
            print(f"Deleted scratch directory: {scratch}")
        elif scratch.exists():
            print(f"Kept scratch directory for resume or inspection: {scratch}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
