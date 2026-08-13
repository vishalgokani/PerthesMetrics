"""Export a trained PerthesMetrics nnU-Net model as an installable zip."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained PerthesMetrics 2D nnU-Net model.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("perthesmetrics_nnunet2d.yaml"))
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--trainer", default="nnUNetTrainer")
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--checkpoint", default="checkpoint_final.pth")
    parser.add_argument("--output-zip", type=Path)
    return parser.parse_args()


def find_executable(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    candidate = Path(sys.executable).resolve().parent / "Scripts" / f"{name}.exe"
    return str(candidate) if candidate.is_file() else name


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset_id = int(config["dataset"]["id"])
    dataset_name = f"Dataset{dataset_id:03d}_{config['dataset']['name']}"
    work_dir = args.work_dir.resolve()
    model_dir = work_dir / "nnUNet_results" / dataset_name / f"{args.trainer}__{args.plans}__{args.configuration}"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model folder not found: {model_dir}")
    folds = sorted(
        child.name.removeprefix("fold_")
        for child in model_dir.glob("fold_*")
        if child.is_dir() and child.name.removeprefix("fold_").isdigit()
    )
    if not folds:
        raise FileNotFoundError(f"No trained folds found in {model_dir}")
    missing = [fold for fold in folds if not (model_dir / f"fold_{fold}" / args.checkpoint).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {args.checkpoint} for folds: {', '.join(missing)}")
    export_dir = work_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    output_zip = (args.output_zip or export_dir / "perthesmetrics_nnunet_model.zip").resolve()
    env = os.environ.copy()
    env.update(
        {
            "nnUNet_raw": str(work_dir / "nnUNet_raw"),
            "nnUNet_preprocessed": str(work_dir / "nnUNet_preprocessed"),
            "nnUNet_results": str(work_dir / "nnUNet_results"),
        }
    )
    command = [
        find_executable("nnUNetv2_export_model_to_zip"), "-d", str(dataset_id), "-o", str(output_zip),
        "-c", args.configuration, "-tr", args.trainer, "-p", args.plans, "-f", *folds,
        "-chk", args.checkpoint,
    ]
    print(" ".join(command))
    subprocess.run(command, check=True, env=env)
    print(f"Exported model: {output_zip}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
