"""Create lightweight per-epoch and cross-fold training reports from nnU-Net logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PATTERNS = {
    "epoch": re.compile(r"Epoch (\d+)"),
    "train_loss": re.compile(r"train_loss ([\-\d.eE]+)"),
    "val_loss": re.compile(r"val_loss ([\-\d.eE]+)"),
    "pseudo_dice": re.compile(r"Pseudo dice \[([^\]]+)\]"),
    "epoch_time_seconds": re.compile(r"Epoch time: ([\d.eE]+) s"),
}


def parse_log(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = PATTERNS["epoch"].search(line)
        if match:
            if current is not None:
                rows.append(current)
            current = {"epoch": int(match.group(1))}
            continue
        if current is None:
            continue
        for key in ("train_loss", "val_loss", "epoch_time_seconds"):
            match = PATTERNS[key].search(line)
            if match:
                current[key] = float(match.group(1))
        match = PATTERNS["pseudo_dice"].search(line)
        if match:
            values = [float(value.strip()) for value in match.group(1).split(",") if value.strip().lower() != "nan"]
            current["mean_pseudo_dice"] = mean(values) if values else None
    if current is not None:
        rows.append(current)
    return rows


def finite(values: list[object]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def analyze_model_folder(model_dir: Path, output_dir: Path | None = None) -> Path:
    model_dir = model_dir.resolve()
    reports = (output_dir or model_dir / "reports").resolve()
    reports.mkdir(parents=True, exist_ok=True)
    fold_summaries = []
    for fold_dir in sorted(model_dir.glob("fold_*")):
        logs = sorted(fold_dir.glob("training_log*.txt"), key=lambda path: path.stat().st_mtime)
        if not logs:
            continue
        by_epoch: dict[int, dict[str, object]] = {}
        for log in logs:
            for row in parse_log(log):
                by_epoch[int(row["epoch"])] = row
        rows = [by_epoch[epoch] for epoch in sorted(by_epoch)]
        if not rows:
            continue
        fields = ["epoch", "train_loss", "val_loss", "mean_pseudo_dice", "epoch_time_seconds"]
        with (reports / f"{fold_dir.name}_epoch_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)
        epochs = [row["epoch"] for row in rows]
        figure, axes = plt.subplots(2, 1, figsize=(7.2, 7), sharex=True)
        axes[0].plot(epochs, [row.get("train_loss") for row in rows], label="training loss")
        axes[0].plot(epochs, [row.get("val_loss") for row in rows], label="validation loss")
        axes[0].legend(frameon=False); axes[0].set_ylabel("Loss")
        axes[1].plot(epochs, [row.get("mean_pseudo_dice") for row in rows])
        axes[1].set_ylabel("Mean pseudo Dice"); axes[1].set_xlabel("Epoch"); axes[1].set_ylim(0, 1)
        figure.suptitle(f"PerthesMetrics {fold_dir.name} training"); figure.tight_layout()
        figure.savefig(reports / f"{fold_dir.name}_training_curves.png", dpi=160); plt.close(figure)
        validation = fold_dir / "validation" / "summary.json"
        validation_dice = None
        if validation.is_file():
            summary = json.loads(validation.read_text(encoding="utf-8"))
            validation_dice = summary.get("foreground_mean", {}).get("Dice")
        dice = finite([row.get("mean_pseudo_dice") for row in rows])
        losses = finite([row.get("val_loss") for row in rows])
        times = finite([row.get("epoch_time_seconds") for row in rows])
        fold_summaries.append({
            "fold": fold_dir.name, "epochs_recorded": len(rows),
            "best_mean_pseudo_dice": max(dice) if dice else None,
            "best_validation_loss": min(losses) if losses else None,
            "final_validation_foreground_dice": validation_dice,
            "median_epoch_time_seconds": median(times) if times else None,
        })
    if not fold_summaries:
        raise FileNotFoundError(f"No training logs found under {model_dir}")
    fields = list(fold_summaries[0])
    with (reports / "cross_fold_training_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(fold_summaries)
    (reports / "cross_fold_training_summary.json").write_text(json.dumps(fold_summaries, indent=2) + "\n", encoding="utf-8")
    return reports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(analyze_model_folder(args.model_dir, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
