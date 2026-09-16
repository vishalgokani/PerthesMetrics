"""Generate publication-ready nnU-Net training and validation supplements.

The script reads nnU-Net v2 training logs and validation ``summary.json`` files
without importing nnU-Net or PyTorch. It expects completed folds 0--4 and writes
five numbered figures as both PNG and PDF, plus compact source-data tables.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


EXPECTED_FOLDS = tuple(range(5))
METRIC_PATTERNS = {
    "epoch": re.compile(r"\bEpoch\s+(\d+)\b"),
    "learning_rate": re.compile(r"Current learning rate:\s*([-+\d.eE]+)"),
    "train_loss": re.compile(r"\btrain_loss\s+([-+\d.eE]+)"),
    "val_loss": re.compile(r"\bval_loss\s+([-+\d.eE]+)"),
    "pseudo_dice": re.compile(r"Pseudo dice\s*\[([^\]]+)\]"),
    "epoch_time_seconds": re.compile(r"Epoch time:\s*([-+\d.eE]+)\s*s"),
}
FOLD_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def parse_training_log(path: Path) -> list[dict[str, object]]:
    """Parse epoch metrics from one nnU-Net v2 text log."""
    rows: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        epoch_match = METRIC_PATTERNS["epoch"].search(line)
        if epoch_match:
            if current is not None:
                rows.append(current)
            current = {"epoch": int(epoch_match.group(1))}
            continue
        if current is None:
            continue
        for key in ("learning_rate", "train_loss", "val_loss", "epoch_time_seconds"):
            match = METRIC_PATTERNS[key].search(line)
            if match:
                current[key] = float(match.group(1))
        dice_match = METRIC_PATTERNS["pseudo_dice"].search(line)
        if dice_match:
            values = []
            for token in dice_match.group(1).split(","):
                token = token.strip()
                values.append(float(token) if token.lower() != "nan" else math.nan)
            current["pseudo_dice"] = values
            valid = [value for value in values if math.isfinite(value)]
            current["mean_pseudo_dice"] = mean(valid) if valid else math.nan
    if current is not None:
        rows.append(current)
    return rows


def load_fold_history(fold_dir: Path) -> list[dict[str, object]]:
    """Merge resumed logs, preferring the newest record for duplicate epochs."""
    logs = sorted(fold_dir.glob("training_log*.txt"), key=lambda path: path.stat().st_mtime)
    if not logs:
        raise FileNotFoundError(f"No training_log*.txt files found in {fold_dir}")
    by_epoch: dict[int, dict[str, object]] = {}
    for log in logs:
        for row in parse_training_log(log):
            by_epoch[int(row["epoch"])] = row
    rows = [by_epoch[epoch] for epoch in sorted(by_epoch)]
    required = {"train_loss", "val_loss", "pseudo_dice"}
    incomplete = [int(row["epoch"]) for row in rows if not required.issubset(row)]
    if incomplete:
        preview = ", ".join(map(str, incomplete[:8]))
        raise ValueError(f"Incomplete epoch records in {fold_dir.name}: {preview}")
    return rows


def moving_average(values: Sequence[float], window: int) -> list[float]:
    """Return a centered finite-value moving average with unchanged length."""
    if window < 1:
        raise ValueError("Smoothing window must be at least 1")
    radius = window // 2
    smoothed: list[float] = []
    for index in range(len(values)):
        subset = [
            float(value)
            for value in values[max(0, index - radius) : min(len(values), index + radius + 1)]
            if _finite(value)
        ]
        smoothed.append(mean(subset) if subset else math.nan)
    return smoothed


def _epoch_statistics(
    histories: dict[int, list[dict[str, object]]], key: str, class_index: int | None = None
) -> tuple[list[int], list[float], list[float]]:
    values_by_epoch: dict[int, list[float]] = defaultdict(list)
    for rows in histories.values():
        for row in rows:
            if class_index is None:
                value = row.get(key)
            else:
                dice = row.get("pseudo_dice", [])
                value = dice[class_index] if class_index < len(dice) else math.nan
            if _finite(value):
                values_by_epoch[int(row["epoch"])].append(float(value))
    epochs = sorted(values_by_epoch)
    averages = [mean(values_by_epoch[epoch]) for epoch in epochs]
    deviations = [stdev(values_by_epoch[epoch]) if len(values_by_epoch[epoch]) > 1 else 0.0 for epoch in epochs]
    return epochs, averages, deviations


def _class_names(model_dir: Path, n_classes: int) -> list[str]:
    dataset_path = model_dir / "dataset.json"
    if not dataset_path.is_file():
        return [f"Class {index}" for index in range(1, n_classes + 1)]
    labels = json.loads(dataset_path.read_text(encoding="utf-8"))["labels"]
    names_by_value = {int(value): str(name) for name, value in labels.items() if int(value) != 0}
    return [names_by_value.get(index, f"Class {index}") for index in range(1, n_classes + 1)]


def _pretty_label(label: str) -> str:
    abbreviations = {"gt": "Greater trochanter", "lt": "Lesser trochanter"}
    return abbreviations.get(label.lower(), label.capitalize())


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.7)
    axis.tick_params(labelsize=8)


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.12, 1.05, label, transform=axis.transAxes, fontweight="bold", fontsize=11)


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    figure.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_fold_series(
    axis: plt.Axes,
    rows: list[dict[str, object]],
    key: str,
    color: str,
    smooth_window: int,
    label: str,
) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    values = [float(row[key]) for row in rows]
    axis.plot(epochs, values, color=color, alpha=0.14, linewidth=0.55)
    axis.plot(epochs, moving_average(values, smooth_window), color=color, linewidth=1.5, label=label)


def figure_loss_curves(
    histories: dict[int, list[dict[str, object]]], output_dir: Path, smooth_window: int, dpi: int
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(10.2, 6.2), sharex=True, sharey=True)
    flat_axes = list(axes.flat)
    for panel, fold in enumerate(EXPECTED_FOLDS):
        axis = flat_axes[panel]
        rows = histories[fold]
        _plot_fold_series(axis, rows, "train_loss", "#0072B2", smooth_window, "Training")
        _plot_fold_series(axis, rows, "val_loss", "#D55E00", smooth_window, "Validation")
        axis.set_title(f"Fold {fold}", fontsize=10)
        _style_axis(axis)
        _panel_label(axis, chr(65 + panel))
    aggregate = flat_axes[-1]
    for key, color, label in (
        ("train_loss", "#0072B2", "Training"),
        ("val_loss", "#D55E00", "Validation"),
    ):
        epochs, averages, deviations = _epoch_statistics(histories, key)
        averages = moving_average(averages, smooth_window)
        deviations = moving_average(deviations, smooth_window)
        aggregate.plot(epochs, averages, color=color, linewidth=1.8, label=label)
        aggregate.fill_between(
            epochs,
            [value - spread for value, spread in zip(averages, deviations)],
            [value + spread for value, spread in zip(averages, deviations)],
            color=color,
            alpha=0.16,
            linewidth=0,
        )
    aggregate.set_title("Cross-fold mean ± SD", fontsize=10)
    aggregate.legend(frameon=False, fontsize=8)
    _style_axis(aggregate)
    _panel_label(aggregate, "F")
    for axis in axes[-1, :]:
        axis.set_xlabel("Epoch")
    for axis in axes[:, 0]:
        axis.set_ylabel("nnU-Net compound loss")
    figure.tight_layout()
    _save_figure(figure, output_dir, "supplementalfigure1", dpi)


def figure_mean_pseudo_dice(
    histories: dict[int, list[dict[str, object]]], output_dir: Path, smooth_window: int, dpi: int
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.7), sharex=True, sharey=True)
    for fold, color in zip(EXPECTED_FOLDS, FOLD_COLORS):
        rows = histories[fold]
        epochs = [int(row["epoch"]) for row in rows]
        values = [float(row["mean_pseudo_dice"]) for row in rows]
        axes[0].plot(epochs, moving_average(values, smooth_window), color=color, linewidth=1.25, label=f"Fold {fold}")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[0].set_title("Fold-specific trajectories", fontsize=10)
    epochs, averages, deviations = _epoch_statistics(histories, "mean_pseudo_dice")
    averages = moving_average(averages, smooth_window)
    deviations = moving_average(deviations, smooth_window)
    axes[1].plot(epochs, averages, color="#222222", linewidth=1.8)
    axes[1].fill_between(
        epochs,
        [max(0.0, value - spread) for value, spread in zip(averages, deviations)],
        [min(1.0, value + spread) for value, spread in zip(averages, deviations)],
        color="#777777",
        alpha=0.24,
        linewidth=0,
    )
    axes[1].set_title("Cross-fold mean ± SD", fontsize=10)
    for panel, axis in enumerate(axes):
        axis.set_xlabel("Epoch")
        axis.set_ylim(0.45, 1.0)
        _style_axis(axis)
        _panel_label(axis, chr(65 + panel))
    axes[0].set_ylabel("Mean foreground pseudo Dice")
    figure.tight_layout()
    _save_figure(figure, output_dir, "supplementalfigure2", dpi)


def figure_class_pseudo_dice(
    histories: dict[int, list[dict[str, object]]],
    class_names: Sequence[str],
    output_dir: Path,
    smooth_window: int,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(11.2, 5.8), sharex=True, sharey=True)
    for class_index, (axis, class_name) in enumerate(zip(axes.flat, class_names)):
        epochs, averages, deviations = _epoch_statistics(histories, "pseudo_dice", class_index)
        averages = moving_average(averages, smooth_window)
        deviations = moving_average(deviations, smooth_window)
        color = FOLD_COLORS[class_index % len(FOLD_COLORS)]
        axis.plot(epochs, averages, color=color, linewidth=1.5)
        axis.fill_between(
            epochs,
            [max(0.0, value - spread) for value, spread in zip(averages, deviations)],
            [min(1.0, value + spread) for value, spread in zip(averages, deviations)],
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        axis.set_title(_pretty_label(class_name), fontsize=9)
        axis.set_ylim(0.3, 1.0)
        _style_axis(axis)
        _panel_label(axis, chr(65 + class_index))
    for axis in axes[-1, :]:
        axis.set_xlabel("Epoch")
    for axis in axes[:, 0]:
        axis.set_ylabel("Pseudo Dice (mean ± SD)")
    figure.tight_layout()
    _save_figure(figure, output_dir, "supplementalfigure3", dpi)


def load_validation_metrics(
    model_dir: Path, class_names: Sequence[str]
) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, list[float]]]]:
    fold_means: dict[int, dict[str, float]] = {}
    per_case: dict[int, dict[str, list[float]]] = {}
    for fold in EXPECTED_FOLDS:
        path = model_dir / f"fold_{fold}" / "validation" / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing final validation summary: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        fold_means[fold] = {"Foreground mean": float(summary["foreground_mean"]["Dice"])}
        for index, name in enumerate(class_names, start=1):
            fold_means[fold][name] = float(summary["mean"][str(index)]["Dice"])
        case_values: dict[str, list[float]] = {name: [] for name in class_names}
        for case in summary.get("metric_per_case", []):
            metrics = case.get("metrics", {})
            for index, name in enumerate(class_names, start=1):
                value = metrics.get(str(index), {}).get("Dice")
                if _finite(value):
                    case_values[name].append(float(value))
        per_case[fold] = case_values
    return fold_means, per_case


def figure_final_validation(
    fold_means: dict[int, dict[str, float]], class_names: Sequence[str], output_dir: Path, dpi: int
) -> None:
    figure, (heat_axis, overall_axis) = plt.subplots(
        1, 2, figsize=(11.2, 4.6), gridspec_kw={"width_ratios": (3.5, 1.25)}
    )
    matrix = [[fold_means[fold][name] for fold in EXPECTED_FOLDS] for name in class_names]
    cmap = LinearSegmentedColormap.from_list("dice", ["#F7FBFF", "#6BAED6", "#08306B"])
    image = heat_axis.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.70, vmax=1.0)
    heat_axis.set_xticks(range(5), [f"Fold {fold}" for fold in EXPECTED_FOLDS])
    heat_axis.set_yticks(range(len(class_names)), [_pretty_label(name) for name in class_names])
    heat_axis.tick_params(labelsize=8)
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            heat_axis.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if value >= 0.88 else "#111111",
            )
    colorbar = figure.colorbar(image, ax=heat_axis, fraction=0.04, pad=0.03)
    colorbar.set_label("Dice", fontsize=9)
    heat_axis.set_title("Class-specific final validation Dice", fontsize=10)
    overall = [fold_means[fold]["Foreground mean"] for fold in EXPECTED_FOLDS]
    overall_axis.bar(range(5), overall, color=FOLD_COLORS, width=0.68)
    overall_axis.set_xticks(range(5), [str(fold) for fold in EXPECTED_FOLDS])
    overall_axis.set_ylim(min(0.80, min(overall) - 0.02), min(1.0, max(overall) + 0.035))
    overall_axis.set_xlabel("Fold")
    overall_axis.set_ylabel("Foreground mean Dice")
    overall_axis.set_title("Overall performance", fontsize=10)
    for index, value in enumerate(overall):
        overall_axis.text(index, value + 0.003, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    _style_axis(overall_axis)
    _panel_label(heat_axis, "A")
    _panel_label(overall_axis, "B")
    figure.tight_layout()
    _save_figure(figure, output_dir, "supplementalfigure4", dpi)


def figure_case_distributions(
    per_case: dict[int, dict[str, list[float]]], class_names: Sequence[str], output_dir: Path, dpi: int
) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(11.2, 5.9), sharex=True, sharey=True)
    for class_index, (axis, class_name) in enumerate(zip(axes.flat, class_names)):
        values = [per_case[fold][class_name] for fold in EXPECTED_FOLDS]
        label_parameter = "tick_labels" if "tick_labels" in inspect.signature(axis.boxplot).parameters else "labels"
        plot = axis.boxplot(
            values,
            widths=0.62,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.2},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
            boxprops={"linewidth": 0.8},
            **{label_parameter: [str(fold) for fold in EXPECTED_FOLDS]},
        )
        for patch, color in zip(plot["boxes"], FOLD_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
        axis.set_title(_pretty_label(class_name), fontsize=9)
        axis.set_ylim(0.0, 1.02)
        _style_axis(axis)
        _panel_label(axis, chr(65 + class_index))
    for axis in axes[-1, :]:
        axis.set_xlabel("Fold")
    for axis in axes[:, 0]:
        axis.set_ylabel("Per-case Dice")
    figure.tight_layout()
    _save_figure(figure, output_dir, "supplementalfigure5", dpi)


def _write_source_tables(
    histories: dict[int, list[dict[str, object]]],
    fold_means: dict[int, dict[str, float]],
    class_names: Sequence[str],
    output_dir: Path,
) -> None:
    epoch_path = output_dir / "supplemental_training_metrics.csv"
    fields = [
        "fold",
        "epoch",
        "train_loss",
        "validation_loss",
        "mean_pseudo_dice",
        "learning_rate",
        "epoch_time_seconds",
    ] + [f"pseudo_dice_{name.replace(' ', '_')}" for name in class_names]
    with epoch_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for fold, rows in histories.items():
            for row in rows:
                output = {
                    "fold": fold,
                    "epoch": row["epoch"],
                    "train_loss": row.get("train_loss"),
                    "validation_loss": row.get("val_loss"),
                    "mean_pseudo_dice": row.get("mean_pseudo_dice"),
                    "learning_rate": row.get("learning_rate"),
                    "epoch_time_seconds": row.get("epoch_time_seconds"),
                }
                for name, value in zip(class_names, row.get("pseudo_dice", [])):
                    output[f"pseudo_dice_{name.replace(' ', '_')}"] = value
                writer.writerow(output)
    validation_path = output_dir / "supplemental_final_validation_dice.csv"
    with validation_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["fold", "foreground_mean_dice"] + [f"dice_{name.replace(' ', '_')}" for name in class_names]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for fold in EXPECTED_FOLDS:
            writer.writerow(
                {
                    "fold": fold,
                    "foreground_mean_dice": fold_means[fold]["Foreground mean"],
                    **{f"dice_{name.replace(' ', '_')}": fold_means[fold][name] for name in class_names},
                }
            )


def _write_captions(output_dir: Path, smooth_window: int) -> None:
    captions = f"""Supplemental Figure 1. Training and validation loss over 1,000 epochs for each of the five cross-validation folds (A-E) and across-fold mean ± standard deviation (F). Thin lines show epoch-level values and thick lines show a centered {smooth_window}-epoch moving average. Loss is the nnU-Net compound training objective and may be negative.

Supplemental Figure 2. Online mean foreground pseudo Dice during training. (A) Fold-specific trajectories. (B) Across-fold mean ± standard deviation. Curves show a centered {smooth_window}-epoch moving average. Pseudo Dice is calculated online from sampled validation patches and is not equivalent to the final full-case validation Dice.

Supplemental Figure 3. Class-specific online validation pseudo Dice over training, shown as the across-fold mean ± standard deviation with a centered {smooth_window}-epoch moving average.

Supplemental Figure 4. Final full-case validation performance from nnU-Net cross-validation. (A) Class-specific mean Dice for each held-out fold. (B) Mean foreground Dice for each fold.

Supplemental Figure 5. Distribution of full-case validation Dice for each anatomical class and held-out fold. Boxes show the interquartile range and median; whiskers extend to 1.5 times the interquartile range. Outliers are omitted visually for readability but remain in the underlying nnU-Net summary files.
"""
    (output_dir / "supplemental_figure_captions.txt").write_text(captions, encoding="utf-8")


def generate_supplement(
    model_dir: Path, output_dir: Path, smooth_window: int = 25, dpi: int = 300
) -> list[Path]:
    """Generate all supplemental figures and return their PNG paths."""
    model_dir = model_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = [fold for fold in EXPECTED_FOLDS if not (model_dir / f"fold_{fold}").is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing expected fold directories under {model_dir}: {missing}")
    histories = {fold: load_fold_history(model_dir / f"fold_{fold}") for fold in EXPECTED_FOLDS}
    class_count = len(histories[0][0]["pseudo_dice"])
    class_names = _class_names(model_dir, class_count)
    if len(class_names) != 8:
        raise ValueError(f"Expected 8 foreground classes for this study; found {len(class_names)}")
    fold_means, per_case = load_validation_metrics(model_dir, class_names)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure_loss_curves(histories, output_dir, smooth_window, dpi)
    figure_mean_pseudo_dice(histories, output_dir, smooth_window, dpi)
    figure_class_pseudo_dice(histories, class_names, output_dir, smooth_window, dpi)
    figure_final_validation(fold_means, class_names, output_dir, dpi)
    figure_case_distributions(per_case, class_names, output_dir, dpi)
    _write_source_tables(histories, fold_means, class_names, output_dir)
    _write_captions(output_dir, smooth_window)
    return [output_dir / f"supplementalfigure{index}.png" for index in range(1, 6)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="nnU-Net model folder containing fold_0 through fold_4",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for figures and tables")
    parser.add_argument("--smooth-window", type=int, default=25, help="Centered moving-average window (default: 25)")
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution (default: 300)")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    outputs = generate_supplement(args.model_dir, args.output_dir, args.smooth_window, args.dpi)
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
