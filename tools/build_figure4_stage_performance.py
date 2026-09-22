"""Build publication Figure 4 from patient-bootstrap performance summaries.

The figure compares overall held-out performance, unaffected hips, and affected
hips stratified by modified Waldenstrom stage for each analysis mask. Inputs
are aggregate CSV files written by ``tools/final_mask_analysis.py``; no
patient-level records are read by this script.
"""

from __future__ import annotations

import argparse
import csv
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


GROUPS = ("Overall", "Unaffected", "Ia", "Ib", "IIa", "IIb", "IIIa", "IIIb", "IV")
STAGE_GROUPS = set(GROUPS[2:])
MASKS = (
    "head",
    "neck_shaft",
    "gt",
    "lt",
    "sourcil",
    "triradiate cartilage",
    "acetabulum",
)
MASK_LABELS = {
    "acetabulum": "Acetabulum",
    "head": "Femoral Head",
    "neck_shaft": "Femoral Neck/Shaft",
    "sourcil": "Sourcil",
    "triradiate cartilage": "Triradiate Cartilage",
    "lt": "Lesser Trochanter",
    "gt": "Greater Trochanter",
}

# This muted stratum palette is deliberately different from the saturated
# anatomical-mask colors used in Figures 1 and 3. Position and marker shape
# provide redundant encoding for grayscale and color-vision accessibility.
GROUP_COLORS = {
    "Overall": "#111111",
    "Unaffected": "#7A7A7A",
    "Ia": "#B8576B",
    "Ib": "#C77943",
    "IIa": "#B59A32",
    "IIb": "#718B3B",
    "IIIa": "#39847F",
    "IIIb": "#526F9B",
    "IV": "#8A5C8F",
}
GROUP_MARKERS = {
    "Overall": "o",
    "Unaffected": "s",
    "Ia": "o",
    "Ib": "s",
    "IIa": "^",
    "IIb": "D",
    "IIIa": "v",
    "IIIb": "P",
    "IV": "X",
}
REQUIRED_COLUMNS = {
    "mask",
    "mean_patient_pooled_dice",
    "ci95_low",
    "ci95_high",
}


@dataclass(frozen=True)
class Estimate:
    mean: float
    ci_low: float
    ci_high: float


def read_csv(path: Path, extra_required: set[str] | None = None) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required Figure 4 input not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        required = REQUIRED_COLUMNS | (extra_required or set())
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        return list(reader)


def parse_estimate(row: dict[str, str], source: Path) -> Estimate:
    try:
        estimate = Estimate(
            mean=float(row["mean_patient_pooled_dice"]),
            ci_low=float(row["ci95_low"]),
            ci_high=float(row["ci95_high"]),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Non-numeric Figure 4 estimate in {source}: mask={row.get('mask')!r}"
        ) from error
    values = (estimate.ci_low, estimate.mean, estimate.ci_high)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite Figure 4 estimate in {source}: {row}")
    if not 0.0 <= estimate.ci_low <= estimate.mean <= estimate.ci_high <= 1.0:
        raise ValueError(
            "Expected 0 <= ci95_low <= mean_patient_pooled_dice <= ci95_high <= 1 "
            f"in {source}; found {values} for mask={row.get('mask')!r}."
        )
    return estimate


def add_estimate(
    output: dict[tuple[str, str], Estimate],
    group: str,
    mask: str,
    row: dict[str, str],
    source: Path,
) -> None:
    key = (group, mask)
    if key in output:
        raise ValueError(f"Duplicate Figure 4 estimate for group={group!r}, mask={mask!r} in {source}")
    output[key] = parse_estimate(row, source)


def load_estimates(analysis_dir: Path) -> dict[tuple[str, str], Estimate]:
    """Load and validate all 63 group-by-mask estimates required by Figure 4."""
    overall_path = analysis_dir / "overall_model_performance_by_mask.csv"
    group_path = analysis_dir / "analysis_group_model_performance_by_mask.csv"
    output: dict[tuple[str, str], Estimate] = {}

    for row in read_csv(overall_path):
        mask = row["mask"].strip()
        if mask in MASKS:
            add_estimate(output, "Overall", mask, row, overall_path)

    for row in read_csv(group_path, {"analysis_group"}):
        group = row["analysis_group"].strip()
        mask = row["mask"].strip()
        if group in {"Unaffected", *STAGE_GROUPS} and mask in MASKS:
            add_estimate(output, group, mask, row, group_path)

    missing = [
        (group, mask)
        for mask in MASKS
        for group in GROUPS
        if (group, mask) not in output
    ]
    if missing:
        formatted = ", ".join(f"{group}/{mask}" for group, mask in missing)
        raise ValueError(f"Missing Figure 4 estimates: {formatted}")
    return output


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.85,
        }
    )


def draw_interval(axis: plt.Axes, x: float, estimate: Estimate, group: str) -> None:
    color = GROUP_COLORS[group]
    axis.vlines(x, estimate.ci_low, estimate.ci_high, color=color, linewidth=2.1, zorder=2)
    axis.hlines(
        [estimate.ci_low, estimate.ci_high],
        x - 0.105,
        x + 0.105,
        color=color,
        linewidth=2.1,
        zorder=2,
    )
    axis.scatter(
        x,
        estimate.mean,
        s=31,
        marker=GROUP_MARKERS[group],
        facecolor=color,
        edgecolor="#171717",
        linewidth=0.55,
        zorder=3,
    )


def finish_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#DEDEDE", linewidth=0.6, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(width=0.8)


def build_figure(
    estimates: dict[tuple[str, str], Estimate], benchmark: float = 0.80
) -> plt.Figure:
    """Create the seven-panel Figure 4 without an embedded title or caption."""
    if not 0.0 <= benchmark <= 1.0:
        raise ValueError("--benchmark must be between 0 and 1.")
    configure_style()
    figure, axes = plt.subplots(2, 4, figsize=(16.2, 8.2), sharey=True)
    figure.subplots_adjust(
        left=0.065,
        right=0.99,
        top=0.96,
        bottom=0.12,
        hspace=0.50,
        wspace=0.13,
    )
    positions = np.arange(1, len(GROUPS) + 1)
    for panel_index, (axis, mask) in enumerate(zip(axes.flat, MASKS)):
        for position, group in zip(positions, GROUPS):
            draw_interval(axis, float(position), estimates[(group, mask)], group)
        axis.axhline(
            benchmark,
            color="#777777",
            linestyle=(0, (5, 3)),
            linewidth=1.05,
            zorder=1,
        )
        axis.set_title(
            f"{'abcdefg'[panel_index]})  {MASK_LABELS[mask]}",
            loc="left",
            fontweight="bold",
            pad=7,
        )
        axis.set_xlim(0.45, len(GROUPS) + 0.55)
        axis.set_ylim(0.47, 1.01)
        axis.set_xticks(positions)
        axis.set_xticklabels(GROUPS, rotation=42, ha="right", rotation_mode="anchor")
        finish_axis(axis)

    axes.flat[-1].axis("off")
    y_label = "Mean Patient-Pooled Dice\n(95% Bootstrap CI)"
    axes[0, 0].set_ylabel(y_label, fontweight="bold")
    axes[1, 0].set_ylabel(y_label, fontweight="bold")
    handles = [
        Line2D(
            [0],
            [0],
            marker=GROUP_MARKERS[group],
            linestyle="none",
            markersize=6,
            markerfacecolor=GROUP_COLORS[group],
            markeredgecolor="#171717",
            label=group,
        )
        for group in GROUPS
    ]
    axes.flat[-1].legend(
        handles=handles,
        title="Analysis stratum",
        frameon=False,
        ncol=2,
        loc="center",
        fontsize=12.5,
        title_fontsize=14.5,
        columnspacing=1.4,
    )
    return figure


def atomic_savefig(figure: plt.Figure, path: Path, **kwargs: object) -> None:
    """Write a complete figure before atomically replacing the destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.",
        suffix=path.suffix,
        dir=path.parent,
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
    try:
        figure.savefig(temporary_path, **kwargs)
        temporary_path.replace(path)
    except PermissionError as error:
        raise PermissionError(
            f"Could not replace {path}. Close any application displaying that file and rerun the command."
        ) from error
    finally:
        temporary_path.unlink(missing_ok=True)


def export_figure(
    figure: plt.Figure, output_dir: Path, dpi: int, output_name: str = "Figure4"
) -> tuple[Path, Path, Path]:
    if dpi < 300:
        raise ValueError("Use --dpi 300 or higher for publication output.")
    if not output_name or Path(output_name).name != output_name or Path(output_name).suffix:
        raise ValueError("--output-name must be a filename stem without a path or extension.")
    prefix = output_dir / output_name
    pdf_path = prefix.with_suffix(".pdf")
    svg_path = prefix.with_suffix(".svg")
    png_path = prefix.with_suffix(".png")
    try:
        atomic_savefig(figure, pdf_path, bbox_inches="tight")
        atomic_savefig(figure, svg_path, bbox_inches="tight")
        atomic_savefig(figure, png_path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(figure)
    return pdf_path, svg_path, png_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        required=True,
        type=Path,
        help=(
            "Directory containing overall_model_performance_by_mask.csv and "
            "analysis_group_model_performance_by_mask.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Destination directory for PDF, SVG, and PNG outputs.",
    )
    parser.add_argument(
        "--output-name",
        default="Figure4",
        help="Output filename stem without an extension (default: Figure4).",
    )
    parser.add_argument("--benchmark", type=float, default=0.80)
    parser.add_argument("--dpi", type=int, default=600)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    estimates = load_estimates(args.analysis_dir.resolve())
    figure = build_figure(estimates, benchmark=args.benchmark)
    paths = export_figure(
        figure, args.output_dir.resolve(), dpi=args.dpi, output_name=args.output_name
    )
    for path in paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
