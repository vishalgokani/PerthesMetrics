"""Build sanitized aggregate PerthesMetrics results from a release model zip."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import zipfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = {
    "1": "acetabulum", "2": "gt", "3": "head", "4": "lt", "5": "neck",
    "6": "shaft", "7": "sourcil", "8": "triradiate_cartilage",
}


def archive_root(names: set[str]) -> str:
    candidates = sorted({name.split("fold_0/", 1)[0] for name in names if "fold_0/validation/summary.json" in name})
    if len(candidates) != 1:
        raise ValueError(f"Expected one nnU-Net model root; found {candidates}")
    return candidates[0]


def aggregate_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "foreground_mean": summary.get("foreground_mean", {}),
        "label_means": {LABELS.get(str(key), str(key)): value for key, value in summary.get("mean", {}).items()},
    }


def metric_rows(name: str, aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for metric, value in aggregate.get("foreground_mean", {}).items():
        rows.append({"fold": name, "scope": "foreground_mean", "label": "foreground", "metric": metric, "value": value})
    for label, metrics in aggregate.get("label_means", {}).items():
        for metric, value in metrics.items():
            rows.append({"fold": name, "scope": "label_mean", "label": label, "metric": metric, "value": value})
    return rows


def dice_row(name: str, aggregate: dict[str, Any]) -> dict[str, Any]:
    row = {"fold": name, "foreground_mean_dice": aggregate.get("foreground_mean", {}).get("Dice")}
    for label in LABELS.values():
        row[f"dice_{label}"] = aggregate.get("label_means", {}).get(label, {}).get("Dice")
    return row


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def plot_dice(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = list(LABELS.values())
    figure, axis = plt.subplots(figsize=(11, 5.5))
    x = list(range(len(rows)))
    width = 0.09
    start = -(len(labels) - 1) * width / 2
    for index, label in enumerate(labels):
        axis.bar([value + start + index * width for value in x], [row.get(f"dice_{label}") or 0 for row in rows], width, label=label.replace("_", " "))
    axis.plot(x, [row.get("foreground_mean_dice") or 0 for row in rows], color="black", marker="o", linewidth=2, label="foreground mean")
    axis.set_xticks(x, [row["fold"] for row in rows], rotation=15)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Dice")
    axis.set_title("PerthesMetrics patient-grouped nnU-Net validation Dice")
    axis.legend(frameon=False, ncol=3, fontsize=8)
    figure.tight_layout(); figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight"); plt.close(figure)


def cross_fold_statistics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean, sample SD, and t-based 95% CI across the five validation folds."""
    output = []
    t_975_df4 = 2.7764451051977987
    fields = ["foreground_mean_dice", *[f"dice_{label}" for label in LABELS.values()]]
    for field in fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        average = statistics.mean(values); sd = statistics.stdev(values)
        margin = t_975_df4 * sd / len(values) ** 0.5
        output.append({"region": field.removeprefix("dice_").replace("_", " "), "n_folds": len(values),
                       "mean_dice": average, "sd": sd, "ci95_low": max(0.0, average-margin),
                       "ci95_high": min(1.0, average+margin)})
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Build aggregate publication results from a PerthesMetrics model zip.")
    parser.add_argument("--model-zip", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "segmentation" / "nnunet2d")
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    fold_rows: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    with zipfile.ZipFile(args.model_zip.resolve()) as archive:
        names = set(archive.namelist())
        ROOT = archive_root(names)
        dataset = json.loads(archive.read(ROOT + "dataset.json"))
        plans = json.loads(archive.read(ROOT + "plans.json"))
        write_json(destination / "dataset.json", dataset)
        two_d = plans.get("configurations", {}).get("2d", {})
        write_json(destination / "plans_summary.json", {
            "dataset_name": plans.get("dataset_name"), "plans_name": plans.get("plans_name"),
            "configuration": "2d", "patch_size": two_d.get("patch_size"), "batch_size": two_d.get("batch_size"),
            "labels": dataset.get("labels"), "channel_names": dataset.get("channel_names"),
        })
        for fold in range(5):
            fold_name = f"fold_{fold}"
            summary_name = ROOT + f"{fold_name}/validation/summary.json"
            if summary_name not in names:
                raise FileNotFoundError(f"Archive lacks {summary_name}")
            aggregate = aggregate_summary(json.loads(archive.read(summary_name)))
            fold_dir = destination / fold_name; fold_dir.mkdir()
            write_json(fold_dir / "validation_summary_aggregate.json", aggregate)
            rows = metric_rows(fold_name, aggregate)
            write_csv(fold_dir / "validation_summary_metrics.csv", rows, ["fold", "scope", "label", "metric", "value"])
            all_metrics.extend(rows); fold_rows.append(dice_row(fold_name, aggregate))
            progress = ROOT + f"{fold_name}/progress.png"
            if progress in names:
                (fold_dir / "progress.png").write_bytes(archive.read(progress))
        crossval = ROOT + "crossval_results_folds_0_1_2_3_4/summary.json"
        if crossval in names:
            aggregate = aggregate_summary(json.loads(archive.read(crossval)))
            write_json(destination / "cross_validation_summary_aggregate.json", aggregate)
            rows = metric_rows("cross_validation", aggregate)
            write_csv(destination / "cross_validation_summary_metrics.csv", rows, ["fold", "scope", "label", "metric", "value"])
            all_metrics.extend(rows); fold_rows.append(dice_row("cross_validation", aggregate))
        postprocessing = ROOT + "crossval_results_folds_0_1_2_3_4/postprocessing.json"
        if postprocessing in names:
            write_json(destination / "postprocessing.json", json.loads(archive.read(postprocessing)))
    dice_fields = ["fold", "foreground_mean_dice", *[f"dice_{label}" for label in LABELS.values()]]
    write_csv(destination / "fold_validation_dice_summary.csv", fold_rows, dice_fields)
    write_csv(destination / "all_fold_validation_summary_metrics.csv", all_metrics, ["fold", "scope", "label", "metric", "value"])
    statistics_rows = cross_fold_statistics(fold_rows[:5])
    write_csv(destination / "cross_fold_dice_statistics.csv", statistics_rows,
              ["region", "n_folds", "mean_dice", "sd", "ci95_low", "ci95_high"])
    plot_dice(destination / "validation_dice_summary.png", fold_rows)
    print(f"Wrote sanitized aggregate results to: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
