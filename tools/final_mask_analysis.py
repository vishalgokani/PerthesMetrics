"""Final held-out mask analysis for PerthesMetrics.

The analysis unit is the patient. Image-level mask overlaps are pooled within
each patient/stratum before patient IDs are bootstrapped for confidence
intervals.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image


IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
OVERLAY_CLASS_ORDER = [
    "acetabulum",
    "head",
    "neck",
    "shaft",
    "sourcil",
    "triradiate cartilage",
    "lt",
    "gt",
]
ANALYSIS_MASK_ORDER = [
    "acetabulum",
    "head",
    "neck_shaft",
    "sourcil",
    "triradiate cartilage",
    "lt",
    "gt",
]
FIGURE3_MASK_ORDER = [
    "head",
    "neck_shaft",
    "gt",
    "lt",
    "sourcil",
    "triradiate cartilage",
    "acetabulum",
]
FROG_PLOT_MASK_ORDER = [
    "head",
    "neck_shaft",
    "sourcil",
    "triradiate cartilage",
    "acetabulum",
]
CLASS_ORDER = OVERLAY_CLASS_ORDER
CLASS_DISPLAY = {
    "acetabulum": "Acetabulum",
    "head": "Femoral Head",
    "neck": "Femoral Neck",
    "shaft": "Femoral Shaft",
    "neck_shaft": "Femoral Neck/Shaft",
    "sourcil": "Sourcil",
    "triradiate cartilage": "Triradiate Cartilage",
    "lt": "Lesser Trochanter",
    "gt": "Greater Trochanter",
}
CLASS_COLORS = {
    "acetabulum": (255, 0, 0),
    "head": (0, 255, 255),
    "neck": (245, 170, 66),
    "shaft": (0, 0, 128),
    "neck_shaft": (230, 159, 0),
    "sourcil": (65, 105, 225),
    "triradiate cartilage": (184, 255, 133),
    "lt": (0, 255, 0),
    "gt": (160, 32, 255),
}
WALDENSTROM_STAGES = {
    "0": "Ia",
    "1": "Ib",
    "2": "IIa",
    "3": "IIb",
    "4": "IIIa",
    "5": "IIIb",
    "6": "IV",
}
STAGE_ORDER = ["Ia", "Ib", "IIa", "IIb", "IIIa", "IIIb", "IV"]
ANALYSIS_GROUP_ORDER = ["Unaffected", *STAGE_ORDER]
VIEW_ORDER = ["ap", "frog"]
PUBLICATION_FONT = "Times New Roman"
matplotlib.rcParams.update(
    {
        "font.family": PUBLICATION_FONT,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


@dataclass(frozen=True)
class ImageMaskDice:
    filename: str
    patient_id: str
    view: str
    affected_status: str
    analysis_group: str
    waldenstrom_class: str
    waldenstrom_stage: str
    mask: str
    dice: float
    intersection_pixels: int
    gt_pixels: int
    pred_pixels: int


@dataclass(frozen=True)
class PatientMaskDice:
    scope: str
    patient_id: str
    view: str
    affected_status: str
    analysis_group: str
    waldenstrom_class: str
    waldenstrom_stage: str
    mask: str
    dice: float
    n_images: int
    intersection_pixels: int
    gt_pixels: int
    pred_pixels: int


def normalized_name(value: str) -> str:
    return re.sub(r"[\s_-]+", " ", value.strip().lower())


def class_directories(mask_root: Path) -> dict[str, Path]:
    if not mask_root.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_root}")
    folders = {normalized_name(path.name): path for path in mask_root.iterdir() if path.is_dir()}
    output: dict[str, Path] = {}
    missing: list[str] = []
    for name in CLASS_ORDER:
        folder = folders.get(normalized_name(name))
        if folder is None:
            missing.append(name)
        else:
            output[name] = folder
    if missing:
        raise FileNotFoundError(f"{mask_root} is missing mask folders: {missing}")
    return output


def source_images(data_dir: Path) -> list[Path]:
    images = sorted(
        path
        for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No radiograph image files found directly in {data_dir}")
    return images


def extract_patient_id(filename: str, patient_regex: str) -> str:
    match = re.search(patient_regex, filename, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not extract patient ID from filename: {filename}")
    if "patient_id" in match.groupdict():
        return match.group("patient_id")
    return match.group(1)


def infer_view(filename: str) -> str:
    tokens = re.split(r"[^A-Za-z0-9]+", Path(filename).stem.lower())
    if "ap" in tokens:
        return "ap"
    if "frog" in tokens:
        return "frog"
    return ""


def filename_key(filename: str) -> str:
    return filename.strip().lower()


def detect_field(fields: Iterable[str], candidates: set[str]) -> str:
    for field in fields:
        if normalized_name(field).replace(" ", "_") in candidates:
            return field
    raise ValueError(f"Could not find one of {sorted(candidates)} in CSV fields {list(fields)}")


def normalize_view(value: str) -> str:
    value = value.strip().lower()
    if value == "ap":
        return "ap"
    if value == "frog":
        return "frog"
    return value


def normalize_affected_status(value: str) -> str:
    normalized = normalized_name(value)
    if normalized == "affected":
        return "Affected"
    if normalized == "unaffected":
        return "Unaffected"
    if is_unknown_text(value):
        return "Unknown"
    return value.strip()


def is_unknown_text(value: str) -> bool:
    return normalized_name(value) in {"", "na", "n/a", "nan", "none", "unknown"}


def normalize_waldenstrom_stage(value: str) -> str:
    value = value.strip()
    if is_unknown_text(value):
        return ""
    return WALDENSTROM_STAGES.get(value, value)


def default_radiographdata_csv(data_dir: Path) -> Path:
    candidates = [
        data_dir / "radiographdata.csv",
        data_dir.parent / "radiographdata.csv",
        data_dir.parent.parent / "radiographdata.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[-1]


def read_stage_csv(path: Path, view: str) -> dict[str, tuple[str, str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Waldenstrom class CSV not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"No header found in {path}")
        filename_field = detect_field(reader.fieldnames, {"filename", "file", "image", "image_name"})
        class_field = detect_field(
            reader.fieldnames,
            {"class", "label", "stage", "waldenstrom_class", "waldenstrom_stage"},
        )
        output: dict[str, tuple[str, str, str]] = {}
        for row in reader:
            filename = (row.get(filename_field) or "").strip()
            raw_class = (row.get(class_field) or "").strip()
            if not filename:
                continue
            stage = normalize_waldenstrom_stage(raw_class)
            key = filename_key(filename)
            previous = output.get(key)
            value = (view, raw_class, stage)
            if previous and previous != value:
                raise ValueError(f"Conflicting Waldenstrom labels for {filename} in {path}")
            output[key] = value
    return output


def read_stage_maps(ap_csv: Path, frog_csv: Path) -> dict[str, tuple[str, str, str]]:
    mapping: dict[str, tuple[str, str, str]] = {}
    for view, path in (("ap", ap_csv), ("frog", frog_csv)):
        for filename, value in read_stage_csv(path, view).items():
            previous = mapping.get(filename)
            if previous and previous != value:
                raise ValueError(f"Conflicting Waldenstrom labels for {filename}")
            mapping[filename] = value
    return mapping


def read_radiograph_metadata(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Radiograph metadata CSV not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"No header found in {path}")
        filename_field = detect_field(reader.fieldnames, {"file_name", "filename", "file", "image", "image_name"})
        affected_field = detect_field(reader.fieldnames, {"affected_vs_unaffected", "affected_unaffected"})
        view_field = ""
        stage_field = ""
        try:
            view_field = detect_field(reader.fieldnames, {"ap_or_frog", "ap_or_frog", "view", "ap_frog"})
        except ValueError:
            view_field = ""
        try:
            stage_field = detect_field(
                reader.fieldnames,
                {"standardized_waldenstrom", "waldenstrom_stage", "waldenstrom"},
            )
        except ValueError:
            stage_field = ""

        output: dict[str, dict[str, str]] = {}
        for row in reader:
            filename = (row.get(filename_field) or "").strip()
            if not filename:
                continue
            key = filename_key(filename)
            affected_status = normalize_affected_status(row.get(affected_field) or "")
            if affected_status not in {"Affected", "Unaffected", "Unknown"}:
                raise ValueError(
                    f"Expected affected_vs_unaffected to be Affected, Unaffected, or NA for {filename}; "
                    f"found {affected_status!r}"
                )
            output[key] = {
                "filename": filename,
                "affected_status": affected_status,
                "affected_status_raw": (row.get(affected_field) or "").strip(),
                "view": normalize_view(row.get(view_field) or "") if view_field else "",
                "waldenstrom_stage": normalize_waldenstrom_stage(row.get(stage_field) or "") if stage_field else "",
            }
    if not output:
        raise ValueError(f"No radiograph metadata rows found in {path}")
    return output


def load_binary_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8))
    return np.asarray(image.resize(size, Image.Resampling.NEAREST)) > 0


def load_analysis_mask(
    class_paths: dict[str, Path],
    analysis_mask: str,
    filename: str,
    target_shape: tuple[int, int] | None,
    allow_resize: bool,
) -> tuple[np.ndarray | None, list[str], list[str]]:
    source_classes = ["neck", "shaft"] if analysis_mask == "neck_shaft" else [analysis_mask]
    loaded_masks: list[np.ndarray] = []
    missing: list[str] = []
    mismatched: list[str] = []
    reference_shape = target_shape
    for class_name in source_classes:
        mask_path = class_paths[class_name] / filename
        if not mask_path.is_file():
            missing.append(f"{filename}:{class_name}")
            continue
        mask = load_binary_mask(mask_path)
        if reference_shape is None:
            reference_shape = mask.shape
        if mask.shape != reference_shape:
            if allow_resize:
                mask = resize_mask(mask, (reference_shape[1], reference_shape[0]))
            else:
                mismatched.append(f"{filename}:{class_name} expected={reference_shape} found={mask.shape}")
                continue
        loaded_masks.append(mask)
    if missing or mismatched:
        return None, missing, mismatched
    combined = np.zeros(loaded_masks[0].shape, dtype=bool)
    for mask in loaded_masks:
        combined |= mask
    return combined, [], []


def dice_from_counts(intersection: int, gt_pixels: int, pred_pixels: int) -> float:
    denominator = gt_pixels + pred_pixels
    if denominator == 0:
        return math.nan
    return 2.0 * intersection / denominator


def metrics_from_counts(intersection: int, gt_pixels: int, pred_pixels: int) -> dict[str, float]:
    """Return overlap metrics using pooled pixels from one patient/stratum."""
    union = gt_pixels + pred_pixels - intersection
    return {
        "dice": dice_from_counts(intersection, gt_pixels, pred_pixels),
        "iou": intersection / union if union else math.nan,
        "precision": intersection / pred_pixels if pred_pixels else (0.0 if gt_pixels else math.nan),
        "recall": intersection / gt_pixels if gt_pixels else (0.0 if pred_pixels else math.nan),
    }


def collect_image_mask_dice(
    data_dir: Path,
    gt_dir: Path,
    pred_dir: Path,
    stage_map: dict[str, tuple[str, str, str]],
    radiograph_metadata: dict[str, dict[str, str]],
    patient_regex: str,
    allow_resize_for_metrics: bool,
) -> list[ImageMaskDice]:
    gt_classes = class_directories(gt_dir)
    pred_classes = class_directories(pred_dir)
    rows: list[ImageMaskDice] = []
    missing_masks: list[str] = []
    shape_mismatches: list[str] = []

    for image_path in source_images(data_dir):
        patient_id = extract_patient_id(image_path.name, patient_regex)
        metadata = radiograph_metadata.get(filename_key(image_path.name))
        if metadata is None:
            raise ValueError(f"{image_path.name} is missing from radiograph metadata CSV")
        affected_status = metadata["affected_status"]
        inferred = infer_view(image_path.name)
        csv_view, stage_class, stage = stage_map.get(filename_key(image_path.name), ("", "", ""))
        view = normalize_view(csv_view or metadata.get("view", "") or inferred)
        if view not in VIEW_ORDER:
            raise ValueError(f"Could not determine AP/frog view for {image_path.name}")
        if affected_status == "Unaffected":
            analysis_group = "Unaffected"
            stage = ""
            stage_class = ""
        elif affected_status == "Unknown":
            analysis_group = ""
            stage = ""
            stage_class = ""
        else:
            stage = normalize_waldenstrom_stage(stage or metadata.get("waldenstrom_stage", ""))
            if stage in STAGE_ORDER:
                analysis_group = stage
            else:
                analysis_group = ""
                stage = ""
                stage_class = ""
        for mask_name in ANALYSIS_MASK_ORDER:
            gt_mask, missing, mismatched = load_analysis_mask(
                gt_classes,
                mask_name,
                image_path.name,
                None,
                allow_resize_for_metrics,
            )
            missing_masks.extend(missing)
            shape_mismatches.extend(mismatched)
            if gt_mask is None:
                continue
            pred_mask, missing, mismatched = load_analysis_mask(
                pred_classes,
                mask_name,
                image_path.name,
                gt_mask.shape,
                allow_resize_for_metrics,
            )
            missing_masks.extend(missing)
            shape_mismatches.extend(mismatched)
            if pred_mask is None:
                continue
            intersection = int(np.count_nonzero(gt_mask & pred_mask))
            gt_pixels = int(np.count_nonzero(gt_mask))
            pred_pixels = int(np.count_nonzero(pred_mask))
            rows.append(
                ImageMaskDice(
                    filename=image_path.name,
                    patient_id=patient_id,
                    view=view,
                    affected_status=affected_status,
                    analysis_group=analysis_group,
                    waldenstrom_class=stage_class,
                    waldenstrom_stage=stage,
                    mask=mask_name,
                    dice=dice_from_counts(intersection, gt_pixels, pred_pixels),
                    intersection_pixels=intersection,
                    gt_pixels=gt_pixels,
                    pred_pixels=pred_pixels,
                )
            )

    if missing_masks:
        raise FileNotFoundError(f"Missing {len(missing_masks)} masks; examples: {missing_masks[:8]}")
    if shape_mismatches:
        raise ValueError(
            "Ground-truth and prediction mask shapes differ. "
            f"Use --allow-resize-for-metrics only after verifying this is intended. Examples: {shape_mismatches[:8]}"
        )
    if not rows:
        raise RuntimeError("No image/mask Dice rows were created.")
    return rows


def pool_by_patient(
    rows: list[ImageMaskDice],
    scope: str,
    group_fields: tuple[str, ...],
    require_stage: bool = False,
) -> list[PatientMaskDice]:
    grouped: dict[tuple[str, ...], dict[str, object]] = {}
    for row in rows:
        if require_stage and not row.waldenstrom_stage:
            continue
        values = {
            "patient_id": row.patient_id,
            "view": row.view,
            "affected_status": row.affected_status,
            "analysis_group": row.analysis_group,
            "waldenstrom_class": row.waldenstrom_class,
            "waldenstrom_stage": row.waldenstrom_stage,
            "mask": row.mask,
        }
        key = tuple(str(values[field]) for field in (*group_fields, "patient_id", "mask"))
        keep_stage = "waldenstrom_stage" in group_fields or "analysis_group" in group_fields
        bucket = grouped.setdefault(
            key,
            {
                "view": row.view if "view" in group_fields else "",
                "affected_status": row.affected_status if "affected_status" in group_fields else "",
                "analysis_group": row.analysis_group if "analysis_group" in group_fields else "",
                "waldenstrom_class": row.waldenstrom_class if keep_stage else "",
                "waldenstrom_stage": row.waldenstrom_stage if keep_stage else "",
                "intersection_pixels": 0,
                "gt_pixels": 0,
                "pred_pixels": 0,
                "filenames": set(),
            },
        )
        bucket["intersection_pixels"] = int(bucket["intersection_pixels"]) + row.intersection_pixels
        bucket["gt_pixels"] = int(bucket["gt_pixels"]) + row.gt_pixels
        bucket["pred_pixels"] = int(bucket["pred_pixels"]) + row.pred_pixels
        filenames = bucket["filenames"]
        assert isinstance(filenames, set)
        filenames.add(row.filename)

    output: list[PatientMaskDice] = []
    for key, bucket in grouped.items():
        patient_id = key[-2]
        mask_name = key[-1]
        intersection = int(bucket["intersection_pixels"])
        gt_pixels = int(bucket["gt_pixels"])
        pred_pixels = int(bucket["pred_pixels"])
        filenames = bucket["filenames"]
        assert isinstance(filenames, set)
        output.append(
            PatientMaskDice(
                scope=scope,
                patient_id=patient_id,
                view=str(bucket["view"]),
                affected_status=str(bucket["affected_status"]),
                analysis_group=str(bucket["analysis_group"]),
                waldenstrom_class=str(bucket["waldenstrom_class"]),
                waldenstrom_stage=str(bucket["waldenstrom_stage"]),
                mask=mask_name,
                dice=dice_from_counts(intersection, gt_pixels, pred_pixels),
                n_images=len(filenames),
                intersection_pixels=intersection,
                gt_pixels=gt_pixels,
                pred_pixels=pred_pixels,
            )
        )
    return sorted(output, key=lambda r: (r.scope, r.analysis_group, r.waldenstrom_stage, r.view, r.mask, r.patient_id))


def finite_values(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def bootstrap_ci(values: np.ndarray, iterations: int, rng: np.random.Generator) -> tuple[float, float]:
    if values.size == 0:
        return math.nan, math.nan
    if values.size == 1 or iterations <= 0:
        value = float(np.mean(values))
        return value, value
    samples = rng.choice(values, size=(iterations, values.size), replace=True)
    means = samples.mean(axis=1)
    return tuple(float(value) for value in np.percentile(means, [2.5, 97.5]))


def summarize_global_patient_mask_dice(
    rows: list[PatientMaskDice],
    iterations: int,
    seed: int,
) -> dict[str, object]:
    """Summarize all patient-mask Dice values with a patient-clustered CI."""
    values_by_patient: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if math.isfinite(row.dice):
            values_by_patient[row.patient_id].append(row.dice)
    if not values_by_patient:
        raise ValueError("No finite patient-level Dice values were available for the global summary.")

    patient_ids = sorted(values_by_patient)
    patient_values = [np.asarray(values_by_patient[patient_id], dtype=float) for patient_id in patient_ids]
    pooled_values = np.concatenate(patient_values)
    median = float(np.median(pooled_values))

    if len(patient_values) == 1 or iterations <= 0:
        ci_low = median
        ci_high = median
    else:
        rng = np.random.default_rng(seed)
        bootstrap_medians = np.empty(iterations, dtype=float)
        for iteration in range(iterations):
            sampled_indices = rng.integers(0, len(patient_values), size=len(patient_values))
            sampled_values = np.concatenate([patient_values[index] for index in sampled_indices])
            bootstrap_medians[iteration] = np.median(sampled_values)
        ci_low, ci_high = (
            float(value) for value in np.percentile(bootstrap_medians, [2.5, 97.5])
        )

    return {
        "scope": "global_all_masks",
        "n_patients": len(patient_ids),
        "n_patient_mask_observations": int(pooled_values.size),
        "median_patient_pooled_dice": median,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "bootstrap_iterations": iterations,
    }


def summarize_patient_rows(
    rows: list[PatientMaskDice],
    output_scope: str,
    group_fields: tuple[str, ...],
    cutoffs: list[float],
    iterations: int,
    seed: int,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[PatientMaskDice]] = defaultdict(list)
    for row in rows:
        values = {
            "view": row.view,
            "affected_status": row.affected_status,
            "analysis_group": row.analysis_group,
            "waldenstrom_class": row.waldenstrom_class,
            "waldenstrom_stage": row.waldenstrom_stage,
            "mask": row.mask,
        }
        key = tuple(str(values[field]) for field in (*group_fields, "mask"))
        grouped[key].append(row)

    summaries: list[dict[str, object]] = []
    for index, (key, group_rows) in enumerate(sorted(grouped.items())):
        rng = np.random.default_rng(seed + index)
        values = finite_values(row.dice for row in group_rows)
        ci_low, ci_high = bootstrap_ci(values, iterations, rng)
        summary: dict[str, object] = {"scope": output_scope}
        for field, value in zip((*group_fields, "mask"), key):
            summary[field] = value
        summary.update(
            {
                "mask_display": CLASS_DISPLAY.get(str(summary["mask"]), str(summary["mask"])),
                "n_patients": int(values.size),
                "n_images": sum(row.n_images for row in group_rows if math.isfinite(row.dice)),
                "mean_patient_pooled_dice": float(np.mean(values)) if values.size else math.nan,
                "median_patient_pooled_dice": float(np.median(values)) if values.size else math.nan,
                "sd_patient_pooled_dice": float(np.std(values, ddof=1)) if values.size > 1 else math.nan,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "bootstrap_iterations": iterations,
            }
        )
        for cutoff in cutoffs:
            successes = (values >= cutoff).astype(float)
            pass_low, pass_high = bootstrap_ci(successes, iterations, rng)
            suffix = f"{cutoff:.2f}".replace(".", "_")
            summary[f"n_patients_ge_{suffix}"] = int(np.count_nonzero(successes))
            summary[f"pass_rate_ge_{suffix}"] = float(np.mean(successes)) if successes.size else math.nan
            summary[f"pass_rate_ge_{suffix}_ci95_low"] = pass_low
            summary[f"pass_rate_ge_{suffix}_ci95_high"] = pass_high
        summaries.append(summary)
    return summaries


def summarize_overall_metrics(
    rows: list[PatientMaskDice],
    iterations: int,
    seed: int,
) -> list[dict[str, object]]:
    """Summarize patient-pooled Dice, IoU, precision, and recall by mask."""
    grouped: dict[str, list[PatientMaskDice]] = defaultdict(list)
    for row in rows:
        grouped[row.mask].append(row)
    output: list[dict[str, object]] = []
    for mask_index, mask_name in enumerate(ANALYSIS_MASK_ORDER):
        group_rows = grouped.get(mask_name, [])
        summary: dict[str, object] = {
            "mask": mask_name,
            "mask_display": CLASS_DISPLAY[mask_name],
        }
        patient_metrics = [
            metrics_from_counts(row.intersection_pixels, row.gt_pixels, row.pred_pixels)
            for row in group_rows
        ]
        dice_values = finite_values(metrics["dice"] for metrics in patient_metrics)
        summary["n_patients"] = int(dice_values.size)
        for metric_index, metric_name in enumerate(("dice", "iou", "precision", "recall")):
            values = finite_values(
                metrics[metric_name]
                for metrics in patient_metrics
                if math.isfinite(metrics["dice"])
            )
            rng = np.random.default_rng(seed + mask_index * 10 + metric_index)
            ci_low, ci_high = bootstrap_ci(values, iterations, rng)
            summary[f"mean_patient_pooled_{metric_name}"] = float(np.mean(values)) if values.size else math.nan
            summary[f"{metric_name}_ci95_low"] = ci_low
            summary[f"{metric_name}_ci95_high"] = ci_high
        summary["bootstrap_iterations"] = iterations
        output.append(summary)
    return output


def analysis_group_counts(rows: list[ImageMaskDice]) -> list[dict[str, object]]:
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        if row.analysis_group and row.view:
            seen.add((row.analysis_group, row.view, row.patient_id, row.filename))
    grouped: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: {"patients": set(), "images": set()})
    for analysis_group, view, patient_id, filename in seen:
        grouped[(analysis_group, view)]["patients"].add(patient_id)
        grouped[(analysis_group, view)]["images"].add(filename)
    output = []
    for analysis_group in ANALYSIS_GROUP_ORDER:
        for view in VIEW_ORDER:
            bucket = grouped.get((analysis_group, view), {"patients": set(), "images": set()})
            output.append(
                {
                    "analysis_group": analysis_group,
                    "view": view,
                    "n_patients": len(bucket["patients"]),
                    "n_images": len(bucket["images"]),
                }
            )
    return output


def unknown_affected_status_rows(rows: list[ImageMaskDice]) -> list[dict[str, object]]:
    seen: dict[str, ImageMaskDice] = {}
    for row in rows:
        if row.affected_status == "Unknown":
            seen[row.filename] = row
    return [
        {
            "filename": row.filename,
            "patient_id": row.patient_id,
            "view": row.view,
            "affected_status": row.affected_status,
            "excluded_from": "affected_status_and_analysis_group_stratified_outputs",
            "reason": "affected_vs_unaffected is NA or missing in radiographdata.csv",
        }
        for row in sorted(seen.values(), key=lambda item: item.filename)
    ]


def missing_affected_stage_rows(rows: list[ImageMaskDice]) -> list[dict[str, object]]:
    seen: dict[str, ImageMaskDice] = {}
    for row in rows:
        if row.affected_status == "Affected" and not row.analysis_group:
            seen[row.filename] = row
    return [
        {
            "filename": row.filename,
            "patient_id": row.patient_id,
            "view": row.view,
            "affected_status": row.affected_status,
            "excluded_from": "waldenstrom_stage_and_analysis_group_stratified_outputs",
            "reason": "affected hip has no usable Waldenstrom stage in class CSVs or radiographdata.csv",
        }
        for row in sorted(seen.values(), key=lambda item: item.filename)
    ]


def format_decimal(value: object, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{float(value):.{digits}f}"


def format_percent(value: object, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{100 * float(value):.{digits}f}%"


def cutoff_suffix(cutoff: float) -> str:
    return f"{cutoff:.2f}".replace(".", "_")


def mask_sort_key(value: object) -> int:
    try:
        return ANALYSIS_MASK_ORDER.index(str(value))
    except ValueError:
        return len(ANALYSIS_MASK_ORDER)


def analysis_group_sort_key(value: object) -> int:
    try:
        return ANALYSIS_GROUP_ORDER.index(str(value))
    except ValueError:
        return len(ANALYSIS_GROUP_ORDER)


def plot_mask_order(view: str) -> list[str]:
    """Return structures appropriate for a radiograph view."""
    return FROG_PLOT_MASK_ORDER if view == "frog" else FIGURE3_MASK_ORDER


def format_metric_ci(row: dict[str, object], metric: str, digits: int = 2) -> str:
    return (
        f"{format_decimal(row[f'mean_patient_pooled_{metric}'], digits)} "
        f"[{format_decimal(row[f'{metric}_ci95_low'], digits)}-"
        f"{format_decimal(row[f'{metric}_ci95_high'], digits)}]"
    )


def publication_overall_rows(summaries: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "Hip Structure": row["mask_display"],
            "n": row["n_patients"],
            "Dice": format_metric_ci(row, "dice"),
            "IoU": format_metric_ci(row, "iou"),
            "Precision": format_metric_ci(row, "precision"),
            "Recall": format_metric_ci(row, "recall"),
        }
        for row in sorted(summaries, key=lambda item: mask_sort_key(item.get("mask")))
    ]


def publication_group_rows(
    summaries: list[dict[str, object]],
    group_field: str,
    groups: list[str],
    group_labels: dict[str, str],
) -> list[dict[str, object]]:
    indexed = {(str(row.get("mask")), str(row.get(group_field))): row for row in summaries}
    output: list[dict[str, object]] = []
    for mask_name in ANALYSIS_MASK_ORDER:
        formatted: dict[str, object] = {"Hip Structure": CLASS_DISPLAY[mask_name]}
        for group in groups:
            label = group_labels[group]
            row = indexed.get((mask_name, group))
            formatted[label] = (
                f"{format_decimal(row['mean_patient_pooled_dice'], 2)} "
                f"[{format_decimal(row['ci95_low'], 2)}-{format_decimal(row['ci95_high'], 2)}] "
                f"(n={row['n_patients']})"
                if row else "NA (n=0)"
            )
        output.append(formatted)
    return output


def write_markdown_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def write_latex_table(path: Path, rows: list[dict[str, object]], caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    column_spec = "l" + "c" * (len(fields) - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{latex_escape(label)}}}",
        f"\\begin{{tabular}}{{{column_spec}}}",
        r"\hline",
        " & ".join(latex_escape(field) for field in fields) + r" \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(row.get(field, "")) for field in fields) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_publication_table_figure(
    path: Path,
    rows: list[dict[str, object]],
    title: str,
) -> None:
    """Render a clean, editable vector table as SVG and PDF."""
    if not rows:
        return
    fields = list(rows[0])
    is_wide_table = len(fields) > 6
    cell_text = []
    for row in rows:
        cells = [str(row.get(field, "")) for field in fields]
        if is_wide_table:
            cells = [cell.replace(" (n=", "\n(n=") for cell in cells]
        cell_text.append(cells)
    max_lengths = [
        max(len(str(field)), *(len(row[column]) for row in cell_text))
        for column, field in enumerate(fields)
    ]
    width_ratios = np.asarray([max(8, length) for length in max_lengths], dtype=float)
    width_ratios[0] *= 1.45
    column_widths = width_ratios / width_ratios.sum()
    figure_width = 12.5 if is_wide_table else max(8.5, min(11.0, float(width_ratios.sum()) * 0.09))
    figure_height = 4.2 if is_wide_table else 1.15 + 0.44 * len(rows)
    with plt.rc_context({"font.family": PUBLICATION_FONT, "svg.fonttype": "none", "pdf.fonttype": 42}):
        figure, axis = plt.subplots(figsize=(figure_width, figure_height))
        figure.subplots_adjust(left=0.015, right=0.985, top=0.92, bottom=0.03)
        axis.axis("off")
        axis.set_title(title, fontsize=14, fontweight="bold", pad=5)
        table = axis.table(
            cellText=cell_text,
            colLabels=fields,
            cellLoc="center",
            colLoc="center",
            colWidths=column_widths,
            bbox=[0.0, 0.02, 1.0, 0.86],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        for (row_index, column_index), cell in table.get_celld().items():
            cell.set_facecolor("white")
            cell.set_edgecolor("black")
            cell.set_linewidth(0.0)
            if row_index == 0:
                cell.set_text_props(weight="bold", fontsize=11)
                cell.visible_edges = "B"
                cell.set_linewidth(0.8)
            elif row_index == len(rows):
                cell.visible_edges = "B"
                cell.set_linewidth(0.8)
            if column_index == 0:
                cell.get_text().set_ha("left")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            atomic_savefig(figure, path.with_suffix(".pdf"), bbox_inches="tight")
            atomic_savefig(figure, path.with_suffix(".svg"), bbox_inches="tight")
            atomic_savefig(figure, path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        finally:
            plt.close(figure)


def write_publication_table_bundle(
    base_path: Path,
    rows: list[dict[str, object]],
    title: str,
    caption: str,
    label: str,
) -> None:
    write_csv(base_path.with_suffix(".csv"), rows)
    write_markdown_table(base_path.with_suffix(".md"), rows)
    write_latex_table(base_path.with_suffix(".tex"), rows, caption, label)
    write_publication_table_figure(base_path, rows, title)


def write_csv(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if hasattr(rows[0], "__dataclass_fields__"):
        dictionaries = [row.__dict__ for row in rows]
    else:
        dictionaries = [dict(row) for row in rows]  # type: ignore[arg-type]
    fields: list[str] = []
    for row in dictionaries:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in dictionaries:
            writer.writerow({field: "" if is_missing(row.get(field)) else row.get(field) for field in fields})


def is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def rgb01(color: tuple[int, int, int]) -> tuple[float, float, float]:
    return tuple(channel / 255.0 for channel in color)


def atomic_savefig(figure: plt.Figure, path: Path, **kwargs: object) -> None:
    """Write a figure completely before atomically replacing its destination."""
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


def plot_stage_view_boxplots(
    path: Path,
    rows: list[PatientMaskDice],
    seed: int,
    cutoff: float | None = None,
    tiff_dpi: int = 600,
) -> None:
    filtered = [row for row in rows if row.analysis_group in ANALYSIS_GROUP_ORDER and row.view in VIEW_ORDER]
    if not filtered:
        return
    rng = np.random.default_rng(seed)
    figure, axes = plt.subplots(
        len(ANALYSIS_GROUP_ORDER),
        len(VIEW_ORDER),
        figsize=(10.5, 1.8 * len(ANALYSIS_GROUP_ORDER)),
        sharex=False,
        sharey=True,
    )
    figure.subplots_adjust(left=0.09, right=0.99, top=0.96, bottom=0.13, hspace=0.30, wspace=0.12)
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in filtered:
        if math.isfinite(row.dice):
            grouped[(row.analysis_group, row.view, row.mask)].append(row.dice)

    # Keep each mask's point jitter unchanged when the display order changes.
    jitter_by_group: dict[tuple[str, str, str], np.ndarray] = {}
    for analysis_group in ANALYSIS_GROUP_ORDER:
        for view in VIEW_ORDER:
            original_order = (
                ["acetabulum", "head", "neck_shaft", "sourcil", "triradiate cartilage"]
                if view == "frog" else ANALYSIS_MASK_ORDER
            )
            for mask_name in original_order:
                values = grouped.get((analysis_group, view, mask_name), [])
                if values:
                    jitter_by_group[(analysis_group, view, mask_name)] = rng.uniform(
                        -0.16, 0.16, size=len(values)
                    )

    for row_index, analysis_group in enumerate(ANALYSIS_GROUP_ORDER):
        for col_index, view in enumerate(VIEW_ORDER):
            axis = axes[row_index, col_index]
            mask_order = plot_mask_order(view)
            values_by_mask = [grouped.get((analysis_group, view, mask_name), []) for mask_name in mask_order]
            positions = np.arange(1, len(mask_order) + 1)
            for position, mask_name, values in zip(positions, mask_order, values_by_mask):
                if not values:
                    continue
                jitter = jitter_by_group[(analysis_group, view, mask_name)]
                axis.scatter(
                    np.full(len(values), position) + jitter,
                    values,
                    s=13,
                    color=rgb01(CLASS_COLORS[mask_name]),
                    edgecolors="black",
                    linewidths=0.25,
                    alpha=0.6,
                    zorder=2,
                )
            boxplot = axis.boxplot(
                values_by_mask,
                positions=positions,
                widths=0.55,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.6},
                boxprops={"facecolor": (1.0, 1.0, 1.0, 0.35), "edgecolor": "black", "linewidth": 1.1},
                whiskerprops={"color": "black", "linewidth": 1.1},
                capprops={"color": "black", "linewidth": 1.1},
            )
            for element in ("boxes", "whiskers", "caps", "medians"):
                for artist in boxplot[element]:
                    artist.set_zorder(4)
            if cutoff is not None:
                axis.axhline(cutoff, color="black", linestyle="--", linewidth=0.9)
            axis.set_ylim(-0.02, 1.02)
            axis.set_xlim(0.4, len(mask_order) + 0.6)
            axis.grid(axis="y", color="0.88", linewidth=0.6)
            axis.tick_params(axis="y", labelsize=11)
            if row_index == 0:
                view_title = "AP" if view == "ap" else "Frog-leg lateral"
                axis.set_title(view_title, fontsize=14, fontweight="bold")
            if col_index == 0:
                axis.set_ylabel(f"{analysis_group}\nDSC", fontsize=13)
            if row_index == len(ANALYSIS_GROUP_ORDER) - 1:
                axis.set_xticks(positions)
                axis.set_xticklabels(
                    [CLASS_DISPLAY[name] for name in mask_order],
                    rotation=38,
                    ha="right",
                    fontsize=12,
                )
            else:
                axis.set_xticks(positions)
                axis.tick_params(axis="x", labelbottom=False)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markerfacecolor=rgb01(CLASS_COLORS[name]),
            markeredgecolor="black",
            markersize=7.5,
            label=CLASS_DISPLAY[name],
        )
        for name in FIGURE3_MASK_ORDER
    ]
    figure.suptitle(
        "Patient-pooled Dice by affected status, Waldenstrom stage, and radiograph view",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.005),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        atomic_savefig(figure, path.with_suffix(".pdf"), bbox_inches="tight")
        atomic_savefig(figure, path.with_suffix(".svg"), bbox_inches="tight")
        atomic_savefig(figure, path, dpi=300, bbox_inches="tight")
        atomic_savefig(
            figure,
            path.with_suffix(".tif"),
            dpi=tiff_dpi,
            bbox_inches="tight",
            pil_kwargs={"compression": "raw"},
        )
    finally:
        plt.close(figure)


def paint_solid_mask(canvas: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    active = mask.astype(bool)
    if np.any(active):
        canvas[active] = np.asarray(color, dtype=np.uint8)
    return canvas


def make_overlay(base_rgb: Image.Image, image_path: Path, class_dirs: dict[str, Path]) -> tuple[Image.Image, list[str]]:
    canvas = np.asarray(base_rgb).copy()
    used: list[str] = []
    for class_name in OVERLAY_CLASS_ORDER:
        mask_path = class_dirs[class_name] / image_path.name
        if not mask_path.is_file():
            continue
        mask = load_binary_mask(mask_path)
        if mask.shape != canvas.shape[:2]:
            mask = resize_mask(mask, base_rgb.size)
        if not np.any(mask):
            continue
        canvas = paint_solid_mask(canvas, mask, CLASS_COLORS[class_name])
        used.append(class_name)
    return Image.fromarray(canvas), used


def figure_source_filename(path: Path) -> str:
    stem = path.stem
    for suffix in ("_ground_truth_masks", "_nnunet_masks", "_original"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)] + ".bmp"
    return path.name


def stage_folder_name(raw_class: str, stage: str) -> str:
    if raw_class in WALDENSTROM_STAGES:
        return raw_class
    if stage in STAGE_ORDER:
        return f"{STAGE_ORDER.index(stage)}"
    return "unknown_waldenstrom"


def organize_mask_figures_by_waldenstrom(
    mask_figures_dir: Path,
    stage_map: dict[str, tuple[str, str, str]],
    output_dir: Path,
) -> None:
    if not mask_figures_dir.is_dir():
        raise FileNotFoundError(f"Mask-figure directory not found: {mask_figures_dir}")
    summary_rows: list[dict[str, object]] = []
    for figure_path in sorted(mask_figures_dir.iterdir()):
        if not figure_path.is_file() or figure_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        source_filename = figure_source_filename(figure_path)
        view, raw_class, stage = stage_map.get(filename_key(source_filename), ("", "", ""))
        view = normalize_view(view or infer_view(source_filename) or "unknown_view")
        folder = stage_folder_name(raw_class, stage)
        destination_dir = output_dir / view / folder
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / figure_path.name
        shutil.copy2(figure_path, destination)
        summary_rows.append(
            {
                "figure_file": figure_path.name,
                "source_filename": source_filename,
                "view": view,
                "waldenstrom_class": raw_class,
                "waldenstrom_stage": stage,
                "stratified_path": str(destination),
            }
        )
    write_csv(output_dir / "mask_figure_stratification_summary.csv", summary_rows)


def create_mask_figures(
    data_dir: Path,
    gt_dir: Path,
    pred_dir: Path,
    output_dir: Path,
    stage_map: dict[str, tuple[str, str, str]] | None = None,
) -> None:
    gt_classes = class_directories(gt_dir)
    pred_classes = class_directories(pred_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    for image_path in source_images(data_dir):
        base_rgb = Image.open(image_path).convert("RGB")
        shutil.copy2(image_path, output_dir / f"{image_path.stem}_original{image_path.suffix}")
        gt_overlay, gt_used = make_overlay(base_rgb, image_path, gt_classes)
        pred_overlay, pred_used = make_overlay(base_rgb, image_path, pred_classes)
        if gt_used:
            gt_overlay.save(output_dir / f"{image_path.stem}_ground_truth_masks.bmp")
        if pred_used:
            pred_overlay.save(output_dir / f"{image_path.stem}_nnunet_masks.bmp")
        summary_rows.append(
            {
                "filename": image_path.name,
                "ground_truth_masks": "|".join(gt_used),
                "nnunet_masks": "|".join(pred_used),
            }
        )
    write_csv(output_dir / "mask_figure_summary.csv", summary_rows)
    if stage_map is not None:
        organize_mask_figures_by_waldenstrom(output_dir, stage_map, output_dir.parent / "mask_figures_stratified")


def write_mask_colors(path: Path) -> None:
    write_csv(
        path,
        [
            {
                "usage": "analysis",
                "mask": name,
                "mask_display": CLASS_DISPLAY[name],
                "red": CLASS_COLORS[name][0],
                "green": CLASS_COLORS[name][1],
                "blue": CLASS_COLORS[name][2],
                "hex": "#%02x%02x%02x" % CLASS_COLORS[name],
            }
            for name in ANALYSIS_MASK_ORDER
        ]
        + [
            {
                "usage": "overlay",
                "mask": name,
                "mask_display": CLASS_DISPLAY[name],
                "red": CLASS_COLORS[name][0],
                "green": CLASS_COLORS[name][1],
                "blue": CLASS_COLORS[name][2],
                "hex": "#%02x%02x%02x" % CLASS_COLORS[name],
            }
            for name in OVERLAY_CLASS_ORDER
        ],
    )


def read_csv_dictionaries(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required saved analysis CSV not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def read_patient_mask_dice(path: Path) -> list[PatientMaskDice]:
    rows: list[PatientMaskDice] = []
    for row in read_csv_dictionaries(path):
        rows.append(
            PatientMaskDice(
                scope=row["scope"],
                patient_id=row["patient_id"],
                view=row["view"],
                affected_status=row["affected_status"],
                analysis_group=row["analysis_group"],
                waldenstrom_class=row["waldenstrom_class"],
                waldenstrom_stage=row["waldenstrom_stage"],
                mask=row["mask"],
                dice=float(row["dice"]) if row["dice"].strip() else math.nan,
                n_images=int(row["n_images"]),
                intersection_pixels=int(row["intersection_pixels"]),
                gt_pixels=int(row["gt_pixels"]),
                pred_pixels=int(row["pred_pixels"]),
            )
        )
    return rows


def render_saved_boxplots(output_dir: Path, cutoffs: list[float], seed: int) -> None:
    """Re-render all boxplot variants from the saved patient-level CSV only."""
    patient_rows = read_patient_mask_dice(output_dir / "patient_pooled_dice_by_analysis_group_view.csv")
    plot_stage_view_boxplots(output_dir / "analysis_group_view_boxplots.png", patient_rows, seed)
    for cutoff in sorted({float(cutoff) for cutoff in cutoffs}, reverse=True):
        plot_name = f"analysis_group_view_boxplots_cutoff_{cutoff:.2f}".replace(".", "_")
        plot_stage_view_boxplots(
            output_dir / f"{plot_name}.png",
            patient_rows,
            seed,
            cutoff=cutoff,
        )


def render_saved_publication_outputs(output_dir: Path, cutoffs: list[float], seed: int) -> None:
    """Re-render the stage table and boxplots from existing analysis CSVs only."""
    analysis_group_summaries: list[dict[str, object]] = [
        dict(row)
        for row in read_csv_dictionaries(output_dir / "analysis_group_model_performance_by_mask.csv")
    ]
    waldenstrom_rows = publication_group_rows(
        analysis_group_summaries,
        "analysis_group",
        ANALYSIS_GROUP_ORDER,
        {group: (group if group == "Unaffected" else f"Stage {group}") for group in ANALYSIS_GROUP_ORDER},
    )
    write_publication_table_figure(
        output_dir / "waldenstrom_stage_model_performance_publication_table",
        waldenstrom_rows,
        "Segmentation Performance by Waldenstrom Stage",
    )

    render_saved_boxplots(output_dir, cutoffs, seed)


def print_saved_global_median(output_dir: Path, iterations: int, seed: int) -> None:
    """Print only the global patient-level median and patient-clustered 95% CI."""
    patient_rows = read_patient_mask_dice(output_dir / "patient_pooled_dice_overall.csv")
    summary = summarize_global_patient_mask_dice(patient_rows, iterations, seed)
    print(
        f"{float(summary['median_patient_pooled_dice']):.3f} "
        f"[{float(summary['ci95_low']):.3f}-{float(summary['ci95_high']):.3f}]"
    )


def default_output_dir(data_dir: Path) -> Path:
    return data_dir / "final_mask_analysis_outputs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Patient-bootstrapped final mask analysis.")
    parser.add_argument("--data-dir", type=Path, help="Directory containing root radiographs.")
    parser.add_argument("--ground-truth-dir", type=Path, help="Ground-truth binary mask root. Defaults to DATA_DIR/masks.")
    parser.add_argument("--prediction-dir", type=Path, help="Inference binary mask root. Defaults to DATA_DIR/nnunet_masks.")
    parser.add_argument("--ap-classes-csv", type=Path, help="AP Waldenstrom CSV. Defaults to DATA_DIR/ap_classes.csv.")
    parser.add_argument("--frog-classes-csv", type=Path, help="Frog Waldenstrom CSV. Defaults to DATA_DIR/frog_classes.csv.")
    parser.add_argument(
        "--radiographdata-csv",
        type=Path,
        help="Radiograph metadata CSV with affected_vs_unaffected. Defaults to DATA_DIR/../../radiographdata.csv.",
    )
    parser.add_argument("--output-dir", type=Path, help="Output directory. Defaults to DATA_DIR/final_mask_analysis_outputs.")
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--cutoffs", nargs="+", type=float, default=[0.80])
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Re-render the Waldenstrom table and all boxplots from saved CSVs; do not read masks.",
    )
    parser.add_argument(
        "--render-boxplots-only",
        action="store_true",
        help="Re-render only boxplots from the saved patient-level CSV; do not read masks or rewrite tables.",
    )
    parser.add_argument(
        "--global-median-only",
        action="store_true",
        help=(
            "Print only the global median of all patient-mask Dice values and its patient-clustered "
            "confidence interval "
            "from OUTPUT_DIR/patient_pooled_dice_overall.csv."
        ),
    )
    parser.add_argument("--patient-regex", default=r"Patient[_ -]*(?P<patient_id>\d+)")
    parser.add_argument(
        "--allow-resize-for-metrics",
        action="store_true",
        help="Resize prediction masks to GT shape with nearest-neighbor before computing metrics.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    saved_output_modes = [args.render_only, args.render_boxplots_only, args.global_median_only]
    if sum(saved_output_modes) > 1:
        parser.error("choose only one saved-output mode")
    if not args.cutoffs or any(not math.isfinite(cutoff) or not 0.0 <= cutoff <= 1.0 for cutoff in args.cutoffs):
        parser.error("--cutoffs must contain one or more finite values between 0 and 1")
    if any(saved_output_modes):
        if args.output_dir is None:
            parser.error("saved-output modes require --output-dir")
        output_dir = args.output_dir.resolve()
        if args.global_median_only:
            print_saved_global_median(output_dir, args.bootstrap_iterations, args.seed)
        elif args.render_boxplots_only:
            render_saved_boxplots(output_dir, args.cutoffs, args.seed)
            print(f"Re-rendered boxplots from the existing patient-level CSV in: {output_dir}")
        else:
            render_saved_publication_outputs(output_dir, args.cutoffs, args.seed)
            print(f"Re-rendered publication outputs from existing CSVs in: {output_dir}")
        return 0
    if args.data_dir is None:
        parser.error("--data-dir is required unless a render-only mode is used")
    data_dir = args.data_dir.resolve()
    gt_dir = (args.ground_truth_dir or data_dir / "masks").resolve()
    pred_dir = (args.prediction_dir or data_dir / "nnunet_masks").resolve()
    ap_csv = (args.ap_classes_csv or data_dir / "ap_classes.csv").resolve()
    frog_csv = (args.frog_classes_csv or data_dir / "frog_classes.csv").resolve()
    radiographdata_csv = (args.radiographdata_csv or default_radiographdata_csv(data_dir)).resolve()
    output_dir = (args.output_dir or default_output_dir(data_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_map = read_stage_maps(ap_csv, frog_csv)
    radiograph_metadata = read_radiograph_metadata(radiographdata_csv)
    image_rows = collect_image_mask_dice(
        data_dir=data_dir,
        gt_dir=gt_dir,
        pred_dir=pred_dir,
        stage_map=stage_map,
        radiograph_metadata=radiograph_metadata,
        patient_regex=args.patient_regex,
        allow_resize_for_metrics=args.allow_resize_for_metrics,
    )
    all_patient_rows = pool_by_patient(image_rows, "overall", ())
    view_patient_rows = pool_by_patient(image_rows, "view", ("view",))
    known_status_image_rows = [row for row in image_rows if row.affected_status in {"Affected", "Unaffected"}]
    known_analysis_group_image_rows = [row for row in image_rows if row.analysis_group in ANALYSIS_GROUP_ORDER]
    affected_status_patient_rows = pool_by_patient(known_status_image_rows, "affected_status", ("affected_status",))
    analysis_group_patient_rows = pool_by_patient(
        known_analysis_group_image_rows,
        "analysis_group",
        ("analysis_group",),
    )
    analysis_group_view_patient_rows = pool_by_patient(
        known_analysis_group_image_rows,
        "analysis_group_view",
        ("analysis_group", "view"),
    )
    stage_patient_rows = pool_by_patient(image_rows, "waldenstrom_stage", ("waldenstrom_stage",), True)
    stage_view_patient_rows = pool_by_patient(
        image_rows,
        "waldenstrom_stage_view",
        ("waldenstrom_stage", "view"),
        True,
    )

    write_csv(output_dir / "image_mask_dice.csv", image_rows)
    write_csv(output_dir / "patient_pooled_dice_overall.csv", all_patient_rows)
    write_csv(output_dir / "patient_pooled_dice_by_view.csv", view_patient_rows)
    write_csv(output_dir / "patient_pooled_dice_by_affected_status.csv", affected_status_patient_rows)
    write_csv(output_dir / "patient_pooled_dice_by_analysis_group.csv", analysis_group_patient_rows)
    write_csv(output_dir / "patient_pooled_dice_by_analysis_group_view.csv", analysis_group_view_patient_rows)
    write_csv(output_dir / "patient_pooled_dice_by_waldenstrom_stage.csv", stage_patient_rows)
    write_csv(output_dir / "patient_pooled_dice_by_waldenstrom_stage_view.csv", stage_view_patient_rows)
    write_csv(output_dir / "analysis_group_view_counts.csv", analysis_group_counts(image_rows))
    write_csv(output_dir / "unknown_affected_status_exclusions.csv", unknown_affected_status_rows(image_rows))
    write_csv(output_dir / "missing_affected_waldenstrom_stage_exclusions.csv", missing_affected_stage_rows(image_rows))
    write_mask_colors(output_dir / "mask_colors.csv")

    overall_summaries = summarize_patient_rows(
        all_patient_rows,
        "overall",
        (),
        args.cutoffs,
        args.bootstrap_iterations,
        args.seed,
    )
    write_csv(
        output_dir / "overall_model_performance_by_mask.csv",
        overall_summaries,
    )
    overall_metric_summaries = summarize_overall_metrics(
        all_patient_rows,
        args.bootstrap_iterations,
        args.seed,
    )
    write_csv(output_dir / "overall_model_performance_all_metrics.csv", overall_metric_summaries)
    publication_rows = publication_overall_rows(overall_metric_summaries)
    write_publication_table_bundle(
        output_dir / "overall_model_performance_publication_table",
        publication_rows,
        "Overall Segmentation Performance",
        "Overall patient-pooled segmentation performance on the held-out test set.",
        "tab:overall-mask-performance",
    )
    view_summaries = summarize_patient_rows(
        view_patient_rows,
        "view",
        ("view",),
        args.cutoffs,
        args.bootstrap_iterations,
        args.seed,
    )
    write_csv(output_dir / "view_model_performance_by_mask.csv", view_summaries)
    view_publication_rows = publication_group_rows(
        view_summaries,
        "view",
        VIEW_ORDER,
        {"ap": "AP", "frog": "Frog-leg"},
    )
    write_publication_table_bundle(
        output_dir / "view_model_performance_publication_table",
        view_publication_rows,
        "Segmentation Performance by Radiograph View",
        "Patient-pooled Dice performance by radiograph view on the held-out test set.",
        "tab:view-mask-performance",
    )
    write_csv(
        output_dir / "affected_status_model_performance_by_mask.csv",
        summarize_patient_rows(
            affected_status_patient_rows,
            "affected_status",
            ("affected_status",),
            args.cutoffs,
            args.bootstrap_iterations,
            args.seed,
        ),
    )
    analysis_group_summaries = summarize_patient_rows(
        analysis_group_patient_rows,
        "analysis_group",
        ("analysis_group",),
        args.cutoffs,
        args.bootstrap_iterations,
        args.seed,
    )
    write_csv(output_dir / "analysis_group_model_performance_by_mask.csv", analysis_group_summaries)
    waldenstrom_publication_rows = publication_group_rows(
        analysis_group_summaries,
        "analysis_group",
        ANALYSIS_GROUP_ORDER,
        {group: (group if group == "Unaffected" else f"Stage {group}") for group in ANALYSIS_GROUP_ORDER},
    )
    write_publication_table_bundle(
        output_dir / "waldenstrom_stage_model_performance_publication_table",
        waldenstrom_publication_rows,
        "Segmentation Performance by Waldenstrom Stage",
        "Patient-pooled Dice performance for unaffected hips and affected hips by Waldenstrom stage.",
        "tab:waldenstrom-mask-performance",
    )
    write_csv(
        output_dir / "analysis_group_view_model_performance_by_mask.csv",
        summarize_patient_rows(
            analysis_group_view_patient_rows,
            "analysis_group_view",
            ("analysis_group", "view"),
            args.cutoffs,
            args.bootstrap_iterations,
            args.seed,
        ),
    )
    write_csv(
        output_dir / "waldenstrom_stage_model_performance_by_mask.csv",
        summarize_patient_rows(
            stage_patient_rows,
            "waldenstrom_stage",
            ("waldenstrom_stage",),
            args.cutoffs,
            args.bootstrap_iterations,
            args.seed,
        ),
    )
    write_csv(
        output_dir / "waldenstrom_stage_view_model_performance_by_mask.csv",
        summarize_patient_rows(
            stage_view_patient_rows,
            "waldenstrom_stage_view",
            ("waldenstrom_stage", "view"),
            args.cutoffs,
            args.bootstrap_iterations,
            args.seed,
        ),
    )

    plot_stage_view_boxplots(
        output_dir / "analysis_group_view_boxplots.png",
        analysis_group_view_patient_rows,
        args.seed,
    )
    for cutoff in sorted({float(cutoff) for cutoff in args.cutoffs}, reverse=True):
        plot_name = f"analysis_group_view_boxplots_cutoff_{cutoff:.2f}".replace(".", "_")
        plot_stage_view_boxplots(
            output_dir / f"{plot_name}.png",
            analysis_group_view_patient_rows,
            args.seed,
            cutoff=cutoff,
        )

    print(f"Wrote final mask analysis outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
