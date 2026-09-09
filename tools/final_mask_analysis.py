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


def publication_overall_rows(summaries: list[dict[str, object]], cutoffs: list[float]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in sorted(summaries, key=lambda item: mask_sort_key(item.get("mask"))):
        formatted: dict[str, object] = {
            "Mask": row["mask_display"],
            "Patients": row["n_patients"],
            "Images": row["n_images"],
            "Patient-pooled Dice, mean (95% CI)": (
                f"{format_decimal(row['mean_patient_pooled_dice'])} "
                f"({format_decimal(row['ci95_low'])}-{format_decimal(row['ci95_high'])})"
            ),
            "Median Dice": format_decimal(row["median_patient_pooled_dice"]),
        }
        for cutoff in cutoffs:
            suffix = cutoff_suffix(cutoff)
            formatted[f"Patients with Dice >= {cutoff:.2f}, % (95% CI)"] = (
                f"{format_percent(row.get(f'pass_rate_ge_{suffix}'))} "
                f"({format_percent(row.get(f'pass_rate_ge_{suffix}_ci95_low'))}-"
                f"{format_percent(row.get(f'pass_rate_ge_{suffix}_ci95_high'))})"
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


def plot_stage_view_boxplots(
    path: Path,
    rows: list[PatientMaskDice],
    cutoff: float,
    seed: int,
) -> None:
    filtered = [row for row in rows if row.analysis_group in ANALYSIS_GROUP_ORDER and row.view in VIEW_ORDER]
    if not filtered:
        return
    rng = np.random.default_rng(seed)
    figure, axes = plt.subplots(
        len(ANALYSIS_GROUP_ORDER),
        len(VIEW_ORDER),
        figsize=(15.5, 2.45 * len(ANALYSIS_GROUP_ORDER)),
        sharex=True,
        sharey=True,
    )
    figure.subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.16, hspace=0.22, wspace=0.08)
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in filtered:
        if math.isfinite(row.dice):
            grouped[(row.analysis_group, row.view, row.mask)].append(row.dice)

    for row_index, analysis_group in enumerate(ANALYSIS_GROUP_ORDER):
        for col_index, view in enumerate(VIEW_ORDER):
            axis = axes[row_index, col_index]
            values_by_mask = [grouped.get((analysis_group, view, mask_name), []) for mask_name in ANALYSIS_MASK_ORDER]
            positions = np.arange(1, len(ANALYSIS_MASK_ORDER) + 1)
            for position, mask_name, values in zip(positions, ANALYSIS_MASK_ORDER, values_by_mask):
                if not values:
                    continue
                jitter = rng.uniform(-0.16, 0.16, size=len(values))
                axis.scatter(
                    np.full(len(values), position) + jitter,
                    values,
                    s=17,
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
            axis.axhline(cutoff, color="black", linestyle="--", linewidth=0.9)
            axis.set_ylim(-0.02, 1.02)
            axis.set_xlim(0.4, len(ANALYSIS_MASK_ORDER) + 0.6)
            axis.grid(axis="y", color="0.88", linewidth=0.6)
            if row_index == 0:
                axis.set_title(view.upper())
            if col_index == 0:
                axis.set_ylabel(f"{analysis_group}\nDice")
            if row_index == len(ANALYSIS_GROUP_ORDER) - 1:
                axis.set_xticks(positions)
                axis.set_xticklabels([CLASS_DISPLAY[name] for name in ANALYSIS_MASK_ORDER], rotation=35, ha="right")
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
            markersize=6,
            label=CLASS_DISPLAY[name],
        )
        for name in ANALYSIS_MASK_ORDER
    ]
    figure.suptitle("Patient-pooled Dice by affected status, Waldenstrom stage, and radiograph view", y=1.01)
    figure.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.01))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
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


def default_output_dir(data_dir: Path) -> Path:
    return data_dir / "final_mask_analysis_outputs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Patient-bootstrapped final mask analysis.")
    parser.add_argument("--data-dir", required=True, type=Path, help="Directory containing root radiographs.")
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
    parser.add_argument("--cutoffs", nargs="+", type=float, default=[0.90, 0.80])
    parser.add_argument("--plot-cutoff", type=float, default=0.90)
    parser.add_argument("--patient-regex", default=r"Patient[_ -]*(?P<patient_id>\d+)")
    parser.add_argument("--make-mask-figures", action="store_true", help="Also write original/GT/nnU-Net overlay BMPs.")
    parser.add_argument(
        "--allow-resize-for-metrics",
        action="store_true",
        help="Resize prediction masks to GT shape with nearest-neighbor before computing metrics.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
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
    publication_rows = publication_overall_rows(overall_summaries, args.cutoffs)
    write_csv(output_dir / "overall_model_performance_publication_table.csv", publication_rows)
    write_markdown_table(output_dir / "overall_model_performance_publication_table.md", publication_rows)
    write_latex_table(
        output_dir / "overall_model_performance_publication_table.tex",
        publication_rows,
        "Overall patient-pooled segmentation performance on the held-out test set.",
        "tab:overall-mask-performance",
    )
    write_csv(
        output_dir / "view_model_performance_by_mask.csv",
        summarize_patient_rows(
            view_patient_rows,
            "view",
            ("view",),
            args.cutoffs,
            args.bootstrap_iterations,
            args.seed,
        ),
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
    write_csv(
        output_dir / "analysis_group_model_performance_by_mask.csv",
        summarize_patient_rows(
            analysis_group_patient_rows,
            "analysis_group",
            ("analysis_group",),
            args.cutoffs,
            args.bootstrap_iterations,
            args.seed,
        ),
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

    plot_cutoffs = sorted({float(cutoff) for cutoff in [*args.cutoffs, args.plot_cutoff]}, reverse=True)
    for cutoff in plot_cutoffs:
        plot_name = f"analysis_group_view_boxplots_cutoff_{cutoff:.2f}".replace(".", "_")
        plot_stage_view_boxplots(output_dir / f"{plot_name}.png", analysis_group_view_patient_rows, cutoff, args.seed)
    if args.make_mask_figures:
        create_mask_figures(data_dir, gt_dir, pred_dir, output_dir / "mask_figures", stage_map)

    print(f"Wrote final mask analysis outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
