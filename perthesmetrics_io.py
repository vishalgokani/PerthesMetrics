"""Shared BMP/NIfTI conversion for PerthesMetrics training and inference."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
LABELS = {
    "background": 0,
    "acetabulum": 1,
    "gt": 2,
    "head": 3,
    "lt": 4,
    "neck": 5,
    "shaft": 6,
    "sourcil": 7,
    "triradiate cartilage": 8,
}
AFFINE = np.diag([999.0, 1.0, 1.0, 1.0])


def safe_case_id(path: Path) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._-")
    if not value:
        raise ValueError(f"Cannot create a case ID for {path}")
    return value


def discover_radiographs(data_dir: Path) -> list[Path]:
    data_dir = data_dir.resolve()
    images = sorted(
        path for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No supported radiographs found in {data_dir}")
    identifiers = [safe_case_id(path) for path in images]
    duplicates = sorted(value for value, count in Counter(identifiers).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate normalized case IDs, including: {duplicates[:3]}")
    return images


def write_rgb_nifti(source: Path, output_dir: Path, case_id: str) -> tuple[int, int]:
    rgb = np.asarray(Image.open(source).convert("RGB"), dtype=np.uint8)
    output_dir.mkdir(parents=True, exist_ok=True)
    for channel in range(3):
        data = rgb[:, :, channel][None, :, :]
        nib.save(nib.Nifti1Image(data, AFFINE), output_dir / f"{case_id}_{channel:04d}.nii.gz")
    return int(rgb.shape[1]), int(rgb.shape[0])


def write_training_label(
    source: Path, mask_root: Path, output_dir: Path, case_id: str, labels: dict[str, int] = LABELS
) -> int:
    expected_shape = np.asarray(Image.open(source)).shape[:2]
    combined = np.zeros(expected_shape, dtype=np.uint8)
    occupied = np.zeros(expected_shape, dtype=bool)
    overlap_pixels = 0
    # Higher label values take precedence in overlapping annotation pixels so
    # the eight binary masks produce one deterministic multiclass label map.
    for name, value in sorted(labels.items(), key=lambda item: item[1]):
        if value == 0:
            continue
        mask_path = mask_root / name / source.name
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing {name} mask for {source.name}: {mask_path}")
        selected = np.asarray(Image.open(mask_path)) > 0
        if selected.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for {mask_path}: radiograph {expected_shape}, mask {selected.shape}"
            )
        overlap_pixels += int(np.count_nonzero(occupied & selected))
        combined[selected] = value
        occupied |= selected
    output_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(combined[None, :, :], AFFINE), output_dir / f"{case_id}.nii.gz")
    return overlap_pixels


def stage_bmp_training_data(
    data_dir: Path, dataset_dir: Path, labels: dict[str, int] = LABELS
) -> list[str]:
    images = discover_radiographs(data_dir)
    images_tr, labels_tr = dataset_dir / "imagesTr", dataset_dir / "labelsTr"
    manifest: list[dict[str, object]] = []
    for source in images:
        case_id = safe_case_id(source)
        width, height = write_rgb_nifti(source, images_tr, case_id)
        overlap = write_training_label(source, data_dir / "masks", labels_tr, case_id, labels)
        manifest.append({
            "case_id": case_id, "source_filename": source.name,
            "width": width, "height": height, "overlap_pixels_resolved": overlap,
        })
    write_conversion_manifest(dataset_dir / "input_conversion_manifest.csv", manifest)
    return [str(row["case_id"]) for row in manifest]


def stage_bmp_inference_data(data_dir: Path, images_ts: Path) -> dict[str, str]:
    manifest: list[dict[str, object]] = []
    mapping: dict[str, str] = {}
    for source in discover_radiographs(data_dir):
        case_id = safe_case_id(source)
        width, height = write_rgb_nifti(source, images_ts, case_id)
        mapping[case_id] = source.name
        manifest.append({
            "case_id": case_id, "source_filename": source.name,
            "width": width, "height": height,
        })
    write_conversion_manifest(images_ts.parent / "input_conversion_manifest.csv", manifest)
    return mapping


def write_conversion_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def load_label_map(path: Path) -> np.ndarray:
    data = np.squeeze(np.asarray(nib.load(path).dataobj))
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D prediction in {path}; found shape {data.shape}")
    return data.astype(np.uint8)


def export_prediction_bmps(
    predictions_dir: Path,
    output_dir: Path,
    case_to_filename: dict[str, str],
    labels: dict[str, int] = LABELS,
) -> list[dict[str, str]]:
    predictions = sorted(predictions_dir.glob("*.nii.gz"))
    if len(predictions) != len(case_to_filename):
        raise RuntimeError(f"Expected {len(case_to_filename)} predictions, found {len(predictions)}")
    rows: list[dict[str, str]] = []
    for name, value in labels.items():
        if value:
            (output_dir / name).mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for prediction in predictions:
        case_id = prediction.name.removesuffix(".nii.gz")
        if case_id not in case_to_filename:
            raise ValueError(f"Unexpected prediction case: {case_id}")
        label_map = load_label_map(prediction)
        filename = case_to_filename[case_id]
        for name, value in labels.items():
            if value == 0:
                continue
            mask = np.where(label_map == value, 255, 0).astype(np.uint8)
            Image.fromarray(mask, mode="L").save(output_dir / name / filename)
        rows.append({"case_id": case_id, "source_filename": filename})
        seen.add(case_id)
    missing = sorted(set(case_to_filename) - seen)
    if missing:
        raise RuntimeError(f"Missing predictions, including: {missing[:3]}")
    return rows
