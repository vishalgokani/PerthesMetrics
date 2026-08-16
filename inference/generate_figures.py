"""Create adjustable color overlays from radiographs and nnU-Net label maps.

This is intentionally separate from inference. Edit OPACITY and COLORS below, or
override opacity on the command line, then rerun without repeating inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


OPACITY = 0.45
COLORS = {
    1: (230, 25, 75), 2: (60, 180, 75), 3: (0, 130, 200), 4: (245, 130, 48),
    5: (145, 30, 180), 6: (70, 240, 240), 7: (240, 50, 230), 8: (255, 225, 25),
}
IMAGE_EXTENSIONS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


def load_mask(path: Path) -> np.ndarray:
    mask = np.asarray(nib.load(path).dataobj)
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D label map in {path}; found shape {mask.shape}")
    return mask.astype(np.uint8)


def source_index(source_dir: Path) -> dict[str, Path]:
    images = [p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    index: dict[str, Path] = {}
    for path in images:
        if path.stem in index:
            raise ValueError(f"Duplicate source image stem: {path.stem}")
        index[path.stem] = path
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description="Render radiograph/segmentation overlays.")
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--predictions-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--opacity", type=float, default=OPACITY)
    args = parser.parse_args()
    if not 0 <= args.opacity <= 1:
        raise ValueError("--opacity must be between 0 and 1")
    sources = source_index(args.source_dir.resolve())
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    predictions = sorted(args.predictions_dir.resolve().glob("*.nii.gz"))
    if not predictions:
        raise FileNotFoundError(f"No .nii.gz predictions in {args.predictions_dir}")
    missing = []
    for prediction in predictions:
        case = prediction.name.removesuffix(".nii.gz")
        source = sources.get(case)
        if source is None:
            missing.append(case); continue
        image = np.asarray(Image.open(source).convert("RGB"), dtype=np.uint8)
        mask = load_mask(prediction)
        if mask.shape != image.shape[:2]:
            if mask.T.shape == image.shape[:2]:
                mask = mask.T
            else:
                raise ValueError(f"Shape mismatch for {case}: image {image.shape[:2]}, mask {mask.shape}")
        overlay = image.astype(np.float32)
        for label, color in COLORS.items():
            selected = mask == label
            overlay[selected] = (1 - args.opacity) * overlay[selected] + args.opacity * np.asarray(color)
        Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(output / f"{case}_overlay.png")
    if missing:
        raise FileNotFoundError(f"No source radiograph for {len(missing)} predictions; examples: {missing[:5]}")
    print(f"Wrote {len(predictions)} overlays to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
