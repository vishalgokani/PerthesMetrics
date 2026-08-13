"""Convert ordinary RGB radiographs into PerthesMetrics nnU-Net NIfTI channels."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


def safe_case_id(path: Path) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._-")
    if not value:
        raise ValueError(f"Cannot create a case ID for {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert RGB images to three nnU-Net NIfTI channels.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="Destination imagesTs directory.")
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()
    input_dir, output_dir = args.input_dir.resolve(), args.output_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extensions = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}
    candidates = input_dir.rglob("*") if args.recursive else input_dir.glob("*")
    images = sorted(path for path in candidates if path.is_file() and path.suffix.lower() in extensions)
    if not images:
        raise FileNotFoundError(f"No supported images found in {input_dir}")
    used: set[str] = set()
    manifest = []
    affine = np.diag([999.0, 1.0, 1.0, 1.0])
    for source in images:
        case = safe_case_id(source)
        if case in used:
            raise ValueError(f"Duplicate case ID '{case}'. Rename source files to unique stems.")
        used.add(case)
        rgb = np.asarray(Image.open(source).convert("RGB"), dtype=np.uint8)
        for channel in range(3):
            # Match the historical model representation: a singleton through-plane
            # dimension with large spacing, while nnU-Net operates in 2D.
            data = rgb[:, :, channel][None, :, :]
            destination = output_dir / f"{case}_{channel:04d}.nii.gz"
            nib.save(nib.Nifti1Image(data, affine), destination)
        manifest.append({"case_id": case, "source": source.relative_to(input_dir).as_posix()})
    with (output_dir.parent / "input_conversion_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case_id", "source"])
        writer.writeheader(); writer.writerows(manifest)
    print(f"Converted {len(images)} RGB radiographs into {len(images) * 3} NIfTI channels in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
