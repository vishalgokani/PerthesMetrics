"""Convert ordinary RGB radiographs into PerthesMetrics nnU-Net NIfTI channels."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perthesmetrics_io import IMAGE_EXTENSIONS, safe_case_id, write_rgb_nifti


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
    candidates = input_dir.rglob("*") if args.recursive else input_dir.glob("*")
    images = sorted(path for path in candidates if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No supported images found in {input_dir}")
    used: set[str] = set()
    manifest = []
    for source in images:
        case = safe_case_id(source)
        if case in used:
            raise ValueError(f"Duplicate case ID '{case}'. Rename source files to unique stems.")
        used.add(case)
        write_rgb_nifti(source, output_dir, case)
        manifest.append({"case_id": case, "source": source.relative_to(input_dir).as_posix()})
    with (output_dir.parent / "input_conversion_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case_id", "source"])
        writer.writeheader(); writer.writerows(manifest)
    print(f"Converted {len(images)} RGB radiographs into {len(images) * 3} NIfTI channels in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
