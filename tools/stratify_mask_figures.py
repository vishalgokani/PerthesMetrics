"""Organize generated mask overlay figures by view and Waldenstrom class."""

from __future__ import annotations

import argparse
from pathlib import Path

from final_mask_analysis import (
    organize_mask_figures_by_waldenstrom,
    read_stage_maps,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stratify mask figure outputs by AP/frog and Waldenstrom class.")
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Test data directory containing ap_classes.csv and frog_classes.csv.",
    )
    parser.add_argument(
        "--mask-figures-dir",
        type=Path,
        help="Directory containing generated mask figures. Defaults to DATA_DIR/final_mask_analysis_outputs/mask_figures.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory. Defaults to DATA_DIR/final_mask_analysis_outputs/mask_figures_stratified.",
    )
    parser.add_argument("--ap-classes-csv", type=Path, help="Defaults to DATA_DIR/ap_classes.csv.")
    parser.add_argument("--frog-classes-csv", type=Path, help="Defaults to DATA_DIR/frog_classes.csv.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data_dir = args.data_dir.resolve()
    mask_figures_dir = (args.mask_figures_dir or data_dir / "final_mask_analysis_outputs" / "mask_figures").resolve()
    output_dir = (args.output_dir or data_dir / "final_mask_analysis_outputs" / "mask_figures_stratified").resolve()
    ap_csv = (args.ap_classes_csv or data_dir / "ap_classes.csv").resolve()
    frog_csv = (args.frog_classes_csv or data_dir / "frog_classes.csv").resolve()
    stage_map = read_stage_maps(ap_csv, frog_csv)
    organize_mask_figures_by_waldenstrom(mask_figures_dir, stage_map, output_dir)
    print(f"Wrote stratified mask figures to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
