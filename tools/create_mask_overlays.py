"""Create optional ground-truth and nnU-Net mask overlay figures."""

from __future__ import annotations

import argparse
from pathlib import Path

from final_mask_analysis import create_mask_figures, read_stage_maps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create mask overlays separately from the quantitative final-mask analysis."
    )
    parser.add_argument("--data-dir", required=True, type=Path, help="Directory containing root radiographs.")
    parser.add_argument("--ground-truth-dir", type=Path, help="Defaults to DATA_DIR/masks.")
    parser.add_argument("--prediction-dir", type=Path, help="Defaults to DATA_DIR/nnunet_masks.")
    parser.add_argument("--ap-classes-csv", type=Path, help="Defaults to DATA_DIR/ap_classes.csv.")
    parser.add_argument("--frog-classes-csv", type=Path, help="Defaults to DATA_DIR/frog_classes.csv.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to DATA_DIR/final_mask_analysis_outputs/mask_figures.",
    )
    parser.add_argument(
        "--stratify",
        action="store_true",
        help="Also copy overlays into mask_figures_stratified by view and Waldenstrom stage.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data_dir = args.data_dir.resolve()
    gt_dir = (args.ground_truth_dir or data_dir / "masks").resolve()
    pred_dir = (args.prediction_dir or data_dir / "nnunet_masks").resolve()
    output_dir = (
        args.output_dir or data_dir / "final_mask_analysis_outputs" / "mask_figures"
    ).resolve()
    stage_map = None
    if args.stratify:
        ap_csv = (args.ap_classes_csv or data_dir / "ap_classes.csv").resolve()
        frog_csv = (args.frog_classes_csv or data_dir / "frog_classes.csv").resolve()
        stage_map = read_stage_maps(ap_csv, frog_csv)
    create_mask_figures(data_dir, gt_dir, pred_dir, output_dir, stage_map)
    print(f"Wrote mask overlay figures to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
