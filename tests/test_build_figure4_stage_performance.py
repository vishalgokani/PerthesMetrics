import csv
from pathlib import Path

import pytest
from PIL import Image

from tools.build_figure4_stage_performance import (
    GROUPS,
    MASKS,
    build_figure,
    export_figure,
    load_estimates,
)


FIELDS = [
    "scope",
    "analysis_group",
    "mask",
    "mean_patient_pooled_dice",
    "ci95_low",
    "ci95_high",
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def make_inputs(root: Path) -> None:
    overall_rows = []
    group_rows = []
    for mask_index, mask in enumerate(MASKS):
        mean = 0.72 + mask_index * 0.03
        overall_rows.append(
            {
                "scope": "overall",
                "analysis_group": "",
                "mask": mask,
                "mean_patient_pooled_dice": mean,
                "ci95_low": mean - 0.02,
                "ci95_high": min(mean + 0.02, 1.0),
            }
        )
        for group_index, group in enumerate(GROUPS[1:]):
            group_mean = min(mean + group_index * 0.005, 0.97)
            group_rows.append(
                {
                    "scope": "analysis_group",
                    "analysis_group": group,
                    "mask": mask,
                    "mean_patient_pooled_dice": group_mean,
                    "ci95_low": group_mean - 0.02,
                    "ci95_high": min(group_mean + 0.02, 1.0),
                }
            )
    write_csv(root / "overall_model_performance_by_mask.csv", overall_rows)
    write_csv(root / "analysis_group_model_performance_by_mask.csv", group_rows)


def test_load_estimates_requires_every_group_and_mask(tmp_path: Path) -> None:
    make_inputs(tmp_path)
    estimates = load_estimates(tmp_path)
    assert len(estimates) == len(GROUPS) * len(MASKS)
    assert estimates[("Overall", "acetabulum")].mean == pytest.approx(0.72)
    assert estimates[("IV", "gt")].ci_high <= 1.0


def test_load_estimates_reports_missing_stratum(tmp_path: Path) -> None:
    make_inputs(tmp_path)
    path = tmp_path / "analysis_group_model_performance_by_mask.csv"
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    write_csv(
        path,
        [row for row in rows if not (row["analysis_group"] == "IV" and row["mask"] == "gt")],
    )
    with pytest.raises(ValueError, match="IV/gt"):
        load_estimates(tmp_path)


def test_build_and_export_figure4(tmp_path: Path) -> None:
    analysis_dir = tmp_path / "analysis"
    output_dir = tmp_path / "figures"
    make_inputs(analysis_dir)
    figure = build_figure(load_estimates(analysis_dir))
    pdf_path, svg_path, png_path = export_figure(figure, output_dir, dpi=300)

    assert pdf_path.read_bytes().startswith(b"%PDF")
    svg_text = svg_path.read_text(encoding="utf-8")
    assert "a)  Acetabulum" in svg_text
    assert "g)  Greater Trochanter" in svg_text
    assert "Analysis stratum" in svg_text
    with Image.open(png_path) as image:
        assert image.width > image.height > 1000

