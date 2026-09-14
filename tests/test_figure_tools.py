from pathlib import Path

import pytest
from PIL import Image

from tools.cropper import crop_images, square_box_from_drag, validate_inputs
from tools.waldenstrom_figure_builder import (
    CANVAS_HEIGHT,
    IMAGE_COUNT,
    build_svg,
    discover_images,
    export_png_and_pdf,
    validate_images,
)


def make_image(path: Path, size: tuple[int, int] = (30, 20)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)
    return path


def make_figure_inputs(root: Path) -> list[Path]:
    for stage in ("1a", "1b", "2a", "2b", "3a", "3b", "4"):
        for view in ("ap", "frog"):
            for kind in ("original", "ground_truth_masks", "nnunet_masks"):
                make_image(root / stage / f"case_{view}_{kind}_cropped.png", (10, 10))
    return discover_images(root)


def test_square_box_stays_inside_image() -> None:
    box = square_box_from_drag(-10, 2, 29, 17, width=30, height=20)
    left, top, right, bottom = box
    assert right - left == bottom - top
    assert 0 <= left < right <= 30
    assert 0 <= top < bottom <= 20


def test_crop_images_applies_same_square_box(tmp_path: Path) -> None:
    inputs = [make_image(tmp_path / f"input_{index}.bmp") for index in range(2)]
    assert validate_inputs(inputs) == (30, 20)
    outputs = crop_images(inputs, tmp_path / "crops", (5, 2, 17, 14), "_crop")
    assert [Image.open(path).size for path in outputs] == [(12, 12), (12, 12)]


def test_builder_discovers_42_images_in_panel_order(tmp_path: Path) -> None:
    paths = make_figure_inputs(tmp_path)
    assert len(paths) == IMAGE_COUNT
    assert [path.parent.name for path in paths[:6]] == ["1a"] * 6
    assert [path.name for path in paths[:6]] == [
        "case_ap_original_cropped.png",
        "case_ap_ground_truth_masks_cropped.png",
        "case_ap_nnunet_masks_cropped.png",
        "case_frog_original_cropped.png",
        "case_frog_ground_truth_masks_cropped.png",
        "case_frog_nnunet_masks_cropped.png",
    ]
    validate_images(paths)


def test_builder_rejects_missing_panel(tmp_path: Path) -> None:
    make_figure_inputs(tmp_path)
    (tmp_path / "1a" / "case_ap_original_cropped.png").unlink()
    with pytest.raises(ValueError, match="Expected exactly one Ia AP original"):
        discover_images(tmp_path)


def test_builder_embeds_images_and_uses_compact_stage_labels(tmp_path: Path) -> None:
    paths = make_figure_inputs(tmp_path)
    svg = tmp_path / "figure.svg"
    build_svg(paths, svg)
    text = svg.read_text(encoding="utf-8")
    assert text.count("data:image/png;base64,") == IMAGE_COUNT * 2
    assert ">IIa</text>" in text
    assert ">IIIb</text>" in text


def test_builder_exports_png_and_pdf(tmp_path: Path) -> None:
    paths = make_figure_inputs(tmp_path / "inputs")
    prefix = tmp_path / "waldenstrom"
    build_svg(paths, prefix.with_suffix(".svg"))
    export_png_and_pdf(prefix.with_suffix(".svg"), prefix, dpi=72)
    assert Image.open(prefix.with_suffix(".png")).size == (
        510,
        round((CANVAS_HEIGHT / 10) / 25.4 * 72),
    )
    assert prefix.with_suffix(".pdf").read_bytes().startswith(b"%PDF")
