from pathlib import Path

import pytest
from PIL import Image

from tools.cropper import crop_images, square_box_from_drag, validate_inputs
from tools.waldenstrom_figure_builder import (
    IMAGE_COUNT,
    build_svg,
    export_png_and_pdf,
    validate_images,
)


def make_image(path: Path, size: tuple[int, int] = (30, 20)) -> Path:
    Image.new("RGB", size, "white").save(path)
    return path


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


def test_builder_requires_28_square_images(tmp_path: Path) -> None:
    paths = [make_image(tmp_path / f"panel_{index}.png", (10, 10)) for index in range(IMAGE_COUNT)]
    validate_images(paths)
    svg = tmp_path / "figure.svg"
    build_svg(paths, svg)
    text = svg.read_text(encoding="utf-8")
    assert text.count("data:image/png;base64,") == IMAGE_COUNT * 2
    assert "Frog-leg lateral" in text

    with pytest.raises(ValueError, match="Exactly 28"):
        validate_images(paths[:-1])


def test_builder_exports_matching_png_and_pdf(tmp_path: Path) -> None:
    paths = [make_image(tmp_path / f"panel_{index}.png", (10, 10)) for index in range(IMAGE_COUNT)]
    prefix = tmp_path / "waldenstrom"
    build_svg(paths, prefix.with_suffix(".svg"))
    export_png_and_pdf(prefix.with_suffix(".svg"), prefix, dpi=72)
    assert Image.open(prefix.with_suffix(".png")).size == (510, 709)
    assert prefix.with_suffix(".pdf").read_bytes().startswith(b"%PDF")
