"""Build a publication-quality Waldenstrom staging figure.

The input directory must contain stage subdirectories named ``1a``, ``1b``,
``2a``, ``2b``, ``3a``, ``3b``, and ``4``. Each directory must contain one AP
and one frog-leg image for each panel type: original, ground-truth masks, and
nnU-Net masks. Files are discovered from their names, so individual image paths
do not need to be entered on the command line.
"""

from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path
from typing import Sequence
from xml.sax.saxutils import escape

from PIL import Image, ImageOps


STAGES = (
    ("1a", "Ia"), ("1b", "Ib"), ("2a", "IIa"), ("2b", "IIb"),
    ("3a", "IIIa"), ("3b", "IIIb"), ("4", "IV"),
)
COLUMNS = (
    ("ap", "original", "Original"),
    ("ap", "ground_truth_masks", "Ground truth"),
    ("ap", "nnunet_masks", "Mask"),
    ("frog", "original", "Original"),
    ("frog", "ground_truth_masks", "Ground truth"),
    ("frog", "nnunet_masks", "Mask"),
)
IMAGE_COUNT = len(STAGES) * len(COLUMNS)
SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
FONT = "Times New Roman, Times, serif"

CANVAS_WIDTH = 1800
CELL = 250
LEFT_MARGIN = 30
RIGHT_MARGIN = 30
ROW_LABEL_WIDTH = 150
COLUMN_GAP = 10
GROUP_GAP = 20
HEADER_RULE_Y = 158
FIRST_ROW_Y = 174
WITHIN_STAGE_GAP = 8
BETWEEN_STAGE_GAP = 22
BOTTOM_MARGIN = 20


def row_positions() -> tuple[int, ...]:
    positions: list[int] = []
    y = FIRST_ROW_Y
    for index in range(len(STAGES)):
        positions.append(y)
        y += CELL
        if index < len(STAGES) - 1:
            y += WITHIN_STAGE_GAP if index in (0, 2, 4) else BETWEEN_STAGE_GAP
    return tuple(positions)


def x_positions() -> tuple[int, ...]:
    first_x = LEFT_MARGIN + ROW_LABEL_WIDTH
    return tuple(
        first_x + index * (CELL + COLUMN_GAP) + (GROUP_GAP if index >= 3 else 0)
        for index in range(len(COLUMNS))
    )


CANVAS_HEIGHT = row_positions()[-1] + CELL + BOTTOM_MARGIN


def discover_images(input_dir: Path) -> list[Path]:
    """Discover panels in deterministic stage, view, and type order."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    ordered: list[Path] = []
    for directory_name, stage_label in STAGES:
        stage_dir = input_dir / directory_name
        if not stage_dir.is_dir():
            raise FileNotFoundError(
                f"Missing directory for Waldenstrom stage {stage_label}: {stage_dir}"
            )
        candidates = sorted(
            path for path in stage_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        for view, kind, _label in COLUMNS:
            matches = [
                path for path in candidates
                if f"_{view}_" in path.name.lower()
                and f"_{kind}_cropped" in path.stem.lower()
            ]
            if len(matches) != 1:
                names = ", ".join(path.name for path in matches) or "none"
                raise ValueError(
                    f"Expected exactly one {stage_label} {view.upper()} {kind} "
                    f"image in {stage_dir}; found {len(matches)}: {names}"
                )
            ordered.append(matches[0])
    return ordered


def validate_images(paths: Sequence[Path]) -> None:
    if len(paths) != IMAGE_COUNT:
        raise ValueError(f"Expected {IMAGE_COUNT} images; discovered {len(paths)}.")
    for path in paths:
        with Image.open(path) as image:
            oriented = ImageOps.exif_transpose(image)
            if oriented.width != oriented.height:
                raise ValueError(
                    f"Input is not square ({oriented.width} x {oriented.height}): {path}. "
                    "Prepare it with tools/cropper.py first."
                )


def png_data_uri(path: Path) -> str:
    """Convert any Pillow-supported image to an embedded PNG data URI."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        if image.mode not in {"L", "LA", "RGB", "RGBA"}:
            image = image.convert("RGBA" if "transparency" in image.info else "RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def svg_text(
    x: float, y: float, value: str, size: int, *,
    anchor: str = "middle", weight: str = "normal",
) -> str:
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
        f'fill="#111">{escape(value)}</text>'
    )


def placeholder_panel(x: int, y: int, label: str) -> list[str]:
    return [
        f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
        'fill="#eeeeee" stroke="#111" stroke-width="2"/>',
        f'<line x1="{x + 18}" y1="{y + CELL - 18}" x2="{x + CELL - 18}" '
        f'y2="{y + 18}" stroke="#c5c5c5" stroke-width="3"/>',
        svg_text(x + CELL / 2, y + CELL / 2 + 10, label.upper(), 24, weight="bold"),
    ]


def build_svg(image_paths: Sequence[Path] | None, output_svg: Path) -> None:
    """Write a self-contained SVG with embedded PNG panels."""
    xs = x_positions()
    ys = row_positions()
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{CANVAS_WIDTH / 10:g}mm" height="{CANVAS_HEIGHT / 10:g}mm" '
        f'viewBox="0 0 {CANVAS_WIDTH} {CANVAS_HEIGHT}">',
        f'<rect width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}" fill="white"/>',
        svg_text(LEFT_MARGIN + ROW_LABEL_WIDTH / 2, 91, "Waldenström", 30, weight="bold"),
        svg_text(LEFT_MARGIN + ROW_LABEL_WIDTH / 2, 128, "stage", 30, weight="bold"),
        svg_text((xs[0] + xs[2] + CELL) / 2, 49, "AP", 50, weight="bold"),
        svg_text((xs[3] + xs[5] + CELL) / 2, 49, "Frog-leg lateral", 50, weight="bold"),
    ]
    for x, (_view, _kind, label) in zip(xs, COLUMNS):
        parts.append(svg_text(x + CELL / 2, 117, label, 35, weight="bold"))
    parts.append(
        f'<line x1="{LEFT_MARGIN}" y1="{HEADER_RULE_Y}" '
        f'x2="{CANVAS_WIDTH - RIGHT_MARGIN}" y2="{HEADER_RULE_Y}" '
        'stroke="#222" stroke-width="2"/>'
    )
    for row_index, ((_directory, stage_label), row_y) in enumerate(zip(STAGES, ys)):
        parts.append(svg_text(
            LEFT_MARGIN + ROW_LABEL_WIDTH - 20, row_y + CELL / 2 + 14,
            stage_label, 44, anchor="end", weight="bold",
        ))
        for column_index, (x, (_view, _kind, label)) in enumerate(zip(xs, COLUMNS)):
            if image_paths is None:
                parts.extend(placeholder_panel(x, row_y, label))
            else:
                uri = png_data_uri(image_paths[row_index * len(COLUMNS) + column_index])
                parts.append(
                    f'<image x="{x}" y="{row_y}" width="{CELL}" height="{CELL}" '
                    f'preserveAspectRatio="xMidYMid meet" href="{uri}" xlink:href="{uri}"/>'
                )
                parts.append(
                    f'<rect x="{x}" y="{row_y}" width="{CELL}" height="{CELL}" '
                    'fill="none" stroke="#111" stroke-width="2"/>'
                )
        if row_index in (1, 3, 5):
            separator_y = row_y + CELL + BETWEEN_STAGE_GAP / 2
            parts.append(
                f'<line x1="{LEFT_MARGIN}" y1="{separator_y}" '
                f'x2="{CANVAS_WIDTH - RIGHT_MARGIN}" y2="{separator_y}" '
                'stroke="#a9a9a9" stroke-width="1.5"/>'
            )
    parts.append("</svg>")
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    output_svg.write_text("\n".join(parts) + "\n", encoding="utf-8")


def export_png_and_pdf(svg_path: Path, output_prefix: Path, dpi: int) -> None:
    try:
        import cairosvg
    except ImportError as error:
        raise RuntimeError(
            "PNG/PDF export requires CairoSVG. Install it with: pip install CairoSVG"
        ) from error
    cairosvg.svg2png(
        url=str(svg_path), write_to=str(output_prefix.with_suffix(".png")),
        output_width=round((CANVAS_WIDTH / 10) / 25.4 * dpi),
        output_height=round((CANVAS_HEIGHT / 10) / 25.4 * dpi),
    )
    cairosvg.svg2pdf(url=str(svg_path), write_to=str(output_prefix.with_suffix(".pdf")))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Discover prepared panels by filename and build SVG, PDF, and PNG "
            "versions of the full Waldenstrom staging figure."
        )
    )
    parser.add_argument(
        "--input-dir", type=Path,
        help="Directory containing stage folders 1a, 1b, 2a, 2b, 3a, 3b, and 4",
    )
    parser.add_argument(
        "--output-prefix", required=True, type=Path,
        help="Output path without an extension (writes .svg, .pdf, and .png)",
    )
    parser.add_argument(
        "--dpi", type=int, default=600,
        help="PNG resolution in dots per inch (default: 600)",
    )
    parser.add_argument(
        "--layout-preview", action="store_true",
        help="Build a labeled placeholder figure without an input directory",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.dpi < 72:
        raise ValueError("--dpi must be at least 72.")
    if args.layout_preview:
        if args.input_dir is not None:
            raise ValueError("Do not provide --input-dir with --layout-preview.")
        paths: Sequence[Path] | None = None
    else:
        if args.input_dir is None:
            raise ValueError("--input-dir is required unless --layout-preview is used.")
        paths = discover_images(args.input_dir.resolve())
        validate_images(paths)
    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    svg_path = prefix.with_suffix(".svg")
    build_svg(paths, svg_path)
    export_png_and_pdf(svg_path, prefix, args.dpi)
    print(f"SVG: {svg_path}")
    print(f"PDF: {prefix.with_suffix('.pdf')}")
    print(f"PNG ({args.dpi} DPI): {prefix.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
