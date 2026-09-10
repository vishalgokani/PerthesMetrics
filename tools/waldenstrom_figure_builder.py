"""Build a publication-quality 28-panel Waldenstrom figure.

Supply images from left to right and then top to bottom. Each of the seven
rows contains: AP ground truth, AP mask, frog-leg ground truth, frog-leg mask.
Inputs must already be square; use ``tools/cropper.py`` to prepare them.
"""

from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path
from typing import Sequence
from xml.sax.saxutils import escape

from PIL import Image, ImageOps


STAGES = ("I-A", "I-B", "II-A", "II-B", "III-A", "III-B", "IV")
COLUMNS = ("AP ground truth", "AP mask", "Frog-leg ground truth", "Frog-leg mask")
IMAGE_COUNT = len(STAGES) * len(COLUMNS)
FONT = "Times New Roman, Times, serif"


def validate_images(paths: Sequence[Path]) -> None:
    if len(paths) != IMAGE_COUNT:
        raise ValueError(
            f"Exactly {IMAGE_COUNT} images are required; received {len(paths)}. "
            "Order each row as AP ground truth, AP mask, frog-leg ground truth, "
            "frog-leg mask, for stages I-A, I-B, II-A, II-B, III-A, III-B, IV."
        )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Input image not found: {path}")
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
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/png;base64," + encoded


def svg_text(
    x: float,
    y: float,
    value: str,
    size: int,
    *,
    anchor: str = "middle",
    weight: str = "normal",
) -> str:
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
        f'fill="#111">{escape(value)}</text>'
    )


def build_svg(image_paths: Sequence[Path], output_svg: Path) -> None:
    """Write a self-contained SVG with every input converted to embedded PNG."""
    width, height = 1800, 2500
    left_margin = 80
    row_label_width = 235
    column_gap = 34
    group_gap = 78
    cell = 285
    first_x = left_margin + row_label_width
    x_positions = (
        first_x,
        first_x + cell + column_gap,
        first_x + 2 * (cell + column_gap) + group_gap,
        first_x + 3 * (cell + column_gap) + group_gap,
    )

    row_positions: list[int] = []
    y = 235
    for index in range(len(STAGES)):
        row_positions.append(y)
        y += cell
        if index < len(STAGES) - 1:
            y += 22 if index in (0, 2, 4) else 54

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="180mm" height="250mm" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        svg_text(left_margin + row_label_width / 2, 162, "Waldenström stage", 34),
        svg_text((x_positions[0] + x_positions[1] + cell) / 2, 92, "AP", 45, weight="bold"),
        svg_text(
            (x_positions[2] + x_positions[3] + cell) / 2,
            92,
            "Frog-leg lateral",
            45,
            weight="bold",
        ),
        svg_text(x_positions[0] + cell / 2, 162, "Ground truth", 31),
        svg_text(x_positions[1] + cell / 2, 162, "Mask", 31),
        svg_text(x_positions[2] + cell / 2, 162, "Ground truth", 31),
        svg_text(x_positions[3] + cell / 2, 162, "Mask", 31),
        f'<line x1="{left_margin}" y1="200" x2="{width-left_margin}" y2="200" '
        'stroke="#222" stroke-width="2"/>',
    ]

    for row_index, (stage, row_y) in enumerate(zip(STAGES, row_positions)):
        parts.append(
            svg_text(
                left_margin + row_label_width - 28,
                row_y + cell / 2 + 12,
                stage,
                40,
                anchor="end",
            )
        )
        for column_index, x in enumerate(x_positions):
            path = image_paths[row_index * len(COLUMNS) + column_index]
            uri = png_data_uri(path)
            parts.append(
                f'<image x="{x}" y="{row_y}" width="{cell}" height="{cell}" '
                f'preserveAspectRatio="xMidYMid meet" href="{uri}" xlink:href="{uri}"/>'
            )
            parts.append(
                f'<rect x="{x}" y="{row_y}" width="{cell}" height="{cell}" '
                'fill="none" stroke="#111" stroke-width="2"/>'
            )
        if row_index in (1, 3, 5):
            separator_y = row_y + cell + 27
            parts.append(
                f'<line x1="{left_margin}" y1="{separator_y}" '
                f'x2="{width-left_margin}" y2="{separator_y}" '
                'stroke="#b5b5b5" stroke-width="1.5"/>'
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

    output_width = round(180 / 25.4 * dpi)
    output_height = round(250 / 25.4 * dpi)
    cairosvg.svg2png(
        url=str(svg_path),
        write_to=str(output_prefix.with_suffix(".png")),
        output_width=output_width,
        output_height=output_height,
    )
    cairosvg.svg2pdf(
        url=str(svg_path), write_to=str(output_prefix.with_suffix(".pdf"))
    )


def build_parser() -> argparse.ArgumentParser:
    order = " | ".join(
        f"{stage}: " + ", ".join(COLUMNS) for stage in STAGES
    )
    parser = argparse.ArgumentParser(
        description="Build SVG, PDF, and PNG versions of a 28-panel Waldenstrom figure.",
        epilog=f"Required left-to-right, top-to-bottom order: {order}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="Exactly 28 already-cropped square images in the order shown below",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        type=Path,
        help="Output path without an extension (writes .svg, .pdf, and .png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG resolution in dots per inch (default: 600)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.dpi < 72:
        raise ValueError("--dpi must be at least 72.")

    paths = [path.resolve() for path in args.images]
    validate_images(paths)
    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    svg_path = output_prefix.with_suffix(".svg")
    build_svg(paths, svg_path)
    export_png_and_pdf(svg_path, output_prefix, args.dpi)

    print(f"SVG: {svg_path}")
    print(f"PDF: {output_prefix.with_suffix('.pdf')}")
    print(f"PNG ({args.dpi} DPI): {output_prefix.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
