"""Build full and simplified publication-quality Waldenstrom figures.

Supply 42 square images from left to right and then top to bottom. Each stage
row contains AP original, AP ground truth, AP mask, frog-leg original,
frog-leg ground truth, and frog-leg mask. One run creates both the complete
six-column figure and a four-column version containing only originals/masks.

Use ``--layout-preview`` without image paths to create labeled placeholder
figures for reviewing the layout. Use ``tools/cropper.py`` to prepare images.
"""

from __future__ import annotations

import argparse
import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from xml.sax.saxutils import escape

from PIL import Image, ImageOps


STAGES = ("I-A", "I-B", "II-A", "II-B", "III-A", "III-B", "IV")
KINDS = ("Original", "Ground truth", "Mask")
INPUT_COLUMNS = (
    "AP original",
    "AP ground truth",
    "AP mask",
    "Frog-leg original",
    "Frog-leg ground truth",
    "Frog-leg mask",
)
IMAGE_COUNT = len(STAGES) * len(INPUT_COLUMNS)
FONT = "Times New Roman, Times, serif"


@dataclass(frozen=True)
class Layout:
    suffix: str
    column_indices: tuple[int, ...]
    subheaders: tuple[str, ...]
    width: int
    description: str


FULL_LAYOUT = Layout(
    suffix="full",
    column_indices=(0, 1, 2, 3, 4, 5),
    subheaders=("Original", "Ground truth", "Mask") * 2,
    width=1800,
    description="Original, ground truth, and mask",
)
ORIGINAL_MASK_LAYOUT = Layout(
    suffix="original_masks",
    column_indices=(0, 2, 3, 5),
    subheaders=("Original", "Mask") * 2,
    width=1280,
    description="Original and mask",
)
LAYOUTS = (FULL_LAYOUT, ORIGINAL_MASK_LAYOUT)

# Geometry uses a common scale in both figures. The compact figure is narrower
# rather than stretching its four panels, so panel size and typography remain
# identical between manuscript versions.
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


CANVAS_HEIGHT = row_positions()[-1] + CELL + BOTTOM_MARGIN


def x_positions(layout: Layout) -> tuple[int, ...]:
    first_x = LEFT_MARGIN + ROW_LABEL_WIDTH
    per_group = len(layout.column_indices) // 2
    values: list[int] = []
    for index in range(len(layout.column_indices)):
        extra_group_gap = GROUP_GAP if index >= per_group else 0
        values.append(first_x + index * (CELL + COLUMN_GAP) + extra_group_gap)
    return tuple(values)


def validate_images(paths: Sequence[Path]) -> None:
    if len(paths) != IMAGE_COUNT:
        raise ValueError(
            f"Exactly {IMAGE_COUNT} images are required; received {len(paths)}. "
            "For each stage, order them as AP original, AP ground truth, AP mask, "
            "frog-leg original, frog-leg ground truth, frog-leg mask."
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


def png_data_uri(path: Path, *, flip_horizontal: bool = False) -> str:
    """Convert any Pillow-supported image to an embedded PNG data URI."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        if flip_horizontal:
            image = ImageOps.mirror(image)
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


def placeholder_panel(x: int, y: int, label: str) -> list[str]:
    return [
        f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
        'fill="#eeeeee" stroke="#111" stroke-width="2"/>',
        f'<line x1="{x + 18}" y1="{y + CELL - 18}" x2="{x + CELL - 18}" '
        f'y2="{y + 18}" stroke="#c5c5c5" stroke-width="3"/>',
        svg_text(x + CELL / 2, y + CELL / 2 + 10, label.upper(), 24, weight="bold"),
    ]


def build_svg(
    image_paths: Sequence[Path] | None,
    output_svg: Path,
    layout: Layout = FULL_LAYOUT,
    flip_stages: frozenset[str] = frozenset(),
) -> None:
    """Write one self-contained figure SVG, optionally using placeholders."""
    xs = x_positions(layout)
    ys = row_positions()
    physical_width_mm = layout.width / 10
    physical_height_mm = CANVAS_HEIGHT / 10
    columns_per_view = len(layout.column_indices) // 2

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{physical_width_mm:g}mm" height="{physical_height_mm:g}mm" '
        f'viewBox="0 0 {layout.width} {CANVAS_HEIGHT}">',
        f'<rect width="{layout.width}" height="{CANVAS_HEIGHT}" fill="white"/>',
        svg_text(LEFT_MARGIN + ROW_LABEL_WIDTH / 2, 91, "Waldenström", 30, weight="bold"),
        svg_text(LEFT_MARGIN + ROW_LABEL_WIDTH / 2, 128, "stage", 30, weight="bold"),
        svg_text((xs[0] + xs[columns_per_view - 1] + CELL) / 2, 49, "AP", 50, weight="bold"),
        svg_text((xs[columns_per_view] + xs[-1] + CELL) / 2, 49, "Frog-leg lateral", 50, weight="bold"),
    ]

    for x, subheader in zip(xs, layout.subheaders):
        parts.append(svg_text(x + CELL / 2, 117, subheader, 35, weight="bold"))

    parts.append(
        f'<line x1="{LEFT_MARGIN}" y1="{HEADER_RULE_Y}" '
        f'x2="{layout.width - RIGHT_MARGIN}" y2="{HEADER_RULE_Y}" '
        'stroke="#222" stroke-width="2"/>'
    )

    for row_index, (stage, row_y) in enumerate(zip(STAGES, ys)):
        parts.append(
            svg_text(
                LEFT_MARGIN + ROW_LABEL_WIDTH - 20,
                row_y + CELL / 2 + 14,
                stage,
                44,
                anchor="end",
                weight="bold",
            )
        )
        for output_column, (input_column, x) in enumerate(zip(layout.column_indices, xs)):
            if image_paths is None:
                parts.extend(placeholder_panel(x, row_y, layout.subheaders[output_column]))
            else:
                path = image_paths[row_index * len(INPUT_COLUMNS) + input_column]
                uri = png_data_uri(path, flip_horizontal=stage in flip_stages)
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
                f'x2="{layout.width - RIGHT_MARGIN}" y2="{separator_y}" '
                'stroke="#a9a9a9" stroke-width="1.5"/>'
            )

    parts.append("</svg>")
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    output_svg.write_text("\n".join(parts) + "\n", encoding="utf-8")


def export_png_and_pdf(
    svg_path: Path,
    output_prefix: Path,
    dpi: int,
    layout: Layout = FULL_LAYOUT,
) -> None:
    try:
        import cairosvg
    except ImportError as error:
        raise RuntimeError(
            "PNG/PDF export requires CairoSVG. Install it with: pip install CairoSVG"
        ) from error

    width_mm = layout.width / 10
    height_mm = CANVAS_HEIGHT / 10
    output_width = round(width_mm / 25.4 * dpi)
    output_height = round(height_mm / 25.4 * dpi)
    cairosvg.svg2png(
        url=str(svg_path),
        write_to=str(output_prefix.with_suffix(".png")),
        output_width=output_width,
        output_height=output_height,
    )
    cairosvg.svg2pdf(
        url=str(svg_path), write_to=str(output_prefix.with_suffix(".pdf"))
    )


def output_path(prefix: Path, layout: Layout, extension: str) -> Path:
    return prefix.parent / f"{prefix.name}_{layout.suffix}.{extension}"


def build_parser() -> argparse.ArgumentParser:
    order = "\n".join(
        f"  {stage}: " + ", ".join(INPUT_COLUMNS) for stage in STAGES
    )
    parser = argparse.ArgumentParser(
        description=(
            "Build full and Original/Mask-only SVG, PDF, and PNG Waldenstrom figures."
        ),
        epilog=f"Required left-to-right, top-to-bottom input order:\n{order}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "images",
        nargs="*",
        type=Path,
        help="42 already-cropped square images in the order shown below",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        type=Path,
        help=(
            "Base output path; writes *_full and *_original_masks files in "
            "SVG, PDF, and PNG formats"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG resolution in dots per inch (default: 600)",
    )
    parser.add_argument(
        "--layout-preview",
        action="store_true",
        help="Build labeled placeholder figures; do not supply image paths",
    )
    parser.add_argument(
        "--flip-stages",
        nargs="+",
        choices=STAGES,
        default=(),
        metavar="STAGE",
        help=(
            "Mirror every AP and frog-leg panel left-to-right for the selected "
            "stages, for example: --flip-stages II-A II-B III-A"
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.dpi < 72:
        raise ValueError("--dpi must be at least 72.")
    if args.layout_preview and args.images:
        raise ValueError("Do not provide image paths with --layout-preview.")
    if not args.layout_preview:
        paths: Sequence[Path] | None = [path.resolve() for path in args.images]
        validate_images(paths)
    else:
        paths = None

    prefix = args.output_prefix.resolve()
    flip_stages = frozenset(args.flip_stages)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for layout in LAYOUTS:
        layout_prefix = prefix.parent / f"{prefix.name}_{layout.suffix}"
        svg_path = output_path(prefix, layout, "svg")
        build_svg(paths, svg_path, layout, flip_stages)
        export_png_and_pdf(svg_path, layout_prefix, args.dpi, layout)
        print(f"{layout.description} SVG: {svg_path}")
        print(f"{layout.description} PDF: {output_path(prefix, layout, 'pdf')}")
        print(f"{layout.description} PNG ({args.dpi} DPI): {output_path(prefix, layout, 'png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
