"""Build publication Figure 1: cohort, annotation, and model workflow.

The four square radiograph panels are embedded in the SVG as PNG data URIs, so
the SVG remains self-contained. PDF and high-resolution PNG versions are
exported from that same vector source.
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image


CANVAS_WIDTH = 2100
CANVAS_HEIGHT = 1050
FONT = "Times New Roman, Times, serif"

CLASS_COLORS = (
    ("Acetabulum", "#FF0000"),
    ("Femoral head", "#00FFFF"),
    ("Femoral neck", "#F5AA42"),
    ("Femoral shaft", "#000080"),
    ("Sourcil", "#4169E1"),
    ("Triradiate cartilage", "#B8FF85"),
    ("Lesser trochanter", "#00FF00"),
    ("Greater trochanter", "#A020F0"),
)


def text(
    x: float,
    y: float,
    value: str,
    size: int,
    *,
    anchor: str = "middle",
    weight: str = "normal",
    transform: str | None = None,
) -> str:
    transform_attr = f' transform="{transform}"' if transform else ""
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
        f'fill="#111111"{transform_attr}>{escape(value)}</text>'
    )


def box(x: int, y: int, width: int, height: int, *, radius: int = 7) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" '
        'fill="#ffffff" stroke="#202020" stroke-width="3"/>'
    )


def validate_square_png(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError(f"Expected a PNG image: {path}")
        if image.width != image.height:
            raise ValueError(
                f"Expected a square image, found {image.width} x {image.height}: {path}"
            )


def png_data_uri(path: Path) -> str:
    validate_square_png(path)
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def image_panel(x: int, y: int, size: int, path: Path) -> list[str]:
    uri = png_data_uri(path)
    return [
        f'<image x="{x}" y="{y}" width="{size}" height="{size}" '
        f'preserveAspectRatio="xMidYMid meet" href="{uri}"/>',
        f'<rect x="{x}" y="{y}" width="{size}" height="{size}" '
        'fill="none" stroke="#202020" stroke-width="2"/>',
    ]


def build_svg(
    ap_original: Path,
    ap_ground_truth: Path,
    frog_original: Path,
    frog_ground_truth: Path,
    output_svg: Path,
) -> None:
    parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="210mm" height="105mm" viewBox="0 0 {CANVAS_WIDTH} {CANVAS_HEIGHT}">',
        '<title>Figure 1. Cohort construction, annotation, training, and prediction</title>',
        '<desc>Three-panel methods overview with embedded AP and frog-leg lateral radiographs.</desc>',
        '<defs>',
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="8" markerHeight="8" orient="auto-start-reverse">',
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#202020"/>',
        '</marker>',
        '</defs>',
        f'<rect width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}" fill="#ffffff"/>',
    ]

    # Equal-height panels and narrow gutters create a compact landscape figure.
    panels = ((24, 620), (664, 752), (1436, 640))
    for panel_x, panel_width in panels:
        parts.append(
            f'<rect x="{panel_x}" y="24" width="{panel_width}" height="1002" rx="12" '
            'fill="#fbfbfb" stroke="#b8b8b8" stroke-width="2"/>'
        )

    # (a) Cohort construction.
    parts.extend(
        [
            text(50, 80, "(a) Cohort construction", 41, anchor="start", weight="bold"),
            box(66, 150, 536, 230),
            text(334, 205, "Database export", 38, weight="bold"),
            text(334, 258, "Radiographs, n = 6,293", 34),
            text(334, 307, "Patients, n = 803", 34),
            text(334, 356, "38 centers in 15 countries", 34),
            '<path d="M334 380 V473 M334 473 H177 V518 M334 473 H491 V518" '
            'fill="none" stroke="#202020" stroke-width="4"/>',
            '<path d="M177 473 V518" fill="none" stroke="#202020" stroke-width="4" '
            'marker-end="url(#arrow)"/>',
            '<path d="M491 473 V518" fill="none" stroke="#202020" stroke-width="4" '
            'marker-end="url(#arrow)"/>',
            '<rect x="194" y="418" width="280" height="46" fill="#fbfbfb"/>',
            text(334, 452, "80% / 20% institution-level split", 27),
            box(40, 520, 274, 390),
            text(177, 574, "Training / validation", 29, weight="bold"),
            text(177, 628, "Radiographs", 29, weight="bold"),
            text(177, 665, "n = 4,463", 31),
            text(177, 716, "Patients", 29, weight="bold"),
            text(177, 753, "n = 646", 31),
            text(177, 804, "Institutions", 29, weight="bold"),
            text(177, 841, "n = 30", 31),
            box(354, 520, 274, 390),
            text(491, 574, "Held-out test", 33, weight="bold"),
            text(491, 628, "Radiographs", 29, weight="bold"),
            text(491, 665, "n = 1,830", 31),
            text(491, 716, "Patients", 29, weight="bold"),
            text(491, 753, "n = 157", 31),
            text(491, 804, "Institutions", 29, weight="bold"),
            text(491, 841, "n = 8", 31),
        ]
    )

    # (b) Data and ground-truth annotation.
    parts.extend(
        [
            text(690, 80, "(b) Data and annotation", 41, anchor="start", weight="bold"),
            text(930, 136, "AP", 33, weight="bold"),
            text(1234, 136, "Frog-leg lateral", 33, weight="bold"),
            text(
                700,
                305,
                "Original",
                34,
                weight="bold",
                transform="rotate(-90 700 305)",
            ),
            text(
                700,
                578,
                "Ground truth",
                34,
                weight="bold",
                transform="rotate(-90 700 578)",
            ),
        ]
    )
    image_size = 250
    parts.extend(image_panel(805, 160, image_size, ap_original))
    parts.extend(image_panel(1109, 160, image_size, frog_original))
    parts.extend(image_panel(805, 438, image_size, ap_ground_truth))
    parts.extend(image_panel(1109, 438, image_size, frog_ground_truth))

    parts.append(text(1040, 750, "Anatomical label classes", 32, weight="bold"))
    legend_positions = (
        (720, 796),
        (720, 848),
        (720, 900),
        (720, 952),
        (1062, 796),
        (1062, 848),
        (1062, 900),
        (1062, 952),
    )
    for (label, color), (x, y) in zip(CLASS_COLORS, legend_positions):
        parts.extend(
            [
                f'<rect x="{x}" y="{y - 28}" width="30" height="30" '
                f'fill="{color}" stroke="#202020" stroke-width="1.5"/>',
                text(x + 46, y - 3, label, 27, anchor="start"),
            ]
        )

    # (c) Training and prediction.
    parts.extend(
        [
            text(1462, 80, "(c) Training and prediction", 41, anchor="start", weight="bold"),
            box(1510, 125, 492, 105),
            text(1756, 171, "Annotated radiographs", 32, weight="bold"),
            text(1756, 211, "AP and frog-leg lateral views", 25),
            '<path d="M1756 230 V286" fill="none" stroke="#202020" stroke-width="4" '
            'marker-end="url(#arrow)"/>',
            box(1544, 288, 424, 100),
            text(1756, 350, "2D nnU-Net", 34, weight="bold"),
            '<path d="M1756 388 V444" fill="none" stroke="#202020" stroke-width="4" '
            'marker-end="url(#arrow)"/>',
            text(1756, 487, "5-fold cross-validation", 32, weight="bold"),
        ]
    )
    fold_xs = (1460, 1580, 1700, 1820, 1940)
    for fold_number, fold_x in enumerate(fold_xs, start=1):
        parts.extend(
            [
                box(fold_x, 520, 112, 82, radius=5),
                text(fold_x + 56, 572, f"Fold {fold_number}", 23, weight="bold"),
            ]
        )
    parts.extend(
        [
            '<path d="M1756 602 V669" fill="none" stroke="#202020" stroke-width="4" '
            'marker-end="url(#arrow)"/>',
            box(1544, 671, 424, 105),
            text(1756, 736, "Final ensemble", 33, weight="bold"),
            '<path d="M1756 776 V843" fill="none" stroke="#202020" stroke-width="4" '
            'marker-end="url(#arrow)"/>',
            box(1494, 845, 524, 130),
            text(1756, 901, "Prediction", 34, weight="bold"),
            text(1756, 948, "Eight-class masks on held-out radiographs", 27),
        ]
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
    cairosvg.svg2pdf(url=str(svg_path), write_to=str(output_prefix.with_suffix(".pdf")))
    cairosvg.svg2png(
        url=str(svg_path),
        write_to=str(output_prefix.with_suffix(".png")),
        output_width=round(210 / 25.4 * dpi),
        output_height=round(105 / 25.4 * dpi),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ap-original", type=Path, required=True)
    parser.add_argument("--ap-ground-truth", type=Path, required=True)
    parser.add_argument("--frog-original", type=Path, required=True)
    parser.add_argument("--frog-ground-truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dpi < 300:
        raise ValueError("Use --dpi 300 or higher for publication output.")
    prefix = args.output_dir / "Figure1"
    svg_path = prefix.with_suffix(".svg")
    build_svg(
        args.ap_original,
        args.ap_ground_truth,
        args.frog_original,
        args.frog_ground_truth,
        svg_path,
    )
    export_png_and_pdf(svg_path, prefix, args.dpi)
    print(f"Wrote {svg_path}")
    print(f"Wrote {prefix.with_suffix('.pdf')}")
    print(f"Wrote {prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
