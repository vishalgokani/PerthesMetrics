#!/usr/bin/env python
# -----------------------------------------------------------------------------
# README
# -----------------------------------------------------------------------------
# Purpose
#   Build a publication-style Waldenström staging SVG with 28 image slots:
#       7 rows × [AP ground truth, AP mask, Frog ground truth, Frog mask].
#
#   The script interactively opens ONLY the 14 ground-truth radiographs.
#   For each radiograph:
#       1. Click-drag a crop region.
#       2. The selection is converted to a square crop.
#       3. Click "Approve" to accept, or "Redraw" to select again.
#   The paired mask is NEVER displayed and receives the identical crop box.
#
# Input
#   BMP/PNG/JPG/TIFF images are accepted. BMP inputs are converted to PNG after
#   cropping. Ground-truth and paired mask images must have identical dimensions.
#
# Output
#   - A self-contained SVG with all 28 cropped PNGs embedded as base64.
#   - An assets directory containing the cropped PNGs.
#   - A JSON file recording all 14 crop boxes for reproducibility.
#
# Install in Anaconda Prompt
#   conda create -n waldenstrom_fig python=3.11 -y
#   conda activate waldenstrom_fig
#   pip install pillow matplotlib
#
# Run
#   python waldenstrom_figure_builder.py ^
#     --ap-i-a-gt "path\AP_I_A.bmp" --ap-i-a-mask "path\AP_I_A_mask.bmp" ^
#     --ap-i-b-gt "path\AP_I_B.bmp" --ap-i-b-mask "path\AP_I_B_mask.bmp" ^
#     --ap-ii-a-gt "path\AP_II_A.bmp" --ap-ii-a-mask "path\AP_II_A_mask.bmp" ^
#     --ap-ii-b-gt "path\AP_II_B.bmp" --ap-ii-b-mask "path\AP_II_B_mask.bmp" ^
#     --ap-iii-a-gt "path\AP_III_A.bmp" --ap-iii-a-mask "path\AP_III_A_mask.bmp" ^
#     --ap-iii-b-gt "path\AP_III_B.bmp" --ap-iii-b-mask "path\AP_III_B_mask.bmp" ^
#     --ap-iv-gt "path\AP_IV.bmp" --ap-iv-mask "path\AP_IV_mask.bmp" ^
#     --frog-i-a-gt "path\Frog_I_A.bmp" --frog-i-a-mask "path\Frog_I_A_mask.bmp" ^
#     --frog-i-b-gt "path\Frog_I_B.bmp" --frog-i-b-mask "path\Frog_I_B_mask.bmp" ^
#     --frog-ii-a-gt "path\Frog_II_A.bmp" --frog-ii-a-mask "path\Frog_II_A_mask.bmp" ^
#     --frog-ii-b-gt "path\Frog_II_B.bmp" --frog-ii-b-mask "path\Frog_II_B_mask.bmp" ^
#     --frog-iii-a-gt "path\Frog_III_A.bmp" --frog-iii-a-mask "path\Frog_III_A_mask.bmp" ^
#     --frog-iii-b-gt "path\Frog_III_B.bmp" --frog-iii-b-mask "path\Frog_III_B_mask.bmp" ^
#     --frog-iv-gt "path\Frog_IV.bmp" --frog-iv-mask "path\Frog_IV_mask.bmp" ^
#     --output-svg "waldenstrom_figure.svg"
#
# Optional
#   --assets-dir cropped_assets
#   --crop-json waldenstrom_crop_boxes.json
#   --cell-px 900
#   --reuse-crops
#
# Notes
#   - The SVG uses Times New Roman, black/gray lines, and a formal minimalist
#     layout intended for manuscript figures.
#   - The SVG is self-contained; moving it does not break image links.
#   - Press Enter to approve a selected crop, R to redraw, Esc to abort.
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
from typing import Dict, Tuple

from PIL import Image

ROWS = [
    ("i-a", "I-A"),
    ("i-b", "I-B"),
    ("ii-a", "II-A"),
    ("ii-b", "II-B"),
    ("iii-a", "III-A"),
    ("iii-b", "III-B"),
    ("iv", "IV"),
]
VIEWS = [("ap", "AP"), ("frog", "Frog-leg lateral")]
KINDS = [("gt", "Ground truth"), ("mask", "Mask")]

FONT = "Times New Roman, Times, serif"

def flag_name(view: str, row_key: str, kind: str) -> str:
    return f"{view}-{row_key}-{kind}"

def dest_name(view: str, row_key: str, kind: str) -> str:
    return flag_name(view, row_key, kind).replace("-", "_")

def add_image_args(parser: argparse.ArgumentParser) -> None:
    for view, _ in VIEWS:
        for row_key, _ in ROWS:
            for kind, _ in KINDS:
                parser.add_argument(
                    f"--{flag_name(view, row_key, kind)}",
                    dest=dest_name(view, row_key, kind),
                    type=Path,
                    help=f"{view.upper()} {row_key.upper()} {kind} image path",
                )

def validate_inputs(args: argparse.Namespace) -> None:
    missing = []
    for view, _ in VIEWS:
        for row_key, _ in ROWS:
            for kind, _ in KINDS:
                value = getattr(args, dest_name(view, row_key, kind))
                if value is None:
                    missing.append(f"--{flag_name(view, row_key, kind)}")
                elif not value.exists():
                    raise FileNotFoundError(f"Input not found: {value}")
    if missing:
        raise SystemExit("Missing required image flags:\n  " + "\n  ".join(missing))

def square_box_from_drag(
    x0: float, y0: float, x1: float, y1: float, width: int, height: int
) -> Tuple[int, int, int, int]:
    """Convert an arbitrary drag rectangle into a square centered on the drag."""
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    side = max(abs(x1 - x0), abs(y1 - y0))
    side = max(2.0, min(side, width, height))

    left = cx - side / 2.0
    top = cy - side / 2.0

    left = min(max(0.0, left), width - side)
    top = min(max(0.0, top), height - side)

    l = int(round(left))
    t = int(round(top))
    s = int(round(side))
    r = min(width, l + s)
    b = min(height, t + s)

    # Preserve exact squareness after integer clipping.
    s = min(r - l, b - t)
    return (l, t, l + s, t + s)

def choose_square_crop(image_path: Path, title: str) -> Tuple[int, int, int, int]:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, RectangleSelector
    from matplotlib.patches import Rectangle

    img = Image.open(image_path)
    arr = img.convert("L")

    state = {"box": None, "approved": False, "patch": None}

    fig, ax = plt.subplots(figsize=(10, 9))
    plt.subplots_adjust(bottom=0.16)
    ax.imshow(arr, cmap="gray")
    ax.set_title(
        f"{title}\nDrag crop → Approve, or Redraw",
        fontfamily="serif",
        fontsize=13,
    )
    ax.axis("off")

    def draw_square(box):
        if state["patch"] is not None:
            state["patch"].remove()
        l, t, r, b = box
        state["patch"] = Rectangle(
            (l, t), r - l, b - t,
            fill=False, linewidth=1.8, edgecolor="black"
        )
        ax.add_patch(state["patch"])
        fig.canvas.draw_idle()

    def on_select(eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            return
        box = square_box_from_drag(
            eclick.xdata, eclick.ydata,
            erelease.xdata, erelease.ydata,
            img.width, img.height,
        )
        state["box"] = box
        draw_square(box)

    selector = RectangleSelector(
        ax, on_select,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=False,
    )

    approve_ax = fig.add_axes([0.32, 0.045, 0.16, 0.055])
    redraw_ax = fig.add_axes([0.52, 0.045, 0.16, 0.055])
    approve_btn = Button(approve_ax, "Approve")
    redraw_btn = Button(redraw_ax, "Redraw")

    def approve(_event=None):
        if state["box"] is None:
            ax.set_title(
                f"{title}\nSelect a crop before approving.",
                fontfamily="serif",
                fontsize=13,
            )
            fig.canvas.draw_idle()
            return
        state["approved"] = True
        plt.close(fig)

    def redraw(_event=None):
        state["box"] = None
        if state["patch"] is not None:
            state["patch"].remove()
            state["patch"] = None
        ax.set_title(
            f"{title}\nDrag a new crop → Approve",
            fontfamily="serif",
            fontsize=13,
        )
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "enter":
            approve()
        elif event.key and event.key.lower() == "r":
            redraw()
        elif event.key == "escape":
            plt.close(fig)

    approve_btn.on_clicked(approve)
    redraw_btn.on_clicked(redraw)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()

    if not state["approved"] or state["box"] is None:
        raise RuntimeError(f"Crop not approved for {image_path}")
    return state["box"]

def crop_pair(
    gt_path: Path,
    mask_path: Path,
    box: Tuple[int, int, int, int],
    out_gt: Path,
    out_mask: Path,
    cell_px: int,
) -> None:
    gt = Image.open(gt_path)
    mask = Image.open(mask_path)

    if gt.size != mask.size:
        raise ValueError(
            f"Dimension mismatch:\n  GT:   {gt_path} {gt.size}\n"
            f"  Mask: {mask_path} {mask.size}\n"
            "Paired masks must match the ground-truth image dimensions exactly."
        )

    gt_crop = gt.crop(box).resize((cell_px, cell_px), Image.Resampling.LANCZOS)
    mask_crop = mask.crop(box).resize((cell_px, cell_px), Image.Resampling.NEAREST)

    out_gt.parent.mkdir(parents=True, exist_ok=True)
    gt_crop.save(out_gt, format="PNG", optimize=True)
    mask_crop.save(out_mask, format="PNG", optimize=True)

def png_data_uri(path: Path) -> str:
    raw = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")

def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )

def build_svg(image_paths: Dict[str, Path] | None, output_svg: Path) -> None:
    # Publication-scale canvas, approximately 180 × 250 mm.
    W, H = 1800, 2500

    left_margin = 80
    row_label_w = 235
    col_gap = 34
    group_gap = 78
    cell = 285

    x0 = left_margin + row_label_w
    xs = [
        x0,
        x0 + cell + col_gap,
        x0 + 2 * (cell + col_gap) + group_gap,
        x0 + 3 * (cell + col_gap) + group_gap,
    ]

    header_y_major = 92
    header_y_sub = 162
    header_rule_y = 200

    y_start = 235
    within_gap = 22
    between_gap = 54
    row_ys = []
    y = y_start
    for i, (_row_key, _label) in enumerate(ROWS):
        row_ys.append(y)
        y += cell
        if i < len(ROWS) - 1:
            # Larger gap between I→II, II→III, III→IV than within A/B pairs.
            y += within_gap if i in (0, 2, 4) else between_gap

    def text(x, y, s, size=38, anchor="middle", weight="normal", italic=False):
        style = "italic" if italic else "normal"
        return (
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
            f'font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
            f'font-style="{style}" fill="#111">{svg_escape(s)}</text>'
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="180mm" height="250mm" viewBox="0 0 {W} {H}">',
        '<rect x="0" y="0" width="1800" height="2500" fill="white"/>',
        text(left_margin + row_label_w / 2, header_y_sub, "Waldenström stage", 34),
        text((xs[0] + xs[1] + cell) / 2, header_y_major, "AP", 45, weight="bold"),
        text((xs[2] + xs[3] + cell) / 2, header_y_major, "Frog-leg lateral", 45, weight="bold"),
        text(xs[0] + cell / 2, header_y_sub, "Ground truth", 31),
        text(xs[1] + cell / 2, header_y_sub, "Mask", 31),
        text(xs[2] + cell / 2, header_y_sub, "Ground truth", 31),
        text(xs[3] + cell / 2, header_y_sub, "Mask", 31),
        f'<line x1="{left_margin}" y1="{header_rule_y}" x2="{W-left_margin}" '
        f'y2="{header_rule_y}" stroke="#222" stroke-width="2"/>',
    ]

    # Subtle stage-group separators.
    separator_after = {1, 3, 5}
    for i, ((row_key, row_label), y) in enumerate(zip(ROWS, row_ys)):
        parts.append(text(left_margin + row_label_w - 28, y + cell / 2 + 12, row_label, 40, anchor="end"))

        slots = [
            ("ap", "gt"),
            ("ap", "mask"),
            ("frog", "gt"),
            ("frog", "mask"),
        ]

        for x, (view, kind) in zip(xs, slots):
            key = f"{view}_{row_key}_{kind}"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="#f7f7f7" stroke="#111" stroke-width="2"/>'
            )

            if image_paths and key in image_paths:
                uri = png_data_uri(image_paths[key])
                parts.append(
                    f'<image x="{x}" y="{y}" width="{cell}" height="{cell}" '
                    f'preserveAspectRatio="xMidYMid slice" href="{uri}" xlink:href="{uri}"/>'
                )
                # Restore a crisp border above the raster.
                parts.append(
                    f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                    f'fill="none" stroke="#111" stroke-width="2"/>'
                )
            else:
                placeholder = "RADIOGRAPH" if kind == "gt" else "MASK"
                parts.append(text(x + cell / 2, y + cell / 2 + 7, placeholder, 26))
                parts.append(
                    f'<line x1="{x+60}" y1="{y+cell-70}" x2="{x+cell-60}" y2="{y+70}" '
                    f'stroke="#c9c9c9" stroke-width="2"/>'
                )

        if i in separator_after:
            sep_y = y + cell + between_gap / 2
            parts.append(
                f'<line x1="{left_margin}" y1="{sep_y}" x2="{W-left_margin}" y2="{sep_y}" '
                f'stroke="#b5b5b5" stroke-width="1.5"/>'
            )

    parts.append("</svg>")
    output_svg.write_text("\n".join(parts), encoding="utf-8")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively crop 14 radiographs and build a 28-image Waldenström SVG."
    )
    add_image_args(parser)
    parser.add_argument("--output-svg", type=Path, default=Path("waldenstrom_figure.svg"))
    parser.add_argument("--assets-dir", type=Path, default=Path("waldenstrom_cropped_png"))
    parser.add_argument("--crop-json", type=Path, default=Path("waldenstrom_crop_boxes.json"))
    parser.add_argument("--cell-px", type=int, default=900)
    parser.add_argument(
        "--reuse-crops",
        action="store_true",
        help="Reuse crop boxes from --crop-json instead of reopening the crop UI.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    validate_inputs(args)

    crop_boxes: Dict[str, Tuple[int, int, int, int]] = {}
    if args.reuse_crops:
        if not args.crop_json.exists():
            raise FileNotFoundError(f"Crop JSON not found: {args.crop_json}")
        loaded = json.loads(args.crop_json.read_text(encoding="utf-8"))
        crop_boxes = {k: tuple(v) for k, v in loaded.items()}

    embedded: Dict[str, Path] = {}

    # Crop all 14 GT/mask pairs; masks are never shown in the crop UI.
    for view, view_label in VIEWS:
        for row_key, row_label in ROWS:
            gt = getattr(args, dest_name(view, row_key, "gt"))
            mask = getattr(args, dest_name(view, row_key, "mask"))
            pair_key = f"{view}_{row_key}"

            if pair_key not in crop_boxes:
                crop_boxes[pair_key] = choose_square_crop(
                    gt, f"{view_label} — Waldenström {row_label}"
                )

            box = tuple(crop_boxes[pair_key])
            gt_out = args.assets_dir / f"{pair_key}_gt.png"
            mask_out = args.assets_dir / f"{pair_key}_mask.png"
            crop_pair(gt, mask, box, gt_out, mask_out, args.cell_px)

            embedded[f"{view}_{row_key}_gt"] = gt_out
            embedded[f"{view}_{row_key}_mask"] = mask_out

    args.crop_json.write_text(
        json.dumps(crop_boxes, indent=2), encoding="utf-8"
    )
    build_svg(embedded, args.output_svg)

    print(f"SVG written:       {args.output_svg.resolve()}")
    print(f"Cropped PNGs:      {args.assets_dir.resolve()}")
    print(f"Crop coordinates:  {args.crop_json.resolve()}")

if __name__ == "__main__":
    main()
