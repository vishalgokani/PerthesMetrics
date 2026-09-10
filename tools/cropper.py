"""Interactively apply one square crop to as many as ten images.

The crop is selected on the first image and applied at the same pixel
coordinates to every input. Outputs are PNG files so they can be passed
directly to publication-figure tools without modifying the source images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageOps


Box = tuple[int, int, int, int]


def square_box_from_drag(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: int,
    height: int,
) -> Box:
    """Return the largest drag-defined square that remains inside the image."""
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    side = max(abs(x1 - x0), abs(y1 - y0))
    side = max(1.0, min(side, float(width), float(height)))

    left = min(max(0.0, center_x - side / 2.0), width - side)
    top = min(max(0.0, center_y - side / 2.0), height - side)
    integer_side = max(1, min(int(round(side)), width, height))
    integer_left = min(max(0, int(round(left))), width - integer_side)
    integer_top = min(max(0, int(round(top))), height - integer_side)
    return (
        integer_left,
        integer_top,
        integer_left + integer_side,
        integer_top + integer_side,
    )


def read_oriented_image(path: Path) -> Image.Image:
    """Load an image and apply its EXIF orientation before measuring/cropping."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).copy()


def choose_square_crop(image_path: Path) -> Box:
    """Show the first image and return the square crop approved by the user."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.widgets import Button, RectangleSelector

    image = read_oriented_image(image_path)
    state: dict[str, Box | Rectangle | bool | None] = {
        "box": None,
        "patch": None,
        "confirmed": False,
    }

    figure, axis = plt.subplots(figsize=(10, 9))
    figure.subplots_adjust(bottom=0.16)
    axis.imshow(image, cmap="gray" if image.mode in {"1", "L", "I", "F"} else None)
    axis.set_title("Draw a crop on the first image, then click Confirm")
    axis.axis("off")

    def display_box(box: Box) -> None:
        old_patch = state["patch"]
        if isinstance(old_patch, Rectangle):
            old_patch.remove()
        left, top, right, bottom = box
        patch = Rectangle(
            (left, top),
            right - left,
            bottom - top,
            fill=False,
            edgecolor="#00ffff",
            linewidth=2.0,
        )
        axis.add_patch(patch)
        state["patch"] = patch
        figure.canvas.draw_idle()

    def on_select(click, release) -> None:
        if click.xdata is None or click.ydata is None:
            return
        if release.xdata is None or release.ydata is None:
            return
        box = square_box_from_drag(
            click.xdata,
            click.ydata,
            release.xdata,
            release.ydata,
            image.width,
            image.height,
        )
        state["box"] = box
        display_box(box)

    selector = RectangleSelector(
        axis,
        on_select,
        useblit=True,
        button=[1],
        minspanx=3,
        minspany=3,
        spancoords="pixels",
    )
    # Matplotlib widgets stop responding if their only strong reference is lost.
    state["selector"] = selector

    confirm_axis = figure.add_axes([0.31, 0.045, 0.18, 0.06])
    redraw_axis = figure.add_axes([0.52, 0.045, 0.18, 0.06])
    confirm_button = Button(confirm_axis, "Confirm")
    redraw_button = Button(redraw_axis, "Redraw")

    def confirm(_event=None) -> None:
        if state["box"] is None:
            axis.set_title("Draw a crop before clicking Confirm")
            figure.canvas.draw_idle()
            return
        state["confirmed"] = True
        plt.close(figure)

    def redraw(_event=None) -> None:
        state["box"] = None
        old_patch = state["patch"]
        if isinstance(old_patch, Rectangle):
            old_patch.remove()
        state["patch"] = None
        axis.set_title("Draw a new crop, then click Confirm")
        figure.canvas.draw_idle()

    confirm_button.on_clicked(confirm)
    redraw_button.on_clicked(redraw)

    def on_key(event) -> None:
        if event.key == "enter":
            confirm()
        elif event.key and event.key.lower() == "r":
            redraw()
        elif event.key == "escape":
            plt.close(figure)

    figure.canvas.mpl_connect("key_press_event", on_key)
    plt.show(block=True)

    box = state["box"]
    if not state["confirmed"] or not isinstance(box, tuple):
        raise RuntimeError("Crop cancelled: no crop was confirmed.")
    return box


def validate_inputs(paths: Sequence[Path]) -> tuple[int, int]:
    if not 1 <= len(paths) <= 10:
        raise ValueError("Provide between 1 and 10 input images.")

    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Input image not found: " + str(missing[0]))

    sizes = [(path, read_oriented_image(path).size) for path in paths]
    expected = sizes[0][1]
    mismatches = [(path, size) for path, size in sizes if size != expected]
    if mismatches:
        details = "; ".join(f"{path.name}: {size}" for path, size in sizes)
        raise ValueError(
            "All inputs must have identical dimensions because the same pixel "
            f"crop is applied to each image. Found {details}."
        )
    return expected


def crop_images(
    paths: Sequence[Path], output_dir: Path, box: Box, suffix: str
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = [output_dir / f"{path.stem}{suffix}.png" for path in paths]
    if len(set(destinations)) != len(destinations):
        raise ValueError(
            "Input filenames would produce duplicate output names. Rename the "
            "inputs or run them in separate output directories."
        )

    for source, destination in zip(paths, destinations):
        cropped = read_oriented_image(source).crop(box)
        if cropped.width != cropped.height:
            raise AssertionError(f"Internal error: crop is not square: {box}")
        cropped.save(destination, format="PNG", optimize=True)
    return destinations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select one square crop on the first image and apply it to 1-10 "
            "same-sized images. Crops are written as PNG files."
        )
    )
    parser.add_argument("images", nargs="+", type=Path, help="1-10 input images")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--suffix",
        default="_cropped",
        help="Text appended to each input stem (default: _cropped)",
    )
    parser.add_argument(
        "--metadata-name",
        help=(
            "Metadata filename within --output-dir (default: derived from the "
            "first input filename)"
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    image_paths = [path.resolve() for path in args.images]
    width, height = validate_inputs(image_paths)
    box = choose_square_crop(image_paths[0])
    outputs = crop_images(image_paths, args.output_dir.resolve(), box, args.suffix)

    metadata = {
        "source_dimensions_px": [width, height],
        "crop_box_px": list(box),
        "crop_size_px": box[2] - box[0],
        "inputs": [path.name for path in image_paths],
        "outputs": [path.name for path in outputs],
    }
    metadata_name = args.metadata_name or f"{image_paths[0].stem}_crop_metadata.json"
    metadata_path = args.output_dir.resolve() / metadata_name
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Saved {len(outputs)} square PNG crop(s) to: {args.output_dir.resolve()}")
    print(f"Crop box (left, top, right, bottom): {box}")
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
