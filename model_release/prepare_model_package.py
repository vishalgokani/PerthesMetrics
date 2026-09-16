"""Normalize, validate, and checksum a PerthesMetrics nnU-Net release zip."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


DATASET = "Dataset001_PerthesMetrics"
MODEL = "nnUNetTrainer__nnUNetPlans__2d"
LABELS = {
    "background": 0, "acetabulum": 1, "gt": 2, "head": 3, "lt": 4,
    "neck": 5, "shaft": 6, "sourcil": 7, "triradiate cartilage": 8,
}
CHANNELS = {"0": "R", "1": "G", "2": "B"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_model_root(names: list[str]) -> str:
    suffix = f"/{MODEL}/dataset.json"
    candidates = [name.removesuffix("dataset.json") for name in names if name.endswith(suffix)]
    if not candidates:
        # Historical archive used the bare Dataset root.
        candidates = [name.removesuffix("dataset.json") for name in names if name.endswith(f"{MODEL}/dataset.json")]
    if len(set(candidates)) != 1:
        raise ValueError(f"Could not identify one {MODEL} model root in archive.")
    return candidates[0]


def normalized_name(name: str, old_root: str) -> str:
    if not name.startswith(old_root):
        raise ValueError(f"Unexpected archive entry outside model root: {name}")
    relative = name[len(old_root):]
    path = PurePosixPath(DATASET) / MODEL / relative
    if ".." in path.parts:
        raise ValueError(f"Unsafe archive path: {name}")
    return path.as_posix()


def normalize_json(relative: str, data: bytes) -> bytes:
    payload = json.loads(data.decode("utf-8"))
    if relative == "dataset.json":
        payload.update({
            "name": DATASET,
            "description": "Multiclass anatomic segmentation of RGB pediatric hip radiographs in Perthes disease",
            "channel_names": CHANNELS,
            "labels": LABELS,
            "file_ending": ".nii.gz",
        })
        payload.pop("region_class_order", None)
    elif relative == "plans.json":
        payload["dataset_name"] = DATASET
        configurations = payload.get("configurations", {})
        if "2d" not in configurations:
            raise ValueError("plans.json does not contain the required 2d configuration.")
    return (json.dumps(payload, indent=2) + "\n").encode("utf-8")


def validate_archive(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        root = f"{DATASET}/{MODEL}/"
        required = {root + "dataset.json", root + "plans.json"}
        required.update(root + f"fold_{fold}/checkpoint_final.pth" for fold in range(5))
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"Release archive is missing: {missing}")
        dataset = json.loads(archive.read(root + "dataset.json"))
        plans = json.loads(archive.read(root + "plans.json"))
        if (
            dataset.get("name") != DATASET
            or dataset.get("labels") != LABELS
            or dataset.get("channel_names") != CHANNELS
        ):
            raise ValueError("Release dataset.json has an incorrect identity, label map, or channel map.")
        if plans.get("dataset_name") != DATASET or "2d" not in plans.get("configurations", {}):
            raise ValueError("Release plans.json has an incorrect dataset identity or configuration.")
        return {"entries": len(names), "folds": [0, 1, 2, 3, 4], "dataset": DATASET, "model": MODEL}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the PerthesMetrics nnU-Net release package.")
    parser.add_argument("--source-zip", required=True, type=Path)
    parser.add_argument("--output-zip", type=Path, default=Path("perthesmetrics_nnunet_model.zip"))
    args = parser.parse_args()
    source, output = args.source_zip.resolve(), args.output_zip.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == output:
        raise ValueError("Source and output zip must be different files.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as input_archive:
        old_root = locate_model_root(input_archive.namelist())
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as target:
                for entry in input_archive.infolist():
                    if entry.is_dir():
                        continue
                    target_name = normalized_name(entry.filename, old_root)
                    relative = entry.filename[len(old_root):]
                    data = input_archive.read(entry)
                    if relative in ("dataset.json", "plans.json"):
                        data = normalize_json(relative, data)
                    target.writestr(target_name, data)
            if output.exists():
                output.unlink()
            temporary_path.replace(output)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
    details = validate_archive(output)
    checksum = sha256(output)
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    checksum_path.write_text(f"{checksum}  {output.name}\n", encoding="ascii")
    print(json.dumps({**details, "path": str(output), "bytes": output.stat().st_size, "sha256": checksum}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
