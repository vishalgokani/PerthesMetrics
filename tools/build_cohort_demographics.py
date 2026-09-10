"""Build a publication table of cohort demographics and imaging composition.

The source radiograph CSV contains one row per postprocessed hip image. Left and
right hip images from the same acquisition are paired before radiographs are
counted. Waldenstrom ratings are joined to hip images by the exact filename
(case-insensitive), preserving the distinction between ``Patient_`` filenames
and the legacy numeric filename namespace.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont


MISSING = {"", "na", "n/a", "nan", "none", "unknown", "not reported"}
STAGE_ORDER = ("Ia", "Ib", "IIa", "IIb", "IIIa", "IIIb", "IV")
SIDE_SUFFIX = re.compile(r"_(L|R)\.bmp$", re.IGNORECASE)
PUBLICATION_FONT = "Times New Roman"


@dataclass(frozen=True)
class TableRow:
    section: str
    characteristic: str
    train: str
    test: str
    total: str


@dataclass(frozen=True)
class AnalysisResult:
    table_rows: tuple[TableRow, ...]
    patient_counts: Mapping[str, int]
    radiograph_counts: Mapping[str, int]
    hip_counts: Mapping[str, int]
    excluded_unsplit_images: int
    source_split_patient_counts: Mapping[str, int]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def require_fields(rows: Sequence[Mapping[str, str]], fields: Iterable[str], source: str) -> None:
    if not rows:
        raise ValueError(f"{source} contains no data rows")
    missing = sorted(set(fields) - set(rows[0]))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def clean(value: object) -> str:
    return "" if value is None else str(value).strip()


def is_missing(value: object) -> bool:
    return clean(value).lower() in MISSING


def normalized_split(value: object) -> str:
    token = clean(value).lower().replace(" ", "")
    if token in {"train", "train/val", "trainval", "training", "validation"}:
        return "Train"
    if token in {"test", "testing"}:
        return "Test"
    raise ValueError(f"Unsupported split label: {value!r}")


def build_split_map(rows: Sequence[Mapping[str, str]]) -> tuple[dict[str, str], dict[str, int]]:
    require_fields(rows, ("institution", "split"), "train/test split CSV")
    mapping: dict[str, str] = {}
    source_counts = {"Train": 0, "Test": 0}
    for row in rows:
        institution = clean(row["institution"])
        if not institution:
            raise ValueError("train/test split CSV contains a blank institution")
        split = normalized_split(row["split"])
        if institution in mapping and mapping[institution] != split:
            raise ValueError(f"Institution {institution!r} is assigned to multiple splits")
        mapping[institution] = split
        if "n_patients" in row and not is_missing(row["n_patients"]):
            try:
                source_counts[split] += int(float(clean(row["n_patients"])))
            except ValueError as error:
                raise ValueError(f"Invalid n_patients for {institution!r}: {row['n_patients']!r}") from error
    return mapping, source_counts


def patient_namespace(filename: str) -> str:
    return "patient" if clean(filename).lower().startswith("patient_") else "legacy"


def one_nonmissing(values: Iterable[object], field: str, patient_key: tuple[str, str]) -> str:
    unique = {clean(value) for value in values if not is_missing(value)}
    if len(unique) > 1:
        raise ValueError(f"Patient {patient_key!r} has conflicting {field}: {sorted(unique)}")
    return next(iter(unique), "")


def normalize_gender(value: str) -> str:
    token = clean(value).lower()
    if token in {"m", "male"}:
        return "Male"
    if token in {"f", "female"}:
        return "Female"
    if token in MISSING:
        return "Unavailable"
    raise ValueError(f"Unsupported gender value: {value!r}")


def normalize_stage(value: str) -> str:
    token = clean(value).lower().replace("stage", "").replace(" ", "")
    mapping = {stage.lower(): stage for stage in STAGE_ORDER}
    if token in MISSING:
        return ""
    if token not in mapping:
        raise ValueError(f"Unsupported final Waldenstrom rating: {value!r}")
    return mapping[token]


def infer_view(row: Mapping[str, str]) -> str:
    token = clean(row.get("AP_or_frog", "")).lower()
    if token == "ap":
        return "AP"
    if token in {"frog", "frog-leg", "frogleg"}:
        return "Frog"
    filename = clean(row["file_name"])
    return "Frog" if "frog" in filename.lower() else "AP"


def format_count(count: int, denominator: int | None = None) -> str:
    if denominator is None:
        return f"{count:,}"
    percentage = 100.0 * count / denominator if denominator else 0.0
    return f"{count:,} ({percentage:.1f}%)"


def format_age(values: Sequence[float]) -> str:
    if not values:
        return "NA"
    if len(values) == 1:
        return f"{values[0]:.2f} (n=1)"
    return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f} (n={len(values):,})"


def exposure_key(filename: str) -> str:
    match = SIDE_SUFFIX.search(clean(filename))
    if not match:
        raise ValueError(f"Split hip filename lacks _L.bmp or _R.bmp suffix: {filename!r}")
    return SIDE_SUFFIX.sub(".bmp", clean(filename)).lower()


def build_stage_map(rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
    require_fields(rows, ("ap_file_name", "frog_file_name", "final rating"), "Waldenstrom CSV")
    mapping: dict[str, str] = {}
    for row in rows:
        stage = normalize_stage(row["final rating"])
        for field in ("ap_file_name", "frog_file_name"):
            filename = clean(row[field]).lower()
            if not filename:
                raise ValueError(f"Waldenstrom CSV contains a blank {field}")
            if filename in mapping:
                raise ValueError(f"Waldenstrom filename appears more than once: {row[field]!r}")
            mapping[filename] = stage
    return mapping


def analyze(
    radiograph_rows: Sequence[Mapping[str, str]],
    staging_rows: Sequence[Mapping[str, str]],
    split_rows: Sequence[Mapping[str, str]],
) -> AnalysisResult:
    require_fields(
        radiograph_rows,
        (
            "file_name",
            "parsed_ID",
            "institution",
            "gender",
            "age_initial_presentation",
            "affected_vs_unaffected",
            "standardized_laterality",
        ),
        "radiograph demographics CSV",
    )
    split_map, source_split_counts = build_split_map(split_rows)
    stage_map = build_stage_map(staging_rows)

    source_filenames = {clean(row["file_name"]).lower() for row in radiograph_rows}
    missing_stage_files = sorted(set(stage_map) - source_filenames)
    if missing_stage_files:
        raise ValueError(
            f"{len(missing_stage_files)} Waldenstrom filenames are absent from radiographdata.csv; "
            f"first: {missing_stage_files[0]!r}"
        )

    grouped_patients: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in radiograph_rows:
        institution = clean(row["institution"])
        if institution not in split_map:
            raise ValueError(f"No train/test split for institution {institution!r}")
        patient_id = clean(row["parsed_ID"])
        if not patient_id:
            raise ValueError(f"Blank parsed_ID for {row['file_name']!r}")
        key = (patient_namespace(clean(row["file_name"])), patient_id)
        grouped_patients[key].append(row)

    patients: list[dict[str, object]] = []
    for key, rows in grouped_patients.items():
        institution = one_nonmissing((row["institution"] for row in rows), "institution", key)
        gender = normalize_gender(one_nonmissing((row["gender"] for row in rows), "gender", key))
        age_text = one_nonmissing(
            (row["age_initial_presentation"] for row in rows), "age_initial_presentation", key
        )
        age = None
        if age_text:
            try:
                age = float(age_text)
            except ValueError as error:
                raise ValueError(f"Patient {key!r} has invalid age {age_text!r}") from error
        patients.append(
            {"key": key, "institution": institution, "split": split_map[institution], "gender": gender, "age": age}
        )

    patient_counts = Counter(patient["split"] for patient in patients)
    patient_counts["Total"] = len(patients)
    if source_split_counts != {"Train": patient_counts["Train"], "Test": patient_counts["Test"]}:
        warnings.warn(
            "The split CSV n_patients totals do not match namespace-aware patient counts "
            f"(source={source_split_counts}, observed={{'Train': {patient_counts['Train']}, "
            f"'Test': {patient_counts['Test']}}}). Institution assignments were used.",
            stacklevel=2,
        )

    hip_rows = [
        row
        for row in radiograph_rows
        if clean(row["standardized_laterality"]).lower() in {"left", "right"}
    ]
    excluded_unsplit = len(radiograph_rows) - len(hip_rows)
    grouped_exposures: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in hip_rows:
        grouped_exposures[exposure_key(clean(row["file_name"]))].append(row)

    radiographs: list[dict[str, str]] = []
    for key, rows in grouped_exposures.items():
        sides = {clean(row["standardized_laterality"]).lower() for row in rows}
        if len(rows) != 2 or sides != {"left", "right"}:
            raise ValueError(f"Radiograph {key!r} does not have exactly one left and one right hip image")
        institutions = {clean(row["institution"]) for row in rows}
        views = {infer_view(row) for row in rows}
        if len(institutions) != 1 or len(views) != 1:
            raise ValueError(f"Paired hip images disagree on institution or view for {key!r}")
        institution = next(iter(institutions))
        radiographs.append({"split": split_map[institution], "view": next(iter(views))})

    radiograph_counts: Counter[str] = Counter()
    for radiograph in radiographs:
        radiograph_counts[radiograph["split"]] += 1
        radiograph_counts[f"{radiograph['split']}:{radiograph['view']}"] += 1
    radiograph_counts["Total"] = len(radiographs)
    radiograph_counts["Total:AP"] = sum(item["view"] == "AP" for item in radiographs)
    radiograph_counts["Total:Frog"] = sum(item["view"] == "Frog" for item in radiographs)

    hip_categories: dict[str, Counter[str]] = {"Train": Counter(), "Test": Counter()}
    for row in hip_rows:
        split = split_map[clean(row["institution"])]
        affected = clean(row["affected_vs_unaffected"]).lower()
        stage = stage_map.get(clean(row["file_name"]).lower(), "")
        if stage and affected != "affected":
            raise ValueError(f"Final stage is assigned to a non-affected hip: {row['file_name']!r}")
        if affected == "unaffected":
            category = "Unaffected"
        elif affected == "affected" and stage:
            category = stage
        elif affected == "affected":
            category = "Affected, stage unavailable"
        else:
            category = "Status unavailable"
        hip_categories[split][category] += 1

    hip_counts: Counter[str] = Counter()
    for split in ("Train", "Test"):
        for category, count in hip_categories[split].items():
            hip_counts[f"{split}:{category}"] = count
            hip_counts[f"Total:{category}"] += count
        hip_counts[split] = sum(hip_categories[split].values())
    hip_counts["Total"] = hip_counts["Train"] + hip_counts["Test"]

    def patient_value(split: str, predicate) -> str:
        cohort = [patient for patient in patients if split == "Total" or patient["split"] == split]
        return format_count(sum(predicate(patient) for patient in cohort), len(cohort))

    def age_value(split: str) -> str:
        values = [
            float(patient["age"])
            for patient in patients
            if (split == "Total" or patient["split"] == split) and patient["age"] is not None
        ]
        return format_age(values)

    def hip_value(split: str, category: str, denominator_category: str | None = None) -> str:
        count = hip_counts[f"{split}:{category}"]
        if denominator_category is None:
            denominator = hip_counts[split]
        elif denominator_category == "Affected":
            denominator = hip_counts[f"{split}:Affected, stage unavailable"] + sum(
                hip_counts[f"{split}:{stage}"] for stage in STAGE_ORDER
            )
        elif denominator_category == "Staged":
            denominator = sum(hip_counts[f"{split}:{stage}"] for stage in STAGE_ORDER)
        else:
            raise ValueError(f"Unknown denominator category: {denominator_category}")
        return format_count(count, denominator)

    rows: list[TableRow] = []
    rows.append(
        TableRow(
            "Participants",
            "Patients",
            format_count(patient_counts["Train"]),
            format_count(patient_counts["Test"]),
            format_count(patient_counts["Total"]),
        )
    )
    rows.append(TableRow("Participants", "Age at initial presentation, mean ± SD (years)", age_value("Train"), age_value("Test"), age_value("Total")))
    for gender in ("Male", "Female", "Unavailable"):
        rows.append(
            TableRow(
                "Participants",
                f"Sex: {gender.lower()}",
                patient_value("Train", lambda patient, value=gender: patient["gender"] == value),
                patient_value("Test", lambda patient, value=gender: patient["gender"] == value),
                patient_value("Total", lambda patient, value=gender: patient["gender"] == value),
            )
        )

    rows.extend(
        [
            TableRow("Radiographs", "Total radiographs", format_count(radiograph_counts["Train"]), format_count(radiograph_counts["Test"]), format_count(radiograph_counts["Total"])),
            TableRow("Radiographs", "AP radiographs", format_count(radiograph_counts["Train:AP"]), format_count(radiograph_counts["Test:AP"]), format_count(radiograph_counts["Total:AP"])),
            TableRow("Radiographs", "Frog-leg lateral radiographs", format_count(radiograph_counts["Train:Frog"]), format_count(radiograph_counts["Test:Frog"]), format_count(radiograph_counts["Total:Frog"])),
        ]
    )

    def affected_count(split: str) -> int:
        return hip_counts[f"{split}:Affected, stage unavailable"] + sum(
            hip_counts[f"{split}:{stage}"] for stage in STAGE_ORDER
        )

    rows.append(TableRow("Hip-level analysis", "Total hips", format_count(hip_counts["Train"]), format_count(hip_counts["Test"]), format_count(hip_counts["Total"])))
    rows.append(TableRow("Hip-level analysis", "Unaffected hips", hip_value("Train", "Unaffected"), hip_value("Test", "Unaffected"), hip_value("Total", "Unaffected")))
    rows.append(
        TableRow(
            "Hip-level analysis",
            "Affected hips",
            format_count(affected_count("Train"), hip_counts["Train"]),
            format_count(affected_count("Test"), hip_counts["Test"]),
            format_count(affected_count("Total"), hip_counts["Total"]),
        )
    )
    rows.append(
        TableRow(
            "Hip-level analysis",
            "Affected hips with final staging",
            format_count(sum(hip_counts[f"Train:{stage}"] for stage in STAGE_ORDER), affected_count("Train")),
            format_count(sum(hip_counts[f"Test:{stage}"] for stage in STAGE_ORDER), affected_count("Test")),
            format_count(sum(hip_counts[f"Total:{stage}"] for stage in STAGE_ORDER), affected_count("Total")),
        )
    )
    for stage in STAGE_ORDER:
        rows.append(TableRow("Hip-level analysis", f"  Waldenstrom stage {stage}", hip_value("Train", stage, "Staged"), hip_value("Test", stage, "Staged"), hip_value("Total", stage, "Staged")))
    rows.append(
        TableRow(
            "Hip-level analysis",
            "Affected hips without final staging",
            hip_value("Train", "Affected, stage unavailable", "Affected"),
            hip_value("Test", "Affected, stage unavailable", "Affected"),
            hip_value("Total", "Affected, stage unavailable", "Affected"),
        )
    )
    rows.append(TableRow("Hip-level analysis", "Hip status unavailable", hip_value("Train", "Status unavailable"), hip_value("Test", "Status unavailable"), hip_value("Total", "Status unavailable")))

    return AnalysisResult(
        table_rows=tuple(rows),
        patient_counts=dict(patient_counts),
        radiograph_counts=dict(radiograph_counts),
        hip_counts=dict(hip_counts),
        excluded_unsplit_images=excluded_unsplit,
        source_split_patient_counts=source_split_counts,
    )


def write_csv(path: Path, rows: Sequence[TableRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("Section", "Characteristic", "Train Cohort", "Test Cohort", "Total Cohort"))
        for row in rows:
            writer.writerow((row.section, row.characteristic, row.train, row.test, row.total))


def render_table(base_path: Path, rows: Sequence[TableRow]) -> None:
    """Render a clean one-page vector table as SVG and PDF."""
    findfont(PUBLICATION_FONT, fallback_to_default=False)
    display_rows: list[tuple[str, str, str, str, bool]] = []
    section = None
    for row in rows:
        if row.section != section:
            section = row.section
            display_rows.append((section, "", "", "", True))
        display_rows.append((row.characteristic, row.train, row.test, row.total, False))

    figure_height = 0.38 * len(display_rows) + 1.05
    with plt.rc_context(
        {
            "font.family": PUBLICATION_FONT,
            "font.size": 9.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    ):
        figure, axis = plt.subplots(figsize=(9.2, figure_height))
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")
        axis.text(0.02, 0.965, "Cohort demographics and imaging composition", fontsize=12, fontweight="bold", va="top")
        columns = (0.03, 0.59, 0.755, 0.92)
        top = 0.89
        row_height = 0.84 / (len(display_rows) + 1)
        headers = ("Characteristic", "Train Cohort", "Test Cohort", "Total Cohort")
        for x, label in zip(columns, headers):
            axis.text(x, top, label, fontweight="bold", ha="left" if x == columns[0] else "center", va="center")
        axis.plot((0.02, 0.98), (top - row_height * 0.48,) * 2, color="black", linewidth=0.8)

        y = top - row_height
        for characteristic, train, test, total, is_section in display_rows:
            if is_section:
                axis.add_patch(
                    plt.Rectangle((0.02, y - row_height * 0.43), 0.96, row_height * 0.86, facecolor="#E8E8E8", edgecolor="none")
                )
                axis.text(columns[0], y, characteristic, fontweight="bold", ha="left", va="center")
            else:
                axis.text(columns[0], y, characteristic, ha="left", va="center")
                axis.text(columns[1], y, train, ha="center", va="center")
                axis.text(columns[2], y, test, ha="center", va="center")
                axis.text(columns[3], y, total, ha="center", va="center")
            y -= row_height

        axis.plot((0.02, 0.98), (y + row_height * 0.48,) * 2, color="black", linewidth=0.8)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(base_path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.08)
        figure.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
        plt.close(figure)


def build_outputs(
    radiograph_csv: Path,
    staging_csv: Path,
    split_csv: Path,
    output_dir: Path,
    basename: str = "cohort_demographics",
) -> AnalysisResult:
    result = analyze(read_rows(radiograph_csv), read_rows(staging_csv), read_rows(split_csv))
    base_path = output_dir / basename
    write_csv(base_path.with_suffix(".csv"), result.table_rows)
    render_table(base_path, result.table_rows)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/test cohort demographics without double-counting left/right hip images as radiographs."
    )
    parser.add_argument("--radiograph-csv", required=True, type=Path, help="Path to radiographdata.csv")
    parser.add_argument("--staging-csv", required=True, type=Path, help="Path to final_waldenstrom_staging.csv")
    parser.add_argument("--split-csv", required=True, type=Path, help="Path to the institution-level train/test split CSV")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for CSV, SVG, and PDF outputs")
    parser.add_argument("--basename", default="cohort_demographics", help="Output filename stem")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_outputs(
        args.radiograph_csv.resolve(),
        args.staging_csv.resolve(),
        args.split_csv.resolve(),
        args.output_dir.resolve(),
        args.basename,
    )
    print(
        f"Patients: {result.patient_counts['Train']:,} train, {result.patient_counts['Test']:,} test, "
        f"{result.patient_counts['Total']:,} total"
    )
    print(
        f"Radiographs: {result.radiograph_counts['Train']:,} train, {result.radiograph_counts['Test']:,} test, "
        f"{result.radiograph_counts['Total']:,} total"
    )
    print(
        f"Hips: {result.hip_counts['Train']:,} train, {result.hip_counts['Test']:,} test, "
        f"{result.hip_counts['Total']:,} total"
    )
    print(f"Excluded unsplit/source images: {result.excluded_unsplit_images:,}")
    print(f"Wrote {args.output_dir.resolve() / args.basename}.csv/.svg/.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
