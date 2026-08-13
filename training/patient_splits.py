"""Create and audit deterministic patient-grouped nnU-Net cross-validation splits."""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


def infer_patient_id(case_id: str) -> str:
    """Infer a stable patient ID from supported PerthesMetrics case names.

    Synthetic examples: ``Patient_1001_AP_...`` and ``1001_Frog_...`` both
    map to ``1001``; ``X013_L_...`` maps to ``X013``. An explicit patient map is preferred when
    filenames do not follow these conventions.
    """
    value = case_id.strip()
    match = re.match(r"(?i)^patient[_ -]*0*(\d+)(?:[_ -]|$)", value)
    if match:
        return str(int(match.group(1)))
    match = re.match(r"^0*(\d+)(?:[_ -]|$)", value)
    if match:
        return str(int(match.group(1)))
    match = re.match(r"(?i)^([A-Z]+)0*(\d+)(?:[_ -]|$)", value)
    if match:
        return f"{match.group(1).upper()}{int(match.group(2)):03d}"
    raise ValueError(
        f"Cannot infer patient ID from case '{case_id}'. Add it to patient_groups.csv "
        "with columns case_id,patient_id."
    )


def load_patient_map(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"case_id", "patient_id"}.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain case_id and patient_id columns.")
        mapping: dict[str, str] = {}
        for row in reader:
            case_id = (row.get("case_id") or "").strip()
            patient_id = (row.get("patient_id") or "").strip()
            if not case_id or not patient_id:
                raise ValueError(f"Blank case_id or patient_id in {path}.")
            if case_id in mapping and mapping[case_id] != patient_id:
                raise ValueError(f"Case {case_id} has conflicting patient IDs in {path}.")
            mapping[case_id] = patient_id
    return mapping


def resolve_groups(case_ids: list[str], patient_map: dict[str, str]) -> dict[str, str]:
    unknown_map_cases = sorted(set(patient_map) - set(case_ids))
    if unknown_map_cases:
        raise ValueError(f"Patient map contains {len(unknown_map_cases)} unknown cases, including {unknown_map_cases[:3]}.")
    if patient_map:
        missing_map_cases = sorted(set(case_ids) - set(patient_map))
        if missing_map_cases:
            raise ValueError(
                f"Patient map is incomplete: {len(missing_map_cases)} cases are missing, including {missing_map_cases[:3]}."
            )
    return {case: patient_map.get(case) or infer_patient_id(case) for case in case_ids}


def make_grouped_splits(
    case_to_patient: dict[str, str], num_splits: int = 5, seed: int = 2026
) -> tuple[list[dict[str, list[str]]], dict[str, int]]:
    if num_splits < 2:
        raise ValueError("num_splits must be at least 2.")
    grouped: dict[str, list[str]] = defaultdict(list)
    for case, patient in case_to_patient.items():
        grouped[patient].append(case)
    if len(grouped) < num_splits:
        raise ValueError(f"Need at least {num_splits} patients; found {len(grouped)}.")

    # Seeded tie-breaking plus largest-group-first bin packing produces balanced,
    # deterministic validation folds without splitting a patient.
    rng = random.Random(seed)
    patients = list(grouped)
    rng.shuffle(patients)
    patients.sort(key=lambda patient: len(grouped[patient]), reverse=True)
    fold_patients: list[list[str]] = [[] for _ in range(num_splits)]
    fold_case_counts = [0] * num_splits
    for patient in patients:
        fold = min(range(num_splits), key=lambda index: (fold_case_counts[index], len(fold_patients[index]), index))
        fold_patients[fold].append(patient)
        fold_case_counts[fold] += len(grouped[patient])

    all_cases = set(case_to_patient)
    patient_to_fold: dict[str, int] = {}
    splits: list[dict[str, list[str]]] = []
    for fold, validation_patients in enumerate(fold_patients):
        for patient in validation_patients:
            patient_to_fold[patient] = fold
        validation = sorted(case for patient in validation_patients for case in grouped[patient])
        training = sorted(all_cases - set(validation))
        splits.append({"train": training, "val": validation})
    audit_splits(splits, case_to_patient, num_splits)
    return splits, patient_to_fold


def audit_splits(
    splits: list[dict[str, list[str]]], case_to_patient: dict[str, str], num_splits: int = 5
) -> dict[str, object]:
    if len(splits) != num_splits:
        raise ValueError(f"Expected {num_splits} folds, found {len(splits)}.")
    all_cases = set(case_to_patient)
    validation_counts: Counter[str] = Counter()
    fold_summaries = []
    for index, split in enumerate(splits):
        train, val = set(split["train"]), set(split["val"])
        if train & val:
            raise ValueError(f"Fold {index} has {len(train & val)} cases in both train and validation.")
        if train | val != all_cases:
            raise ValueError(f"Fold {index} does not cover the complete dataset.")
        train_patients = {case_to_patient[case] for case in train}
        val_patients = {case_to_patient[case] for case in val}
        leaked = train_patients & val_patients
        if leaked:
            raise ValueError(f"Fold {index} leaks {len(leaked)} patients, including {sorted(leaked)[:3]}.")
        validation_counts.update(val)
        fold_summaries.append(
            {
                "fold": index,
                "training_cases": len(train),
                "validation_cases": len(val),
                "training_patients": len(train_patients),
                "validation_patients": len(val_patients),
                "patient_overlap": 0,
            }
        )
    invalid_counts = {case: count for case, count in validation_counts.items() if count != 1}
    if invalid_counts or set(validation_counts) != all_cases:
        raise ValueError("Every case must occur in validation exactly once across all folds.")
    return {
        "status": "passed",
        "num_cases": len(all_cases),
        "num_patients": len(set(case_to_patient.values())),
        "num_folds": num_splits,
        "each_case_validated_once": True,
        "patient_leakage_detected": False,
        "folds": fold_summaries,
    }


def write_split_artifacts(
    output_dir: Path,
    splits: list[dict[str, list[str]]],
    case_to_patient: dict[str, str],
    patient_to_fold: dict[str, int],
    seed: int,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_path = output_dir / "splits_final.json"
    audit_path = output_dir / "patient_split_audit.json"
    assignments_path = output_dir / "patient_fold_assignments.csv"
    splits_path.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    audit = audit_splits(splits, case_to_patient, len(splits))
    audit.update({"seed": seed, "method": "deterministic patient-grouped balanced five-fold cross-validation"})
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    with assignments_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case_id", "patient_id", "validation_fold"])
        writer.writeheader()
        for case in sorted(case_to_patient):
            patient = case_to_patient[case]
            writer.writerow({"case_id": case, "patient_id": patient, "validation_fold": patient_to_fold[patient]})
    return splits_path, audit_path, assignments_path
