from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from training.patient_splits import (
    audit_splits,
    infer_patient_id,
    make_grouped_splits,
    resolve_groups,
    write_split_artifacts,
)


class PatientSplitTests(unittest.TestCase):
    def test_supported_patient_names_are_normalized(self) -> None:
        self.assertEqual(infer_patient_id("Patient_1001_AP_baseline_L_case01"), "1001")
        self.assertEqual(infer_patient_id("1001_Frog_followup_R_case02"), "1001")
        self.assertEqual(infer_patient_id("X013_L_case03"), "X013")
        self.assertEqual(infer_patient_id("Z001_R_case04"), "Z001")

    def test_grouped_folds_have_no_patient_leakage(self) -> None:
        mapping = {f"Patient_{patient}_view_{view}": str(patient) for patient in range(1000, 1012) for view in range(patient % 3 + 1)}
        splits, assignments = make_grouped_splits(mapping, num_splits=5, seed=7)
        audit = audit_splits(splits, mapping, 5)
        self.assertFalse(audit["patient_leakage_detected"])
        self.assertEqual(set(assignments), set(mapping.values()))
        for split in splits:
            train_patients = {mapping[case] for case in split["train"]}
            val_patients = {mapping[case] for case in split["val"]}
            self.assertFalse(train_patients & val_patients)

    def test_split_artifacts_are_written(self) -> None:
        mapping = {f"{patient}_view": str(patient) for patient in range(1000, 1010)}
        splits, assignments = make_grouped_splits(mapping, 5, 2026)
        with tempfile.TemporaryDirectory() as temporary:
            paths = write_split_artifacts(Path(temporary), splits, mapping, assignments, 2026)
            self.assertTrue(all(path.is_file() for path in paths))

    def test_explicit_patient_map_must_be_complete(self) -> None:
        with self.assertRaisesRegex(ValueError, "incomplete"):
            resolve_groups(["1_AP", "2_AP"], {"1_AP": "1"})


if __name__ == "__main__":
    unittest.main()
