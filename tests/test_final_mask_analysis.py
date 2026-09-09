from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.final_mask_analysis import (
    CLASS_ORDER,
    collect_image_mask_dice,
    missing_affected_stage_rows,
    plot_stage_view_boxplots,
    pool_by_patient,
    read_radiograph_metadata,
    read_stage_maps,
    summarize_patient_rows,
    unknown_affected_status_rows,
)


def write_mask(path: Path, active_pixels: list[tuple[int, int]]) -> None:
    array = np.zeros((4, 4), dtype=np.uint8)
    for row, column in active_pixels:
        array[row, column] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


class FinalMaskAnalysisTests(unittest.TestCase):
    def test_patient_pooling_combines_multiple_images_before_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            gt_dir = data_dir / "masks"
            pred_dir = data_dir / "nnunet_masks"
            data_dir.mkdir()

            filenames = [
                "Patient_1003_AP_1.bmp",
                "Patient_1003_AP_2.bmp",
                "Patient_2000_AP_1.bmp",
                "Patient_3000_AP_1.bmp",
                "Patient_4000_AP_1.bmp",
            ]
            for filename in filenames:
                Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(data_dir / filename)
                for mask_name in CLASS_ORDER:
                    gt_pixels = [(0, 0), (0, 1)] if mask_name == "head" else [(0, 0)]
                    pred_pixels = gt_pixels if filename != "Patient_1003_AP_2.bmp" else [(0, 0)]
                    write_mask(gt_dir / mask_name / filename, gt_pixels)
                    write_mask(pred_dir / mask_name / filename, pred_pixels)

            (data_dir / "ap_classes.csv").write_text(
                "filename,class\n"
                "Patient_1003_AP_1.bmp,0\n"
                "Patient_1003_AP_2.bmp,0\n"
                "Patient_2000_AP_1.bmp,1\n",
                encoding="utf-8",
            )
            (data_dir / "frog_classes.csv").write_text("filename,class\n", encoding="utf-8")
            (root / "radiographdata.csv").write_text(
                "file_name,Source,affected_vs_unaffected,ap_or_frog\n"
                "Patient_1003_AP_1.bmp,test,Affected,AP\n"
                "Patient_1003_AP_2.bmp,test,Affected,AP\n"
                "Patient_2000_AP_1.bmp,test,Unaffected,AP\n"
                "Patient_3000_AP_1.bmp,test,NA,AP\n"
                "Patient_4000_AP_1.bmp,test,Affected,AP\n",
                encoding="utf-8",
            )

            stage_map = read_stage_maps(data_dir / "ap_classes.csv", data_dir / "frog_classes.csv")
            radiograph_metadata = read_radiograph_metadata(root / "radiographdata.csv")
            image_rows = collect_image_mask_dice(
                data_dir,
                gt_dir,
                pred_dir,
                stage_map,
                radiograph_metadata,
                r"Patient[_ -]*(?P<patient_id>\d+)",
                False,
            )
            patient_rows = pool_by_patient(image_rows, "overall", ())
            head_1003 = next(row for row in patient_rows if row.patient_id == "1003" and row.mask == "head")

            self.assertEqual(head_1003.n_images, 2)
            self.assertAlmostEqual(head_1003.dice, 6 / 7)

            summary = summarize_patient_rows(patient_rows, "overall", (), [0.90, 0.80], 100, 123)
            head_summary = next(row for row in summary if row["mask"] == "head")
            self.assertEqual(head_summary["n_patients"], 4)
            self.assertAlmostEqual(head_summary["pass_rate_ge_0_90"], 3 / 4)
            self.assertAlmostEqual(head_summary["pass_rate_ge_0_80"], 1.0)
            exclusions = unknown_affected_status_rows(image_rows)
            self.assertEqual([row["filename"] for row in exclusions], ["Patient_3000_AP_1.bmp"])
            missing_stage = missing_affected_stage_rows(image_rows)
            self.assertEqual([row["filename"] for row in missing_stage], ["Patient_4000_AP_1.bmp"])

            known_rows = [row for row in image_rows if row.analysis_group in {"Unaffected", "Ia"}]
            analysis_group_view_rows = pool_by_patient(
                known_rows,
                "analysis_group_view",
                ("analysis_group", "view"),
            )
            groups = {row.analysis_group for row in analysis_group_view_rows}
            self.assertEqual(groups, {"Ia", "Unaffected"})
            figure_path = root / "stage_view_boxplot.png"
            plot_stage_view_boxplots(figure_path, analysis_group_view_rows, 0.90, 123)
            self.assertTrue(figure_path.is_file())
            self.assertTrue(figure_path.with_suffix(".pdf").is_file())


if __name__ == "__main__":
    unittest.main()
