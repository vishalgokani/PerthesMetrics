from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.final_mask_analysis import (
    CLASS_ORDER,
    build_parser,
    collect_image_mask_dice,
    missing_affected_stage_rows,
    plot_mask_order,
    plot_stage_view_boxplots,
    pool_by_patient,
    read_radiograph_metadata,
    read_stage_maps,
    summarize_overall_metrics,
    summarize_global_patient_mask_dice,
    summarize_patient_rows,
    unknown_affected_status_rows,
    write_publication_table_figure,
)


def write_mask(path: Path, active_pixels: list[tuple[int, int]]) -> None:
    array = np.zeros((4, 4), dtype=np.uint8)
    for row, column in active_pixels:
        array[row, column] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


class FinalMaskAnalysisTests(unittest.TestCase):
    def test_plot_masks_are_view_specific(self) -> None:
        self.assertEqual(plot_mask_order("ap")[-2:], ["lt", "gt"])
        self.assertNotIn("lt", plot_mask_order("frog"))
        self.assertNotIn("gt", plot_mask_order("frog"))
        self.assertEqual(plot_mask_order("frog")[-1], "triradiate cartilage")

    def test_publication_default_is_only_the_080_cutoff(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.cutoffs, [0.80])

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
            metric_summary = summarize_overall_metrics(patient_rows, 100, 123)
            head_metrics = next(row for row in metric_summary if row["mask"] == "head")
            self.assertEqual(head_metrics["n_patients"], 4)
            self.assertAlmostEqual(head_metrics["mean_patient_pooled_dice"], (6 / 7 + 3) / 4)
            self.assertAlmostEqual(head_metrics["mean_patient_pooled_iou"], (3 / 4 + 3) / 4)
            self.assertAlmostEqual(head_metrics["mean_patient_pooled_precision"], 1.0)
            self.assertAlmostEqual(head_metrics["mean_patient_pooled_recall"], (3 / 4 + 3) / 4)
            global_summary = summarize_global_patient_mask_dice(patient_rows, 100, 123)
            self.assertEqual(global_summary["n_patients"], 4)
            self.assertEqual(global_summary["n_patient_mask_observations"], 28)
            self.assertEqual(global_summary["median_patient_pooled_dice"], 1.0)
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
            plot_stage_view_boxplots(figure_path, analysis_group_view_rows, 123)
            self.assertTrue(figure_path.is_file())
            self.assertTrue(figure_path.with_suffix(".pdf").is_file())
            self.assertTrue(figure_path.with_suffix(".svg").is_file())
            cutoff_path = root / "stage_view_boxplot_cutoff_0_90.png"
            plot_stage_view_boxplots(cutoff_path, analysis_group_view_rows, 123, cutoff=0.90)
            self.assertTrue(cutoff_path.is_file())
            self.assertTrue(cutoff_path.with_suffix(".pdf").is_file())
            self.assertTrue(cutoff_path.with_suffix(".svg").is_file())

            table_path = root / "publication_table"
            write_publication_table_figure(
                table_path,
                [{"Hip Structure": "Femoral Head", "n": 4, "Dice": "0.96 [0.93-0.98]"}],
                "Test Publication Table",
            )
            self.assertTrue(table_path.with_suffix(".pdf").is_file())
            self.assertTrue(table_path.with_suffix(".svg").is_file())


if __name__ == "__main__":
    unittest.main()
