# Final mask analysis

This folder documents the held-out test-set mask analysis script. Do not commit
patient-level outputs, mask overlays, raw data, or generated figures from the
test dataset.

The script compares ground-truth binary masks in `masks/` with inference masks
in `nnunet_masks/`, pools Dice by patient ID parsed from filenames such as
`Patient_1003_...bmp`, then bootstraps patients with replacement for 95%
confidence intervals. Affected versus unaffected hips are read from
`radiographdata.csv`; the default location is two folders above the test set,
at `<ALL_RADIOGRAPHS_DIR>/radiographdata.csv`.

For quantitative Dice analysis, femoral neck and femoral shaft are combined into
one `Femoral Neck/Shaft` mask per image before image pixels are pooled to the
patient level. The overlay figures still show `Femoral Neck` and `Femoral
Shaft` as separate colored masks.

Rows where `affected_vs_unaffected` is `NA`, blank, or otherwise unknown are
kept in the overall and AP/frog-only performance tables, but are excluded from
affected/unaffected and Waldenstrom-stage stratified outputs because they cannot
be assigned to a hip-status row. Those exclusions are written to
`unknown_affected_status_exclusions.csv`.

## Run

From the repository root:

```bat
conda activate perthesmetrics
python tools\final_mask_analysis.py ^
  --data-dir "F:\projects\perthesmetrics\lcpd_radiograph_data_for_perthesmetrics\final_dataset\all_radiographs\test\test"
```

The default output directory is:

```text
<DATA_DIR>\final_mask_analysis_outputs
```

To keep results somewhere else:

```bat
python tools\final_mask_analysis.py ^
  --data-dir "<DATA_DIR>" ^
  --output-dir "<OUTPUT_DIR>"
```

## Main outputs

- `overall_model_performance_by_mask.csv`: overall patient-pooled Dice and 95%
  CI for each mask, plus patient pass-rate CIs for the Dice cutoff 0.80.
- `overall_model_performance_publication_table.{csv,md,tex,pdf,svg}`:
  patient-bootstrapped Dice, IoU, precision, and recall with 95% CIs and patient
  counts. The PDF/SVG versions use Times New Roman.
- `waldenstrom_stage_model_performance_publication_table.{csv,md,tex,pdf,svg}`:
  patient-bootstrapped Dice with 95% CIs and patient counts for unaffected hips
  and every Waldenstrom stage (Ia, Ib, IIa, IIb, IIIa, IIIb, and IV).
- `view_model_performance_publication_table.{csv,md,tex,pdf,svg}`:
  patient-bootstrapped Dice with 95% CIs and patient counts for AP and frog-leg
  views.
- `view_model_performance_by_mask.csv`: same analysis split into AP and frog
  views.
- `affected_status_model_performance_by_mask.csv`: patient-bootstrapped
  affected versus unaffected performance.
- `analysis_group_model_performance_by_mask.csv`: unaffected hips plus affected
  hips split by Waldenstrom stage.
- `analysis_group_view_model_performance_by_mask.csv`: unaffected hips plus
  affected Waldenstrom stages, separately for AP and frog views.
- `unknown_affected_status_exclusions.csv`: radiographs with unknown
  affected/unaffected status that were excluded from stratified outputs.
- `missing_affected_waldenstrom_stage_exclusions.csv`: affected radiographs
  excluded from Waldenstrom-stage rows because no usable stage was present in
  the class CSVs or metadata.
- `waldenstrom_stage_model_performance_by_mask.csv`: Waldenstrom-stage
  descriptive performance for affected hips only, across views.
- `waldenstrom_stage_view_model_performance_by_mask.csv`: stage-by-view
  performance for affected hips only, for AP and frog.
- `patient_pooled_dice_*.csv`: patient-level pooled Dice values used for the
  bootstraps.
- `analysis_group_view_boxplots.{png,pdf,svg}`: AP/Frog boxplot figure with no
  cutoff reference line.
- `analysis_group_view_boxplots_cutoff_0_80.{png,pdf,svg}`: the same boxplots
  with a dashed Dice cutoff line. In both versions, the first row is unaffected
  hips, followed by affected hips split into Waldenstrom stages. AP panels show
  all seven analysis structures; frog-leg panels omit the lesser and greater
  trochanters because those structures are not reported for the lateral view.
- `mask_figures_stratified/`: optional overlay copies organized as
  `<view>/<waldenstrom_class>/...`, using `ap_classes.csv` and
  `frog_classes.csv`.
- `mask_colors.csv`: RGB and hex colors used consistently for overlays and
  boxplot dots.

## Notes

Waldenstrom classes are read from `ap_classes.csv` and `frog_classes.csv` as
numeric classes `0` through `6`, mapped to `Ia`, `Ib`, `IIa`, `IIb`, `IIIa`,
`IIIb`, and `IV`. Unaffected hips are not placed into Waldenstrom-stage rows;
they are pooled into the separate `Unaffected` analysis row.

Affected hips with no usable stage in `ap_classes.csv`, `frog_classes.csv`, or
`radiographdata.csv` remain in full-dataset, AP/frog, and affected-status
performance tables, but are excluded from Waldenstrom-stage rows.

By default, the script fails if ground-truth and inference mask shapes differ.
Use `--allow-resize-for-metrics` only after confirming nearest-neighbor resizing
is scientifically appropriate for the final analysis.

If overlay figures already exist and only need to be reorganized, run:

```bat
python tools\stratify_mask_figures.py ^
  --data-dir "F:\projects\perthesmetrics\lcpd_radiograph_data_for_perthesmetrics\final_dataset\all_radiographs\test\test"
```

The quantitative analysis never creates or overwrites mask overlays. If new
overlays are deliberately needed, use the separate script:

```bat
python tools\create_mask_overlays.py ^
  --data-dir "<DATA_DIR>" ^
  --stratify
```

To re-render only the Waldenstrom publication table and both boxplot
variants from their existing CSV files, without reading any image or mask, run:

```bat
python tools\final_mask_analysis.py ^
  --render-only ^
  --output-dir "<OUTPUT_DIR>"
```

This overwrites only the PDF, SVG, and PNG renderings of those outputs. It does
not recalculate metrics or change the saved CSV, Markdown, or TeX tables.

To re-render only the no-cutoff and 0.80-cutoff boxplots from the existing
patient-level CSV, use:

```bat
python tools\final_mask_analysis.py ^
  --render-boxplots-only ^
  --output-dir "<OUTPUT_DIR>"
```

To print only the global median across all finite patient-mask Dice values and
its patient-clustered bootstrap 95% CI, without recalculating or writing any
other output, use:

```bat
python tools\final_mask_analysis.py ^
  --global-median-only ^
  --output-dir "<OUTPUT_DIR>"
```

The point estimate pools the patient-level Dice observations for all seven
analysis masks. The confidence interval resamples patients with replacement so
that mask observations from the same patient remain clustered.
