# Tools

Run commands from the repository root. This directory contains evaluation,
publication-output, and data-preparation utilities that are not part of the
primary training or inference entry points.

Patient-level data and generated outputs must remain outside the repository.
Use placeholders such as `<DATA_DIR>`, `<OUTPUT_DIR>`, and `<FIGURE_DIR>` when
recording local commands.

## Command index

| Script | Purpose | Main inputs | Main outputs |
|---|---|---|---|
| `final_mask_analysis.py` | Held-out mask evaluation with patient-pooled metrics and patient-bootstrap confidence intervals | Ground-truth masks, inference masks, stage CSVs, radiograph metadata | Aggregate and patient-level CSVs, tables, and boxplots |
| `create_mask_overlays.py` | Create optional ground-truth and prediction overlays separately from quantitative evaluation | Radiographs and masks | Overlay images |
| `stratify_mask_figures.py` | Organize existing overlays by view and Waldenstrom stage | Overlay images and stage CSVs | Stratified overlay directories |
| `build_figure1_methods.py` | Build publication Figure 1 | Prepared AP and frog-leg panels | PDF, SVG, PNG, and TIFF |
| `build_figure4_stage_performance.py` | Build publication Figure 4 without reading patient-level records | Aggregate evaluation CSVs | PDF, SVG, PNG, and TIFF |
| `waldenstrom_figure_builder.py` | Assemble the Waldenstrom staging figure | Prepared panels under stage folders | PDF, SVG, PNG, and TIFF |
| `build_cohort_demographics.py` | Build the cohort demographics table | Deidentified radiograph, staging, and split CSVs | CSV, SVG, and PDF |
| `generate_nnunet_training_supplement.py` | Build supplemental training and validation figures | nnU-Net model logs and validation summaries | Five PNG/PDF figures, captions, and source CSVs |
| `build_publication_results.py` | Extract lightweight aggregate cross-validation results | Exported model ZIP | Aggregate tables and validation summary figure |
| `cropper.py` | Apply one reproducible square crop to multiple same-sized panels | Up to ten images | Cropped images and crop metadata |
| `prepare_rgb_nifti.py` | Convert RGB images into three nnU-Net NIfTI channels | Image directory | nnU-Net image channels and manifest |

Use `python <script> --help` for the complete arguments of any command.

## Held-out mask evaluation

The evaluation compares ground-truth binary masks under `masks/` with inference
masks under `nnunet_masks/`. It pools pixels by patient, combines femoral neck
and shaft for quantitative analysis, and bootstraps patients with replacement
for 95% confidence intervals. All views, sides, and time points for a patient
remain in the same resampling unit.

```bat
python tools\final_mask_analysis.py ^
  --data-dir "<DATA_DIR>" ^
  --output-dir "<OUTPUT_DIR>"
```

The input directory normally contains root-level radiographs, `masks/`,
`nnunet_masks/`, `ap_classes.csv`, and `frog_classes.csv`. Pass
`--radiographdata-csv <METADATA_CSV>` when the radiograph metadata is not at the
documented default relative location.

Important aggregate outputs include:

- `overall_model_performance_by_mask.csv`
- `affected_status_model_performance_by_mask.csv`
- `analysis_group_model_performance_by_mask.csv`
- `analysis_group_view_model_performance_by_mask.csv`
- `view_model_performance_by_mask.csv`
- `waldenstrom_stage_model_performance_by_mask.csv`
- `waldenstrom_stage_view_model_performance_by_mask.csv`
- `overall_model_performance_publication_table.*`
- `view_model_performance_publication_table.*`
- `waldenstrom_stage_model_performance_publication_table.*`
- `analysis_group_view_boxplots.*`

Patient-level `patient_pooled_dice_*.csv` files are generated to support the
bootstrap analysis but must not be committed. Records with unknown hip status
or missing affected-side stage remain in applicable overall analyses and are
listed in explicit exclusion CSVs for analyses that require those fields.

By default, evaluation fails when ground-truth and prediction dimensions differ.
Use `--allow-resize-for-metrics` only after confirming that nearest-neighbor
resizing is scientifically appropriate.

Existing saved outputs can be rendered again without reading the original
masks:

```bat
python tools\final_mask_analysis.py --render-only --output-dir "<OUTPUT_DIR>"
python tools\final_mask_analysis.py --render-boxplots-only --output-dir "<OUTPUT_DIR>"
python tools\final_mask_analysis.py --global-median-only --output-dir "<OUTPUT_DIR>"
```

## Publication figures

Build Figure 1 from four prepared square panels:

```bat
python tools\build_figure1_methods.py ^
  --ap-original "<AP_IMAGE>" ^
  --ap-ground-truth "<AP_MASK_PANEL>" ^
  --frog-original "<FROG_IMAGE>" ^
  --frog-ground-truth "<FROG_MASK_PANEL>" ^
  --output-dir "<FIGURE_DIR>"
```

Build Figure 4 from aggregate outputs produced by `final_mask_analysis.py`:

```bat
python tools\build_figure4_stage_performance.py ^
  --analysis-dir "<OUTPUT_DIR>" ^
  --output-dir "<FIGURE_DIR>"
```

This reads `overall_model_performance_by_mask.csv` and
`analysis_group_model_performance_by_mask.csv`; it does not read patient-level
records.

The Waldenstrom builder discovers prepared original, ground-truth, and nnU-Net
panels by filename under stage folders `1a`, `1b`, `2a`, `2b`, `3a`, `3b`, and
`4`:

```bat
python tools\waldenstrom_figure_builder.py ^
  --input-dir "<STAGE_PANEL_DIR>" ^
  --output-prefix "<FIGURE_DIR>\waldenstrom_staging"
```

Use `cropper.py` when the source panels require one consistent interactive
square crop. Supplemental training figures are documented in
[`training/README.md`](../training/README.md).

## Optional overlays

Quantitative evaluation does not create or overwrite mask overlays. Generate
them explicitly when needed:

```bat
python tools\create_mask_overlays.py --data-dir "<DATA_DIR>" --stratify
```

To reorganize overlays that already exist:

```bat
python tools\stratify_mask_figures.py --data-dir "<DATA_DIR>"
```
