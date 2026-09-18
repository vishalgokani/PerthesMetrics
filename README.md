# PerthesMetrics

Reproducible 2D nnU-Net v2 training, inference, model-release, and publication-results tools for multiclass segmentation of pediatric hip radiographs in Legg-Calvé-Perthes disease.

Run all commands from the repository root:

```bat
cd /d <github_repo_location>
```

The model uses RGB radiographs represented as three nnU-Net channels and predicts eight anatomic regions:

| Value | Region |
|---:|---|
| 0 | background |
| 1 | acetabulum |
| 2 | greater trochanter (`gt`) |
| 3 | femoral head |
| 4 | lesser trochanter (`lt`) |
| 5 | femoral neck |
| 6 | femoral shaft |
| 7 | sourcil |
| 8 | triradiate cartilage |

```text
training/       patient-grouped five-fold training and model export
inference/      inference from a local or hosted nnU-Net model zip
model_release/  validate and prepare installable release artifacts
results/        lightweight, aggregate publication results
tools/          dataset conversion and results utilities
tests/          fast unit and archive-validation tests
```

No patient data or model weights are committed. Keep source data on a network drive and use a disposable local SSD scratch directory for preprocessing, training, and inference.

## Environment

From an Anaconda Prompt, create the pinned CUDA environment:

```bat
conda env create -f environment.yml
conda activate perthesmetrics
```

The reference environment uses Python 3.12, PyTorch 2.5.1 with CUDA 12.4, and nnU-Net v2.5.1. The training script checks CUDA support before touching scratch data and reports an actionable error if a CPU-only PyTorch build is active.

If installing with `pip` instead, install the CUDA-compatible PyTorch build for your system from the [PyTorch installation guide](https://pytorch.org/get-started/locally/) **before** running `pip install -r requirements.txt`. PyTorch is intentionally not listed in `requirements.txt` because its correct package depends on the user's CUDA setup.

## Train

Training accepts root BMP radiographs plus one matching BMP per class under `masks/<class>/`. They are converted in local scratch to three RGB NIfTI channels and one multiclass label per case.

```bat
python training\train_nnunet2d.py ^
  --data-dir <data_dir> ^
  --scratch-dir <scratch_dir> ^
  --folds 0 1 2 3 4 ^
  --epochs 1000 ^
  --device cuda
```

Five-fold splits are made by patient, not by radiograph. All views, sides, and time points belonging to one patient remain in the same validation fold. See [training/README.md](training/README.md).

## Inference

Inference commands, local-model and Hugging Face workflows, batch examples, inputs, and outputs are documented only in [inference/README.md](inference/README.md). The released model is hosted at [vishalgokani/perthesmetrics-nnunet](https://huggingface.co/vishalgokani/perthesmetrics-nnunet).


## Model release and results

Use [model_release/](model_release/README.md) to validate and package an exported model. Use `tools/build_publication_results.py` to create sanitized aggregate results without committing images, model weights, or case-level records.

For held-out test-set mask analysis, use `tools/final_mask_analysis.py`. It
creates patient-pooled Dice tables, patient-bootstrap 95% confidence intervals,
Waldenstrom-stage AP/frog boxplots, and optional mask overlay figures. See
[testing/final_mask_analysis/README.md](testing/final_mask_analysis/README.md).

To prepare publication panels, use `tools/cropper.py` to apply one interactive
square crop to as many as ten same-sized images. The Waldenstrom figure builder
discovers original, ground-truth, and nnU-Net panels by filename under stage
folders `1a`, `1b`, `2a`, `2b`, `3a`, `3b`, and `4`. It writes a self-contained
SVG, 600-DPI PNG, and PDF; run `tools/waldenstrom_figure_builder.py --help` for
the CLI.

Build Figure 4 from the aggregate patient-bootstrap CSV outputs without reading
patient-level data:

```bat
python tools\build_figure4_stage_performance.py ^
  --analysis-dir "<OUTPUT_DIR>" ^
  --output-dir "<FIGURE_DIR>"
```

This writes `Figure4.pdf`, `Figure4.svg`, and a 600-DPI `Figure4.png`.

## Intended use

This software and associated models are for research use. They are not medical devices and must not be used as the sole basis for diagnosis or treatment.
