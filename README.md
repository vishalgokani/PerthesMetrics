# PerthesMetrics

Reproducible 2D nnU-Net v2 training, inference, model-release, and publication-results tools for multiclass segmentation of pediatric hip radiographs in Legg-Calve-Perthes disease.

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

From an Anaconda Prompt or command prompt:

```bat
cd /d <repo-dir>
conda create -n perthesmetrics python=3.12 -y
conda activate perthesmetrics
python -m pip install -r training\requirements.txt
python -m pip install -r inference\requirements.txt
```

The tested reference environment is Python 3.12, PyTorch 2.5.1 with CUDA 12.4, and nnU-Net v2.5.1. Install the PyTorch build appropriate for your machine before running GPU workloads if `pip` does not select it correctly.

## Train

The training dataset must contain `imagesTr` and `labelsTr`. `imagesTs` and `labelsTs` are optional. Each RGB case is stored as three NIfTI files ending in `_0000`, `_0001`, and `_0002`.

```bat
python training\train_nnunet2d.py ^
  --data-dir <network-dataset-dir> ^
  --scratch-dir <local-scratch-dir> ^
  --folds 0 1 2 3 4 ^
  --epochs 1000 ^
  --device cuda
```

Five-fold splits are made by patient, not by radiograph. All views, sides, and time points belonging to one patient remain in the same validation fold. See [training/README.md](training/README.md).

## Run inference

```bat
python inference\run_inference.py ^
  --data-dir <network-inference-dir> ^
  --scratch-dir <local-scratch-dir> ^
  --model-zip <perthesmetrics_nnunet_model.zip>
```

A Hugging Face model repository can be supplied with `--model-repo` instead. See [inference/README.md](inference/README.md).

## Model release and results

Use [model_release/](model_release/README.md) to validate and package an exported model. Use `tools/build_publication_results.py` to create sanitized aggregate results without committing images, model weights, or case-level records.

## Intended use

This software and associated models are for research use. They are not medical devices and must not be used as the sole basis for diagnosis or treatment.
