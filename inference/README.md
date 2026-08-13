# PerthesMetrics 2D nnU-Net Inference

Inference requires no training data. Provide a network folder containing RGB NIfTI cases, disposable local SSD scratch space, and either a local model zip or a Hugging Face model repository.

## Input layout

```text
<network-inference-dir>/
  imagesTs/
    <case-id>_0000.nii.gz   # red
    <case-id>_0001.nii.gz   # green
    <case-id>_0002.nii.gz   # blue
```

Every case must have exactly the three channels. Use `tools/prepare_rgb_nifti.py` if source radiographs are ordinary PNG, BMP, TIFF, or JPEG files.

## Environment

```bat
cd /d <repo-dir>
conda create -n perthesmetrics python=3.12 -y
conda activate perthesmetrics
python -m pip install -r inference\requirements.txt
```

## Local model

```bat
python inference\run_inference.py ^
  --data-dir <network-inference-dir> ^
  --scratch-dir <local-scratch-dir> ^
  --model-zip <perthesmetrics_nnunet_model.zip>
```

## Hosted model

```bat
python inference\run_inference.py ^
  --data-dir <network-inference-dir> ^
  --scratch-dir <local-scratch-dir> ^
  --model-repo <owner>/perthesmetrics-nnunet-model
```

Predictions and manifests are copied to `<network-inference-dir>/perthesmetrics_nnunet_results/`. The JSON manifest records the label map, model SHA-256 hash, folds, device, and model configuration. Scratch is removed after success and retained after failure for inspection. Use `--device cpu` if CUDA is unavailable; CPU inference will be slower.
