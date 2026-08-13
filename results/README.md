# PerthesMetrics Results

This directory is reserved for sanitized publication-style aggregate results reconstructed from the final patient-grouped model archive.

Run:

```bat
python tools\build_publication_results.py ^
  --model-zip <perthesmetrics_nnunet_model.zip>
```

Generated outputs belong under `results/segmentation/nnunet2d/`. Model weights, patient data, case identifiers, case-level predictions, and full nnU-Net validation summaries are intentionally excluded. Historical image-level cross-validation results must not be presented as the final patient-grouped paper evaluation.
