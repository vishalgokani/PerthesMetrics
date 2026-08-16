# Results

Run commands from the repository root:

```bat
cd /d <github_repo_location>
```

`training/nnunet2d/reports/` reproduces the repository training-metrics output from existing model logs. `segmentation/nnunet2d/` contains aggregate validation tables and 300-dpi PNG/PDF figures. No images, weights, case identifiers, or case-level metrics are included.

```bat
python training\analyze_training_metrics.py --model-dir <model_dir> --output-dir results\training\nnunet2d\reports
python tools\build_publication_results.py --model-zip <model_zip>
```

Confidence intervals are t-based intervals over five fold estimates; they are not patient-level intervals. The external test set remains unevaluated until predictions are compared with an independent reference standard.
