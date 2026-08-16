# Training

The input folder must contain `imagesTr/<case>_0000.nii.gz`, `_0001`, `_0002`, and `labelsTr/<case>.nii.gz`. Provide `patient_groups.csv` (`case_id,patient_id`) so every radiograph from one patient stays in one fold.

```bat
conda env create -f training\environment.yml
conda activate perthesmetrics
python training\train_nnunet2d.py ^
  --data-dir <dataset-dir> ^
  --scratch-dir C:\perthesmetrics_scratch\training ^
  --folds 0 1 2 3 4 ^
  --epochs 1000 ^
  --device cuda
```

The workflow validates and stages data, creates patient-grouped folds, trains nnU-Net 2D, creates per-fold epoch CSVs/curves and a cross-fold summary, exports the model, copies outputs to `<dataset-dir>/perthesmetrics_nnunet_model`, and deletes scratch after success. Use `--resume` with the same scratch path after interruption.

Rebuild only the reports from existing logs:

```bat
python training\analyze_training_metrics.py ^
  --model-dir E:\perthesmetrics\nnUNet_results\Dataset\nnUNetTrainer__nnUNetPlans__2d ^
  --output-dir C:\Users\gokan\Documents\GitHub\PerthesMetrics\results\training\nnunet2d\reports
```
