# PerthesMetrics Patient-Grouped 2D nnU-Net Training

This workflow stages a network-hosted dataset onto a local SSD, creates leakage-free patient-grouped five-fold splits, trains 2D nnU-Net v2, exports an installable model, copies the artifacts back to the network directory, and removes scratch only after success.

## Data layout

```text
<network-dataset-dir>/
  imagesTr/
    <case-id>_0000.nii.gz   # red
    <case-id>_0001.nii.gz   # green
    <case-id>_0002.nii.gz   # blue
  labelsTr/
    <case-id>.nii.gz
  imagesTs/                 # optional; not used for cross-validation
  labelsTs/                 # optional
  patient_groups.csv        # optional but recommended
```

Labels must use integer values 0 through 8 as defined in `perthesmetrics_nnunet2d.yaml`.

## Patient grouping

Cross-validation is performed at the patient level. For every fold, approximately 80% of patients are used for training and 20% for validation. All radiographs belonging to a patient—including AP/frog views, left/right sides, and longitudinal time points—remain together. Across five folds, every case is validated exactly once.

The filename parser recognizes patterns such as these synthetic examples:

- `Patient_1001_AP_baseline_L_...` -> patient `1001`
- `1001_Frog_followup_R_...` -> patient `1001`
- `X013_L_...` -> patient `X013`
- `Z001_R_...` -> patient `Z001`

For publication training, an explicit mapping is recommended:

```csv
case_id,patient_id
Patient_1001_AP_baseline_L_case01,1001
Patient_1001_Frog_followup_R_case02,1001
```

Every training case must appear once, and the map must not contain unknown cases. The workflow writes `patient_fold_assignments.csv`, `patient_split_audit.json`, and the exact nnU-Net `splits_final.json` beside the trained model.

`imagesTs` is an optional external inference set and is never inspected or used when forming cross-validation folds.

## Environment

```bat
cd /d <repo-dir>
conda create -n perthesmetrics python=3.12 -y
conda activate perthesmetrics
python -m pip install -r training\requirements.txt
```

Alternatively, create the fully specified environment with:

```bat
conda env create -f training\environment.yml
conda activate perthesmetrics
```

## Train five folds

```bat
python training\train_nnunet2d.py ^
  --config training\perthesmetrics_nnunet2d.yaml ^
  --data-dir <network-dataset-dir> ^
  --scratch-dir <local-scratch-dir> ^
  --folds 0 1 2 3 4 ^
  --epochs 1000 ^
  --device cuda
```

To select a GPU, add `--gpu 0`. Supported nnU-Net trainer lengths are 1, 5, 10, 20, 50, 100, 250, 500, 750, 1000, 2000, 4000, and 8000 epochs.

Successful output is copied to:

```text
<network-dataset-dir>/perthesmetrics_nnunet_model/
  nnUNet_results/
  export/perthesmetrics_nnunet_model.zip
  patient_fold_assignments.csv
  patient_split_audit.json
  splits_final.json
  training_manifest.json
```

The copied model folder also contains a `reports/` directory with per-fold epoch CSV files, training curves, and a cross-fold training summary.

## Resume

If training is interrupted, repeat the same command with `--resume` and the same scratch path, folds, epoch count, configuration, and patient map. Failed or interrupted runs preserve scratch. Successful runs delete it unless `--keep-scratch` is specified.

## Fast smoke test

Before a full run, use a deidentified miniature dataset and `--folds 0 --epochs 1`. This verifies staging, preprocessing, patient splits, training, export, and copy-back without committing generated artifacts.
