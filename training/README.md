# Training

Run commands from the repository root:

```bat
cd /d <github_repo_location>
```

The input directory contains radiograph BMPs at its root and eight matching binary-mask directories under `masks/`. During local staging, every radiograph is converted into RGB `_0000`, `_0001`, and `_0002` NIfTI channels and the class masks are merged into one multiclass `labelsTr` NIfTI file.

```bat
conda env create -f training\environment.yml
conda activate perthesmetrics
python training\train_nnunet2d.py ^
  --data-dir <data_dir> ^
  --scratch-dir <scratch_dir> ^
  --folds 0 1 2 3 4 ^
  --epochs 1000 ^
  --device cuda
```

Patient-grouped folds are inferred from the required `Patient_<numeric_id>...` filename prefix. All views, sides, and time points sharing that ID remain in one fold. An explicit complete `patient_groups.csv` is also supported.

After training, the complete package is copied to `<data_dir>/perthesmetrics_nnunet_model/`, including `export/perthesmetrics_nnunet_model.zip`, split audits, reports, and model files. No training masks are copied back. Scratch is deleted after success; use `--resume` with the same path after interruption.
