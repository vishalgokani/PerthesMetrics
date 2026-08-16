# PerthesMetrics Model Release

Run commands from the repository root:

```bat
cd /d <github_repo_location>
```

This folder contains release infrastructure, not model weights. The intended hosted filename is:

```text
perthesmetrics_nnunet_model.zip
```

Prepare an nnU-Net export for Hugging Face or another artifact host:

```bat
python model_release\prepare_model_package.py ^
  --source-zip <exported_model_zip> ^
  --output-zip <release_dir>\perthesmetrics_nnunet_model.zip
```

The command creates an installable archive and adjacent `.sha256` checksum. It verifies the canonical `Dataset001_PerthesMetrics` identity, `2d` configuration, RGB channel map, eight-label map, and five final checkpoints. It can also normalize a legacy archive with a noncanonical dataset root or inconsistent dataset metadata.

Before publishing, test the package in an empty scratch directory with `nnUNetv2_install_pretrained_model_from_zip` and a representative deidentified inference case. Do not upload patient data, case lists, or unredacted validation records.
