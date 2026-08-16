# Inference

Run commands from the repository root:

```bat
cd /d <github_repo_location>
```

```bat
conda env create -f inference\environment.yml
conda activate perthesmetrics
python inference\run_inference.py ^
  --data-dir <data_dir> ^
  --scratch-dir <scratch_dir> ^
  --model-zip <model_zip> ^
  --device cuda
```

Root BMP radiographs are converted to three-channel NIfTI only in local scratch. Predictions are converted back into eight class directories of binary BMP files under `<data_dir>/nnunet_masks` by default, using the original filenames. The output directory must not already exist. Scratch is deleted after success. Add `--gpu 0` to select a GPU, `--output-dir` to select another destination, or `--device cpu` for CPU inference.

Figure recoloring is separate from inference. Edit `OPACITY` and `COLORS` in `generate_figures.py`, or pass `--opacity`:

```bat
python inference\generate_figures.py ^
  --source-dir <data_dir> ^
  --predictions-dir <predictions_dir> ^
  --output-dir <figures_dir> ^
  --opacity 0.45
```

Run inference with `--keep-scratch` when NIfTI predictions are needed for figures. The figure command is not part of inference.
