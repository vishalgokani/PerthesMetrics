# Inference

```bat
conda env create -f inference\environment.yml
conda activate perthesmetrics
python inference\run_inference.py ^
  --data-dir E:\perthesmetrics\nnUNet_raw\Dataset ^
  --scratch-dir C:\perthesmetrics_scratch\inference ^
  --model-zip E:\perthesmetrics\export\Dataset_nnUNetTrainer_nnUNetPlans_2d.zip ^
  --output-dir C:\Users\gokan\Documents\GitHub\PerthesMetrics\results\inference\test ^
  --device cuda
```

Input is read-only. Each `imagesTs` case must have `_0000`, `_0001`, and `_0002` NIfTI channels. Successful inference writes predictions and CSV/JSON manifests to `--output-dir` and deletes scratch. Add `--gpu 0` to select a GPU or use `--device cpu`.

Figure recoloring is separate from inference. Edit `OPACITY` and `COLORS` in `generate_figures.py`, or pass `--opacity`:

```bat
python inference\generate_figures.py ^
  --source-dir E:\perthesmetrics\lcpd_radiograph_data_for_perthesmetrics\final_dataset\all_radiographs\test\test ^
  --predictions-dir C:\Users\gokan\Documents\GitHub\PerthesMetrics\results\inference\test\predictions ^
  --output-dir E:\perthesmetrics\lcpd_radiograph_data_for_perthesmetrics\final_dataset\all_radiographs\test\test\figures ^
  --opacity 0.45
```

The figure command writes to E: only when explicitly run; it is not part of inference.
