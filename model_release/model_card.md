# PerthesMetrics 2D nnU-Net Segmentation Model

## Summary

PerthesMetrics is a 2D nnU-Net v2 model for multiclass anatomic segmentation of pediatric hip radiographs in Legg-Calvé-Perthes disease. The canonical nnU-Net dataset identity is `Dataset001_PerthesMetrics`.

## Intended use

The model is intended for research use on AP and frog-leg lateral pediatric hip radiographs. It is not a medical device and must not be used as the sole basis for diagnosis, treatment, or other clinical decisions.

## Model and software

The release uses the standard `nnUNetTrainer`, `nnUNetPlans`, and `2d` configuration with five 1,000-epoch folds. The reference environment uses Python 3.12, PyTorch 2.5.1 with CUDA 12.4, and nnU-Net v2.5.1.

## Inputs and preprocessing

The repository inference command accepts root-level BMP, PNG, JPEG, and TIFF radiographs. Each image is converted to three RGB NIfTI channels before nnU-Net inference. Users running nnU-Net directly must provide the equivalent `_0000`, `_0001`, and `_0002` NIfTI channels.

## Outputs

The integer segmentation contains background (0), acetabulum (1), greater trochanter (2), femoral head (3), lesser trochanter (4), femoral neck (5), femoral shaft (6), sourcil (7), and triradiate cartilage (8). The repository inference command exports one binary mask per foreground class using the source image filename.

## Training and evaluation

The model was developed from deidentified pediatric hip radiographs. Internal development used patient-grouped five-fold cross-validation so that all views, sides, and time points from a patient remained in the same fold. Further cohort, annotation, and evaluation details are provided in the accompanying publication.

## Limitations

Performance may not generalize across institutions, acquisition systems, populations, image-processing pipelines, or radiographs outside the training distribution. Small or poorly visualized structures may have lower accuracy. Users should independently validate performance for their setting.

## Availability

The pretrained model is available from [vishalgokani/perthesmetrics-nnunet](https://huggingface.co/vishalgokani/perthesmetrics-nnunet). It can be downloaded automatically by `inference/run_inference.py` or supplied locally with `--model-zip`. The hosted ZIP is also installable with `nnUNetv2_install_pretrained_model_from_zip`.

Repository software is provided under the MIT License. See the hosted model repository for the model artifact.
