# PerthesMetrics 2D nnU-Net Segmentation Model

## Summary

PerthesMetrics is a 2D nnU-Net v2 model for multiclass anatomic segmentation of pediatric hip radiographs in Legg-Calve-Perthes disease.

## Inputs

Each RGB radiograph is represented as three aligned NIfTI channels.

## Outputs

The integer segmentation contains background (0), acetabulum (1), greater trochanter (2), femoral head (3), lesser trochanter (4), femoral neck (5), femoral shaft (6), sourcil (7), and triradiate cartilage (8).

## Training and evaluation

The release model uses `nnUNetTrainer`, `nnUNetPlans`, and the `2d` configuration with five 1,000-epoch folds. Evaluation must use patient-grouped five-fold cross-validation: every view, side, and time point from a patient remains in one validation fold, with no patient overlap between training and validation.

Complete cohort inclusion criteria, annotation procedures, patient counts, final aggregate metrics, and external validation details should be added from the accepted paper before public release.

## Intended use and limitations

Research use only. The model is not a medical device and is not a substitute for clinician review. Performance may not generalize across institutions, acquisition systems, populations, image processing pipelines, or radiographs outside the training distribution. Small or poorly visualized structures may have lower accuracy. Users must independently validate performance for their setting.

## Package

The hosted artifact is an nnU-Net pretrained model zip installable with `nnUNetv2_install_pretrained_model_from_zip`.
