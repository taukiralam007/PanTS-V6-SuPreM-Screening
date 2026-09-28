# PanTS V6 SuPreM SegResNet

## Overview

This repository contains the PanTS V6 pancreatic tumor segmentation pipeline developed for full-cohort training and official PanTS evaluation.

The model is based on a 3D MONAI SegResNet initialized from a SuPreM abdominal CT pretrained checkpoint.

## Dataset Protocol

- PanTS-tr: 9,000 official training cases
- Training subset: 8,100 cases
- Held-out validation subset: 900 cases
- PanTS-te: 901 official in-distribution test cases

The 900-case validation set was used for inference-parameter selection.
The final configuration was frozen before evaluation on the official 901-case test set.
No threshold tuning was performed using the official test results.

## Model

- Architecture: 3D SegResNet
- Framework: MONAI / PyTorch
- Input: 3D abdominal CT
- Initialization: SuPreM pretrained SegResNet
- Parameters: 4,700,914
- Final checkpoint: `checkpoints/final_v6_full.pt`

## Frozen Validation Configuration

- Lesion threshold: 0.70
- Pancreas gate: 0.20
- Minimum connected component: 200
- Dilation: 3

| Metric | Result |
|---|---:|
| Pancreas DSC | 0.7782 |
| Lesion DSC | 0.3391 |
| P-Sen | 0.9545 |
| T-Sen | 0.5361 |
| Specificity | 0.5283 |
| AUC | 0.8677 |

## Official 901-Case PanTS Test Results

| Metric | Result |
|---|---:|
| Pancreas DSC | 0.8101 |
| Lesion DSC | 0.4007 |
| Patient-wise Sensitivity (P-Sen) | 0.8146 |
| Tumor-wise Sensitivity (T-Sen) | 0.6707 |
| Specificity | 0.4507 |
| AUC | 0.8102 |

Detection counts:

- True positives: 123
- True negatives: 338
- False positives: 412
- False negatives: 28
- Tumors detected: 112 / 167

## Repository Structure

```text
PanTS-V6-SuPreM-Screening/
├── README.md
├── requirements.txt
├── .gitignore
├── code/
│   ├── PanTS_V6_Colab.ipynb
│   └── PanTS_V6_Colab.py
├── checkpoints/
│   └── final_v6_full.pt
└── results/
    ├── official_901_metrics.json
    ├── official_901_cases.csv
    └── frozen_validation_config.json
```

## Code

- `code/PanTS_V6_Colab.ipynb`
- `code/PanTS_V6_Colab.py`

The pipeline includes:

- PanTS data preparation
- SuPreM checkpoint initialization
- 3D SegResNet training
- 8,100 / 900 train-validation split
- Validation-only inference tuning
- Frozen configuration selection
- Official 901-case evaluation
- Per-case result export
- Aggregate metric calculation

## Reproducibility

- `results/frozen_validation_config.json`
- `results/official_901_metrics.json`
- `results/official_901_cases.csv`
- `checkpoints/final_v6_full.pt`

## Evaluation Integrity

The official 901-case test set was evaluated only after the validation configuration was frozen.

The official test set was not used to select thresholds or post-processing parameters.

## PanTS Reference

PanTS: The Pancreatic Tumor Segmentation Dataset

Johns Hopkins University

NeurIPS 2025

Official repository:
https://github.com/MrGiovanni/PanTS

## Author

Taukir Alam, Ph.D.
Feng Chia University
Taichung, Taiwan
