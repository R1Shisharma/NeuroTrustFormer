# NeuroTrustFormer

**An Uncertainty-Aware Multimodal Transformer for Early Prediction of Neurological Disorders Using EEG, MRI, and Clinical Data**

## Overview

NeuroTrustFormer is a research framework investigating uncertainty-aware deep learning for neurological disorder analysis using neurophysiological signals, neuroimaging, and clinical information.

The project includes EEG preprocessing, model development, evaluation, uncertainty estimation, calibration analysis, and robustness experiments. The repository is intended to support transparent inspection of the methods and code associated with the research manuscript submitted to *Health Information Science and Systems*.

**Important scope clarification:** EEG and MRI data in this project originate from separate datasets. They are not synchronized or patient-paired. Results from the separate modalities should therefore be interpreted as modality-specific evaluations, not as validation of joint EEG–MRI fusion on paired patients.

## Research Objectives

The project investigates:

- EEG signal preprocessing and classification.
- Transformer-based representation learning for neurological data.
- Model uncertainty estimation using Monte Carlo dropout.
- Probability calibration and uncertainty evaluation.
- Robustness under structured EEG channel-missingness scenarios.
- Exploratory MRI preprocessing and quality-control workflows.

## Repository Contents

The repository will contain the following components as they are prepared and verified:

- `code/chbmit_eeg_preprocessing.py` — EEG preprocessing utilities.
- `code/neurotrustformer_eeg_model.py` — EEG model implementation.
- `notebooks/` — Relevant research and evaluation notebooks, where included.
- `results/` — Selected non-sensitive evaluation summaries and result tables, where included.
- `README.md` — Project overview and documentation.

Only files actually included in the repository should be considered available. Additional scripts, notebooks, and experimental artifacts may be added as the repository is prepared.

## Datasets

### EEG

The EEG experiments use the CHB-MIT Scalp EEG Database. The evaluated task is seizure versus non-seizure EEG window classification.

This classification task should not be interpreted as seizure forecasting with a predefined preictal prediction horizon.

### MRI and Clinical Data

The project also includes exploratory workflows involving Alzheimer's Disease Neuroimaging Initiative (ADNI) MRI and clinical resources.

MRI preprocessing and quality-control activities must be distinguished from completed clinical validation. The EEG and ADNI datasets are separate cohorts and do not provide paired multimodal observations for this study.

Dataset access, reuse, and redistribution are subject to the applicable data-provider terms. Restricted source data are not distributed through this repository.

## Evaluation and Reproducibility

The EEG evaluation includes case-level five-fold cross-validation, model performance metrics, uncertainty analysis, calibration assessment, and structured channel-missingness experiments.

The reported results should be interpreted according to the specific experiment and evaluation protocol. Calibration metrics may vary across measures, and no uniform improvement should be assumed across all calibration metrics.

Reproduction requires access to the appropriate datasets, compatible software dependencies, the relevant experiment configurations, and any required model checkpoints. Dataset access and preprocessing may require additional steps.

## Software Requirements

The code is written in Python. The exact package versions and installation instructions will be documented after the experimental environment and dependencies have been verified.

Please consult the relevant scripts and notebooks for their specific imports and execution requirements.

## Responsible Data Use

- Do not upload identifiable participant information or restricted clinical or neuroimaging data.
- Respect the access, licensing, and redistribution conditions of the original datasets.
- Do not publish credentials, access tokens, or private filesystem paths.
- Treat exploratory preprocessing results separately from fully validated experimental results.

## Research Manuscript

**Title:** NeuroTrustFormer: An Uncertainty-Aware Multimodal Transformer for Early Prediction of Neurological Disorders Using EEG, MRI, and Clinical Data

**Journal:** Health Information Science and Systems

**Purpose of repository:** Research code transparency and support for methodological review.

## Code Availability

This repository is intended to provide access to the research code and supporting documentation associated with the manuscript. The availability of individual scripts, notebooks, datasets, and experimental artifacts depends on the contents of the repository and applicable data-use restrictions.

## Contact

Research correspondence should be directed to the corresponding author through the contact details provided in the manuscript.
