# NeuroTrustFormer Reproducibility Package

## Purpose

This directory contains source code and supporting documentation for inspecting and reproducing the experiments associated with the NeuroTrustFormer manuscript submitted to *Health Information Science and Systems*.

The package includes separate workflows for CHB-MIT EEG experiments and exploratory ADNI MRI/clinical experiments. These datasets represent separate cohorts and must not be interpreted as synchronized or patient-paired EEG–MRI multimodal data.

## Package Contents

### CHB-MIT EEG

- `chbmit_eeg_preprocessing.py` — EEG preprocessing utilities.
- `neurotrustformer_eeg_model.py` — EEG model implementation.
- `CHBMIT_fivefold_training_source_extract.ipynb` — Notebook containing the EEG training and evaluation workflow.

### Exploratory ADNI MRI and Clinical Experiments

- `R2C3_fivefold_orchestrator_fixed.py` — Five-fold orchestration script.
- `r2c3_baselines_and_export_EXPLORATORY.py` — Baseline experiments and result exports.
- `train_adni_rgaf_shmr.py` — RGAF_SHMR training script.
- `requirements-adni.txt` — Dependencies for the ADNI exploratory workflow.
- `HISS2_R2C3_ADNI_FiveFold_Experimental_Record.docx` — Supporting experimental record.

## ADNI Five-Fold Workflow

The orchestrator is designed to generate stratified five-fold participant assignments, create fold-specific train, validation, and test splits, run the model and baseline scripts, and export fold-level and aggregate metrics.

The existing exploratory results include fold-specific checkpoints, training histories, evaluation summaries, participant assignments, and aggregate metrics. The aggregate file records results for five folds across RGAF_SHMR, MRI_only, and MMSE_only.

For each reported run, the intended workflow selects checkpoints using validation performance rather than test-set performance. The saved evaluation summaries record `test_used_for_checkpoint_selection` as `false`.

### Reproduction Requirements

1. Obtain the relevant source datasets directly from their authorized providers and comply with their data-use terms.
2. Install the dependencies in the relevant requirements file in a compatible Python environment.
3. Inspect the scripts and their command-line help to identify the required manifest, output directory, and execution parameters.
4. Verify the participant manifest, locked fold assignments, and train/validation/test membership before running experiments.
5. Record the Python and package versions, random seeds, configuration parameters, and checkpoint paths for every fold and model.
6. Run the orchestrator using the verified local dataset manifest and a new output directory.
7. Compare the generated fold-level outputs and aggregate metrics with the archived results. Investigate any discrepancies rather than silently overwriting existing results.

Exact reproduction depends on access to the authorized datasets, compatible dependencies, valid local paths, and matching experiment configurations. Do not assume that an existing output file alone proves that a clean end-to-end rerun has succeeded.

## Interpretation and Limitations

The ADNI experiments are exploratory and do not establish clinical validity. MMSE may be strongly associated with diagnosis, so clinical-only and MRI-only baselines should be considered when interpreting performance. Learned fusion weights must not be described as calibrated uncertainty without separate supporting evidence.

The CHB-MIT EEG and ADNI MRI/clinical workflows use separate datasets. Results from these workflows do not demonstrate joint EEG–MRI fusion on paired participants.

The package contains source code and supporting artifacts, but a successful independent, clean end-to-end reproduction must be demonstrated and documented separately before full reproducibility is claimed.

## Data Protection

Do not upload raw EEG, MRI, clinical participant data, identifiable metadata, credentials, or private local machine paths. Obtain restricted datasets through their authorized providers.

## Reporting

Report metrics for each fold as well as aggregate means and standard deviations. State the evaluation protocol, dataset, split strategy, random seeds, checkpoint-selection criterion, and relevant limitations. Do not imply that exploratory results constitute clinical validation.
