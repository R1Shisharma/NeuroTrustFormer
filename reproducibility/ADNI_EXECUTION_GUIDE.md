# ADNI Exploratory Five-Fold Execution Guide

## 1. Purpose and Scope

This guide documents the intended execution workflow for the exploratory ADNI MRI and clinical experiments in this directory. The workflow is separate from the CHB-MIT EEG experiments. These datasets are not synchronized or patient-paired, and the ADNI experiments do not establish clinical validity.

The orchestrator is `R2C3_fivefold_orchestrator_fixed.py`. It invokes `train_adni_rgaf_shmr.py` for RGAF_SHMR and `r2c3_baselines_and_export_EXPLORATORY.py` for the MRI-only and MMSE-only baselines.

## 2. Software and Dependencies

Use a compatible Python environment and install the dependencies listed in `requirements-adni.txt`.

Example Windows commands, run from this directory:

```powershell
py -3.12 -m venv .venv_adni
.\.venv_adni\Scripts\python.exe -m pip install -r requirements-adni.txt
.\.venv_adni\Scripts\python.exe R2C3_fivefold_orchestrator_fixed.py --help
```

The commands above create a separate environment and display the available options. They do not run training.

## 3. Required Input Manifest

The orchestrator accepts a participant-level CSV manifest through `--manifest`. It requires these columns:

- `PTID`
- `diagnosis_label`
- `MMSE`
- `nifti_path`
- `approved_for_training`
- `exploratory_only`
- `final_qc_approval`

The participant identifier must be unique. Diagnosis labels must be `CN`, `MCI`, or `AD`. The script requires `exploratory_only=True`, `final_qc_approval=NOT_GRANTED`, and the existing approved-for-training flag. These guards must not be bypassed or altered to imply final quality-control approval.

Obtain source data through the authorized provider and configure valid local paths. Do not upload participant-level source data or identifiable clinical metadata to this repository.

## 4. Fold Generation and Random Seeds

The orchestrator creates five stratified outer folds from the input manifest and writes:

- `participant_fold_assignments.csv`
- `fold_1/fold_manifest.csv` through `fold_5/fold_manifest.csv`
- A `split_summary.json` within each fold directory

The default main seed is `20261020`. The validation split seed for fold *k* is the main seed plus `k × 100`; the training seed for fold *k* is the main seed plus `k × 1000`. The default validation fraction is `0.15`, with allowed values from `0.05` to `0.30`.

These are the script defaults. The actual command-line arguments and saved run summaries must be checked before attributing these settings to a historical result.

The script checks participant overlap among train, validation, and test sets within each generated fold. For strict repeatability, preserve and version the exact generated participant assignments and fold manifests used for the run. Regenerating folds from a seed should not be treated as proof that the resulting assignments match a previously reported experiment.

## 5. Execution Command

After verifying the manifest, dependencies, and output destination, the intended command structure is:

```powershell
py -3.12 R2C3_fivefold_orchestrator_fixed.py `
  --manifest "PATH_TO_AUTHORIZED_MANIFEST.csv" `
  --outroot "PATH_TO_NEW_OUTPUT_DIRECTORY" `
  --seed 20261020 `
  --epochs 10 `
  --batch-size 2 `
  --patience 3 `
  --validation-fraction 0.15
```

Replace both placeholder paths with valid local paths. Use a new output directory for a new run so that existing results are not overwritten. This command is a template; it should not be executed until the input manifest and run configuration have been verified.

## 6. Output and Model Mapping

For each fold `fold_k`, the orchestrator is intended to produce:

| Model or artifact | Expected location |
|---|---|
| RGAF_SHMR training outputs | `fold_k/RGAF_SHMR/` |
| RGAF_SHMR evaluation summary | `fold_k/RGAF_SHMR/evaluation_summary.json` |
| RGAF_SHMR checkpoint | `fold_k/RGAF_SHMR/best_model.pt` |
| MRI-only baseline | `fold_k/BASELINES/MRI_only/` |
| MMSE-only baseline | `fold_k/BASELINES/MMSE_only/` |
| Baseline combined export | `fold_k/BASELINES/R2C3_combined_results.csv` |
| Participant split manifest | `fold_k/fold_manifest.csv` |
| Split summary | `fold_k/split_summary.json` |

The scripts also export training histories and evaluation summaries. The expected paths above describe the output layout implemented by the workflow; verify each file exists before treating a run as complete.

The orchestrator reads the three model evaluation summaries and writes:

- `fivefold_metrics_by_model.csv`
- `fivefold_metrics_by_model.json`
- `fivefold_mean_sd_summary.csv`
- `fivefold_mean_sd_summary.json`

The aggregate metrics are derived from the test metrics recorded in the fold-level evaluation summaries. Preserve the fold outputs alongside the aggregate files so that this relationship can be audited.

## 7. Checkpoint Selection and Evaluation

The saved evaluation summaries record whether the test set was used for checkpoint selection. The intended procedure selects checkpoints using validation performance and evaluates the selected checkpoint on the held-out test split.

For every fold and model, verify the saved configuration, random seed, validation selection criterion, checkpoint path, test metrics, and split counts. Do not infer checkpoint provenance solely from a filename or file size.

## 8. Verification Checklist

Before reporting a reproduction as successful, verify all of the following:

1. The exact input manifest and its provenance are recorded.
2. The locked participant assignments and all five fold manifests are available and unchanged.
3. Train, validation, and test participants are disjoint in every fold.
4. The command-line arguments, dependency versions, and random seeds are recorded.
5. Each fold contains the expected checkpoint, training history, and evaluation summary for all three models.
6. Each aggregate model-fold row matches the corresponding fold-level test metrics.
7. Aggregate means and sample standard deviations can be regenerated from the fold-level metrics.
8. Any differences from the archived results are documented rather than silently overwritten.

## 9. Current Reproducibility Status and Limitations

The repository contains exploratory source scripts and supporting documentation. Existing local audits have found five-fold aggregate entries for RGAF_SHMR, MRI-only, and MMSE-only and matching displayed metrics for the inspected Fold 1 RGAF_SHMR comparison.

These checks do not, by themselves, demonstrate a successful clean end-to-end reproduction. The exact historical fold assignments, complete checkpoint-to-configuration provenance, and independent regeneration of the reported results must be verified before the experiment is described as fully reproducible.

These ADNI experiments are exploratory and do not establish clinical validity. They must not be presented as synchronized EEG–MRI fusion or as clinical validation.
