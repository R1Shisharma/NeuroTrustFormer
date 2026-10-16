# EEG Experiment Execution Guide

## Scope

The current reproducibility package focuses on the CHB-MIT EEG seizure-versus-non-seizure
window-classification experiments. The CHB-MIT EEG and ADNI MRI/clinical resources are
separate cohorts; this workflow does not reproduce synchronized multimodal fusion.

## Software

- Python 3
- Dependencies listed in `requirements.txt`
- A compatible PyTorch installation for the chosen CPU/GPU environment
- Access to the CHB-MIT Scalp EEG Database under its applicable terms
- The repository's preprocessing and model modules

The package names were checked against imports in the working notebook and repository
modules. The notebook does not establish a complete, tested version lockfile; dependency
versions must be recorded after the workflow is tested in a clean environment.

## Installation

Create and activate a virtual environment, then install the repository dependencies:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PyTorch installation fails or GPU acceleration is required, install the PyTorch build
appropriate for the operating system and hardware using the official PyTorch installation
selector, then rerun the dependency check.

## Data placement and paths

Obtain CHB-MIT data directly from the authorized source; do not place raw EDF files in this
GitHub repository. Configure the local data path expected by `chbmit_eeg_preprocessing.py`
and the experiment notebook before running preprocessing or evaluation. The working notebook
contains Google Colab/Google Drive-specific setup and paths, so it is not yet a fully
platform-independent, one-command reproduction workflow.

Do not assume that a folder path or checkpoint name is correct merely because it appears in
an earlier notebook cell. Verify the final fold-specific split, normalization artifact, and
checkpoint configuration before evaluating a fold.

## Recommended execution order

1. Install dependencies and record Python, NumPy, pandas, PyTorch, MNE, WFDB, pyEDFlib,
   scikit-learn, SciPy, and Matplotlib versions.
2. Place the authorized CHB-MIT EDF data in the configured data directory.
3. Run the preprocessing/data-integrity checks and verify the expected 17-channel,
   256 Hz, 10-second window specification.
4. Verify the locked case-level fold assignments and the fold-specific training-only
   normalization artifacts.
5. For each fold, confirm the exact best checkpoint and configuration before running
   held-out test inference.
6. Generate and audit the five-fold metrics, uncertainty/calibration outputs, structured
   channel-missingness summary, and transformer-ablation summary.
7. Compare generated outputs with the CSVs in `Results/`; investigate discrepancies rather
   than overwriting results silently.

## Important limitations

- This guide documents the intended workflow; a clean, end-to-end run has not yet been
  demonstrated by this repository package.
- The working notebook contains Colab-specific operations and exploratory/failed cells.
  A cleaned reproduction notebook or scripts are still needed for fully straightforward
  reviewer execution.
- Temperature scaling was evaluated on validation predictions; do not claim improved
  calibration on held-out test data unless such an evaluation is separately verified.
- The missingness experiments concern EEG channel loss, not missing MRI or clinical modalities.
- Do not upload raw EEG/MRI/clinical participant data or identifying metadata to this repository.
