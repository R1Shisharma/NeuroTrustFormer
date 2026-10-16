# Experimental Results

This folder contains selected evaluation outputs associated with the NeuroTrustFormer research manuscript submitted to *Health Information Science and Systems*.

## Contents
File                                                                                    Description

CHBMIT_5FOLD_test_metrics.csv                                      Fold-level test metrics for the CHB-MIT EEG classification experiment.

CHBMIT_5FOLD_aggregate_mean_SD_CI.csv                             Aggregate means, standard deviations, and reported confidence intervals across the five folds.

CHBMIT_FIVEFOLD_temperature_scaling_foldwise_analysis.csv         Fold-wise validation results for the temperature-scaling calibration analysis.

missingness_summary.csv                                            Mean and standard deviation of evaluation metrics for complete-input and structured EEG                                                                            channel-missingness scenarios.

TRANSFORMER ABLATION.zip                                           Archived transformer-ablation experiment artifacts.

Without Transformer.zip                                             Archived artifacts for the model without the transformer component.

## Interpretation

The primary quantitative evaluation concerns seizure-versus-non-seizure EEG window classification using the CHB-MIT dataset. The five-fold results reflect the specified case-level evaluation protocol and should not be interpreted as evidence of external cross-site validation or seizure forecasting with a predefined preictal horizon.

The missingness analysis evaluates structured EEG channel loss. It does not establish robustness to missing MRI or clinical modalities.

Temperature scaling was assessed using validation predictions. Its effects differed across calibration metrics; improvement should not be assumed for every metric or for independent held-out test data.

The transformer-ablation results compare the full model with a version in which the transformer component is removed. The comparisons should be interpreted alongside the reported fold-level variability and statistical tests.

## Reproducibility and limitations

These files are selected experimental outputs, not a complete standalone reproduction package. Reproducing the experiments requires the relevant source code, compatible dependencies, experiment configurations, and access to the CHB-MIT dataset under its applicable terms.

Confidence intervals and aggregate statistics should be interpreted according to the calculation method used for each source file. The file `missingness_summary.csv` retains mean and standard deviation values; confidence intervals are not included in that summary.

The repository does not establish synchronized EEG–MRI–clinical validation. Restricted source data are not distributed in this folder.
