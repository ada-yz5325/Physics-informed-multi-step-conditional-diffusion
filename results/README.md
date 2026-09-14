# Results

This directory contains the lightweight final outputs used for evaluation and reporting.

## final_metrics

`final_metrics/` contains the CSV evaluation outputs for the final incompressible (INC) and transonic (TRA) experiments.

For INC experiments, results are reported separately for the low- and high-Reynolds-number test cases using 500-step rollouts. The stored CSV files include:

- aggregate evaluation metrics over the rollout;
- MSE as a function of rollout timestep;
- relative L2 error as a function of rollout timestep.

For TRA experiments, the stored CSV files correspond to the `Tralong_500step` long-rollout evaluation output used for the final 500-step TRA evaluation. The stored CSV files include:

- aggregate evaluation metrics over the rollout;
- MSE as a function of rollout timestep;
- relative L2 error as a function of rollout timestep;
- predicted continuity residual MSE as a function of rollout timestep;
- ground-truth continuity residual MSE as a function of rollout timestep.

All final metrics included in this directory are based on 500-step rollout evaluations.

The numerical values reported in the final report tables were taken from the corresponding CSV evaluation outputs in `final_metrics/` and organised directly in the LaTeX report source. No separate table-generation script is used.

## final_figures

`final_figures/` contains the PDF and PNG versions of the figures generated for the final report, including spatial flow comparisons, rollout-error evolution, training convergence, and the accuracy-efficiency comparison.

## Excluded outputs

Large intermediate experiment outputs are intentionally not tracked in Git. These include model checkpoints, raw rollout arrays (`.npz`), temporary visualisations, and intermediate sampling results.

The included CSV files and final figures provide the lightweight numerical and visual outputs associated with the final experiments.
