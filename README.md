# Physics-Informed Multi-Step Diffusion for Fluid Flow Forecasting

Portfolio version of an MSc Independent Research Project at Imperial College London on **autoregressive conditional diffusion models for 2D fluid-flow forecasting**.

The project extends an Autoregressive Conditional Diffusion Model (ACDM) baseline in two directions:

1. **Flow-specific physics-informed regularisation** to improve physical consistency.
2. **Multi-step conditional diffusion** to jointly predict several future states and reduce repeated reverse-diffusion sampling during long rollouts.

The study evaluates these ideas on incompressible and transonic cylinder-flow benchmarks from four perspectives: **prediction accuracy, temporal behaviour, physical consistency, and inference efficiency**.

> **Report:** [MSc IRP Final Report (public portfolio copy)](docs/Yini_Zhu_MSc_IRP_Final_Report_Public.pdf)

## Key Results

- **INC, high Reynolds number:** vorticity regularisation reduced `predMSE` from `1.32e-2` to `3.61e-4` (**~97.3% lower**) and relative L2 from `0.376` to `0.062`.
- **INC, low Reynolds number:** the same vorticity regularisation reduced `predMSE` from `2.86e-3` to `1.88e-3` and relative L2 from `0.183` to `0.148`.
- **INC multi-step (`C=P=2`):** approximately halved diffusion calls and reverse-denoising steps and delivered about a **2x wall-clock speed-up**, but introduced a clear accuracy penalty relative to the vorticity-regularised single-step model.
- **TRA single-step:** continuity regularisation with `lambda_cont = 0.2` improved `predMSE` from `2.93e-2` to `2.69e-2` and continuity MSE from `3.05e-4` to `2.03e-4`.
- **TRA multi-step:** all tested horizons (`C=P=2,3,4`) reduced prediction error and inference time relative to the selected continuity-regularised single-step reference. `P=2` achieved the lowest `predMSE` (`2.16e-2`), while `P=4` achieved the shortest inference time (`1.22 min`, **2.79x speed-up**).
- The experiments also show an important limitation: lower aggregate prediction error does **not** necessarily imply better local physical consistency. TRA multi-step models improved accuracy and efficiency while increasing temporal-increment and continuity residual errors.

<p align="center">
  <img src="diffusion_model/results/final_figures/fig_A1_inc_highre_vorticity.png" width="48%" />
  <img src="diffusion_model/results/final_figures/fig_tra_accuracy_efficiency_tradeoff.png" width="48%" />
</p>

## Research Question

Single-step autoregressive diffusion has two practical limitations for fluid forecasting:

- the standard noise-prediction objective does not explicitly enforce flow physics;
- every new physical time step requires a complete reverse-diffusion process, making long rollouts expensive.

This project therefore asks whether physics-aware losses can improve flow consistency and whether multi-step diffusion can reduce inference cost without unacceptable losses in forecast quality.

## Datasets

Two 2D cylinder-flow datasets from the benchmark of Kohl, Chen and Thuerey are used.

| Case | Flow regime | State variables | Conditioning parameter | Trajectories | Snapshots per trajectory | Grid |
| --- | --- | --- | --- | ---: | ---: | --- |
| `128_inc` | Incompressible viscous cylinder wake | pressure `p`, velocity `u,v` | Reynolds number | 91 | 1300 | `128 x 64` |
| `128_tra` | Transonic compressible cylinder flow | pressure `p`, density `rho`, velocity `u,v` | Mach number | 41 | 1001 | `128 x 64` |

INC evaluation is performed on held-out low- and high-Reynolds-number regimes (`Re=100-200` and `Re=900-1000`). TRA uses held-out Mach-number ranges from the benchmark split.

Raw simulation data are not included because of size. See [`diffusion_model/data/README.md`](diffusion_model/data/README.md) for the source, expected directory layout, and preprocessing notes.

## Baseline ACDM

The baseline conditions on the two preceding states and generates one future state:

```text
C = 2, P = 1
```

The final experiments use:

- 20 diffusion steps;
- a linear noise schedule;
- noisy conditioning;
- a ConvNeXt-based U-Net denoising backbone with encoder-decoder levels, skip connections, attention modules, and timestep embeddings;
- Huber noise-prediction loss;
- autoregressive long-horizon rollout.

During training, the model predicts the Gaussian noise added to the future target. During inference, each generated state is fed back into the conditioning history for subsequent predictions.

## Physics-Informed Regularisation

The physics losses are applied to reconstructed clean flow states rather than directly to the predicted diffusion noise.

### Incompressible flow

Two quantities are evaluated from the predicted velocity field:

```text
divergence = du/dx + dv/dy
vorticity  = dv/dx - du/dy
```

The INC training objective is:

```text
L_INC = L_diff + lambda_div * L_div + lambda_vort * L_vort
```

where `L_div` penalises predicted divergence and `L_vort` is the MSE between predicted and ground-truth vorticity fields.

The strongest tested INC configuration is **vorticity-only regularisation with `lambda_vort = 0.01`**. Divergence-only and combined regularisation did not produce the same overall improvement.

### Transonic flow

For compressible TRA flow, incompressibility is not applicable. The project instead uses the continuity equation:

```text
drho/dt + d(rho*u)/dx + d(rho*v)/dy = 0
```

The TRA objective is:

```text
L_TRA = L_diff + lambda_cont * L_cont
```

with `L_cont` computed from the squared discrete continuity residual over the valid fluid region. The selected single-step setting is `lambda_cont = 0.2`.

The continuity metric should be interpreted as a **relative consistency diagnostic**, not an exact dimensional conservation error, because finite differences, resampling, normalisation, and post-processing contribute to a non-zero residual even for the ground-truth fields.

## Multi-Step Conditional Diffusion

The multi-step formulation jointly generates several consecutive future states within one reverse-diffusion process.

For history length `C` and prediction horizon `P`:

```text
(s[t-C], ..., s[t-1]) -> (s_hat[t], ..., s_hat[t+P-1])
```

The experiments use symmetric windows, `C=P`. Future states are concatenated along the channel dimension and treated as one joint diffusion target. The diffusion schedule, number of denoising steps, noisy-conditioning strategy, and basic U-Net architecture remain consistent with the single-step baseline.

For a rollout of `T` states, the number of complete diffusion calls changes approximately from:

```text
single-step: T
multi-step: ceil(T / P)
```

Final multi-step experiments include:

| Regime | Configuration | Physics term | Additional variant |
| --- | --- | --- | --- |
| INC | `C=P=2` | vorticity | base multi-step |
| INC | `C=P=2` | vorticity | temporal-consistency loss |
| INC | `C=P=2` | vorticity | horizon weighting (`w1=1.0`, `w2=0.5`) |
| TRA | `C=P=2` | continuity | — |
| TRA | `C=P=3` | continuity | — |
| TRA | `C=P=4` | continuity | — |

The INC temporal-consistency and horizon-weighting refinements were tested separately and did not recover the accuracy lost by the basic multi-step formulation.

## Quantitative Results

### INC physics-loss ablation

| Model | Low-Re `predMSE` | Low-Re rel. L2 | High-Re `predMSE` | High-Re rel. L2 |
| --- | ---: | ---: | ---: | ---: |
| Diffusion-only baseline | `2.86e-3` | `0.183` | `1.32e-2` | `0.376` |
| Divergence `0.01` | `2.67e-3` | `0.176` | `6.22e-3` | `0.258` |
| **Vorticity `0.01`** | **`1.88e-3`** | **`0.148`** | **`3.61e-4`** | **`0.062`** |
| Div `0.01` + Vort `0.01` | `3.04e-3` | `0.188` | `3.11e-3` | `0.182` |
| Div `0.003` + Vort `0.003` | `1.98e-3` | `0.152` | `5.84e-3` | `0.250` |
| Div `0.003` + Vort `0.01` | `8.93e-3` | `0.323` | `7.81e-3` | `0.289` |

Vorticity regularisation also produced the lowest vorticity MSE among the tested single-step INC models in both Reynolds-number regimes.

### INC single-step vs multi-step

| Model | Low-Re `predMSE` | High-Re `predMSE` | Interpretation |
| --- | ---: | ---: | --- |
| SS + vorticity | `1.88e-3` | `3.61e-4` | most accurate selected INC model |
| MS `C=P=2` + vorticity | `5.51e-3` | `1.93e-3` | faster, but less accurate |
| MS + vorticity + temporal loss | `2.42e-2` | `3.34e-2` | refinement did not recover accuracy |
| MS + vorticity + horizon weighting | `1.62e-2` | `2.95e-2` | refinement did not recover accuracy |

The basic `C=P=2` model approximately halves sampling cost. Across five 500-step INC rollouts, diffusion calls decrease from `2500` to `1250` and reverse-denoising steps from `50000` to `25000`, corresponding to about `1.97-1.99x` measured wall-clock speed-up.

### TRA single-step continuity regularisation

| Model | `predMSE` | rel. L2 | Stability slope | Continuity MSE |
| --- | ---: | ---: | ---: | ---: |
| Baseline | `2.93e-2` | `0.266` | `4.35e-5` | `3.05e-4` |
| Continuity `0.01` | `3.26e-2` | `0.281` | `2.84e-5` | `3.85e-4` |
| **Continuity `0.2`** | **`2.69e-2`** | **`0.255`** | **`2.76e-5`** | **`2.03e-4`** |

### TRA multi-step accuracy and efficiency

| Model | `predMSE` | rel. L2 | Final-step MSE | Time (min) | Diffusion calls | Speed-up |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SS + cont. | `2.69e-2` | `0.255` | `2.69e-2` | `3.41` | `1000` | `1.00x` |
| MS `C=P=2` + cont. | **`2.16e-2`** | **`0.228`** | `2.55e-2` | `1.82` | `500` | `1.87x` |
| MS `C=P=3` + cont. | `2.17e-2` | `0.229` | `2.42e-2` | `1.47` | `334` | `2.32x` |
| MS `C=P=4` + cont. | `2.21e-2` | `0.231` | **`2.36e-2`** | **`1.22`** | `250` | **`2.79x`** |

The longer TRA horizons improve efficiency with only a modest increase in average prediction error among the tested multi-step models. However, their continuity residuals and temporal-increment errors are substantially larger than the single-step reference, so the result is best understood as an **accuracy-physics-efficiency trade-off**.

## Evaluation

Final evaluation uses 500-step long-horizon rollouts and includes:

- **Prediction accuracy:** `predMSE`, relative L2, field-specific MSE/L2, final-step MSE.
- **Temporal behaviour:** temporal consistency/increment error, MSE per timestep, relative L2 per timestep, rollout stability slope.
- **INC physics:** divergence statistics, vorticity MSE, vorticity relative L2.
- **TRA physics/structure:** continuity MAE/MSE, vorticity metrics, pressure-gradient and density-gradient diagnostics.
- **Inference efficiency:** wall-clock inference time, diffusion-call count, reverse-denoising-step count.

The rollout stability slope is the linear-fit slope of MSE over rollout timesteps and is used as an error-growth diagnostic rather than as a substitute for absolute prediction error.

## Repository Layout

```text
.
├── README.md
├── LICENSE
├── docs/
│   └── Yini_Zhu_MSc_IRP_Final_Report_Public.pdf
└── diffusion_model/
    ├── data/
    │   └── README.md                   # dataset source and expected layout
    ├── environment.yml                 # Conda environment
    ├── activate_acdm.sh                # portable environment helper
    ├── examples/                       # simplified entry-point examples
    ├── results/
    │   ├── final_metrics/              # final lightweight CSV metrics
    │   ├── final_figures/              # report-quality figures
    │   └── README.md
    ├── src/
    │   ├── turbpred/                   # core model/loss/training implementation
    │   ├── lsim/                       # baseline perceptual-flow metric code
    │   ├── training_*.py               # final experiment training entry points
    │   ├── sample_models_*.py          # sampling and 500-step rollout evaluation
    │   ├── compare_*.py                # experiment comparison utilities
    │   ├── plot_*.py                   # report/result figure generation
    │   └── diagnose_*.py               # physical/flow diagnostics
    └── *.pbs                            # HPC training and sampling job configurations
```

Raw datasets, model checkpoints, full training runs, logs, and large intermediate rollout arrays are intentionally excluded.

## Reproducibility and Paths

The public experiment scripts use environment variables or repository-relative defaults rather than the original Imperial filesystem paths.

From `diffusion_model/`:

```bash
conda env create -f environment.yml
conda activate ACDM

export PROJECT_ROOT="$(pwd)"
export DATA_ROOT="$PROJECT_ROOT/data"
```

Training scripts read `DATA_ROOT`. PBS scripts are retained because they document the HPC experiment matrix and have been sanitised to remove user-specific `/rds/...` and Conda-library paths.

## Code Provenance and Project Contributions

This project builds on the open-source ACDM implementation associated with Kohl, Chen and Thuerey. Baseline components retained in `diffusion_model/src/turbpred/` provide the reference architecture.

Project-specific work represented in this repository includes:

- divergence- and vorticity-based INC physics losses;
- compressible continuity-equation regularisation for TRA;
- multi-step conditional diffusion with joint future-state targets;
- temporal-consistency and horizon-weighting variants;
- long-rollout evaluation and error-growth diagnostics;
- physical-consistency metrics for two different flow regimes;
- accuracy-efficiency analysis and HPC experiment orchestration;
- dataset conversion, preprocessing, comparison, plotting, and diagnostic utilities used in the final study.

The upstream ACDM code is MIT licensed; the original license notice is preserved in this repository.

## Interpretation and Limitations

The experiments support several conclusions that are useful beyond the headline metrics:

- Physics-informed regularisation is **not automatically beneficial**. Its effect depends on the physical quantity, loss weight, and flow regime.
- The physics losses are **soft constraints**; they encourage consistency but do not guarantee exact conservation.
- Multi-step prediction reduces autoregressive feedback and repeated sampling, but a larger joint prediction target can also increase modelling difficulty.
- INC and TRA react differently: INC mainly shows an accuracy-efficiency trade-off, while TRA shows a more favourable prediction-error/efficiency relationship but weaker continuity consistency.
- Longer INC horizons were not pursued because `C=P=2` already introduced a clear accuracy penalty; TRA was evaluated only up to `C=P=4`.
- Physics-loss weights were selected through limited sensitivity experiments rather than exhaustive optimisation.

## References

- Kohl, G., Chen, L.-W. and Thuerey, N. **Benchmarking Autoregressive Conditional Diffusion Models for Turbulent Flow Simulation.** *Neural Networks*, 199:108641, 2026.
- Shysheya, A. et al. **On Conditional Diffusion Models for PDE Simulations.** NeurIPS, 2024.
- Luo, D. et al. **DiffFluid: Plain Diffusion Models Are Effective Predictors of Flow Dynamics.** arXiv:2409.13665, 2024.
- Qiu, J. et al. **Pi-fusion: Physics-Informed Diffusion Model for Learning Fluid Dynamics.** arXiv:2406.03711, 2024.
