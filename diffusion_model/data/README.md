# Data

The datasets used in this project are not included in this repository because of their size.

Two two-dimensional cylinder-flow datasets are used in the final experiments:

- `128_inc`: incompressible viscous flow around a cylinder. Each flow state contains pressure and two velocity components, `u` and `v`. Reynolds number is used as the physical conditioning parameter.
- `128_tra`: transonic compressible flow around a cylinder. Each flow state contains pressure, density, and two velocity components, `u` and `v`. Mach number is used as the physical conditioning parameter.

Both datasets are represented on a `128 × 64` spatial grid and are derived from the fluid-flow benchmark dataset of Kohl, Chen and Thuerey.

## Dataset source

The datasets are derived from the fluid-flow benchmark dataset introduced by Kohl, Chen and Thuerey.

The original dataset can be obtained from:

- [Official dataset download page](https://mediatum.ub.tum.de/1734798)

For further details on dataset generation, simulation settings, and benchmark configuration, see the corresponding reference:

- Kohl, Chen and Thuerey, *Benchmarking Autoregressive Conditional Diffusion Models for Turbulent Flow Simulation*.

## Expected directory structure

The training and evaluation scripts expect the datasets to be placed under:

```text
data/
├── 128_inc/
└── 128_tra/
```

The datasets must be available in these locations before running the supplied training or sampling scripts.

## Evaluation subsets

For the final incompressible-flow evaluation, results are reported separately for low- and high-Reynolds-number test cases.

For the final transonic-flow evaluation, the reported long-rollout results correspond to the `Tralong_500step` evaluation used by the supplied TRA sampling scripts.

## Data preparation utilities

The repository includes preprocessing utilities used during dataset preparation:

- `src/copy_data_lowres.py` for constructing lower-resolution datasets;
- `src/convert_SU2_structure.py` for converting and organising the transonic SU2 dataset structure;
- `src/compute_data_mean_std.py` for computing dataset statistics used during preprocessing and normalisation.

Large raw simulation files and processed dataset arrays are intentionally excluded from Git.
