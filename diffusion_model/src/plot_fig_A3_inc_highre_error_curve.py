#!/usr/bin/env python3

import os
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = "results/final_figures"

OUT_PNG = os.path.join(
    OUT_DIR,
    "fig_A3_inc_highre_error_evolution.png",
)

OUT_PDF = os.path.join(
    OUT_DIR,
    "fig_A3_inc_highre_error_evolution.pdf",
)


FILES = {
    "SS baseline":
        "results/ablation_diffonly_500step/highRey/"
        "ablation-diffusion-only-500step_mse_per_timestep.csv",

    "SS + vort.":
        "results/ablation_vort_500step/highRey/"
        "ablation-vorticity-500step_mse_per_timestep.csv",

    r"MS ($C=P=2$) + vort.":
        "results/sampling_inc_ms_p2_vor001_500step/highRey/"
        "acdm-r20-inc-ms-p2-vor001-500step_mse_per_timestep.csv",

    r"MS ($C=P=2$) + vort. + temp.":
        "results/sampling_inc_ms_p2_vor001_thw_500step/highRey/"
        "acdm-r20-inc-ms-p2-vor001-thw-500step_mse_per_timestep.csv",
}


def load_curve(path):
    df = pd.read_csv(path)

    if len(df.columns) < 2:
        raise ValueError(
            f"Expected at least two columns in {path}, got {df.columns}"
        )

    t = df.iloc[:, 0].to_numpy()
    mse = df.iloc[:, 1].to_numpy()

    return t, mse


def main():

    os.makedirs(
        OUT_DIR,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(8.6, 5.2),
        constrained_layout=True,
    )

    for label, path in FILES.items():

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        t, mse = load_curve(path)

        print(
            f"{label:35s} "
            f"mean={mse.mean():.6e} "
            f"final={mse[-1]:.6e} "
            f"max={mse.max():.6e}"
        )

        ax.plot(
            t,
            mse,
            linewidth=1.8,
            label=label,
        )

    ax.set_xlabel(
        "Rollout timestep",
        fontsize=11,
    )

    ax.set_ylabel(
        "Prediction MSE",
        fontsize=11,
    )

    ax.set_xlim(
        0,
        499,
    )

    ax.grid(
        True,
        alpha=0.25,
        linewidth=0.6,
    )

    ax.legend(
        fontsize=9,
        frameon=False,
    )

    fig.savefig(
        OUT_PNG,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        OUT_PDF,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print("Saved:")
    print(" ", OUT_PNG)
    print(" ", OUT_PDF)


if __name__ == "__main__":
    main()
