#!/usr/bin/env python3

import os
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEQ_INDEX = 0
PRESSURE_CHANNEL = 3
FIELD_TIMESTEPS = [0, 250, 499]
ERROR_FIELD_TIMESTEPS = [0, 250, 499]
ERROR_TIMESTEPS = [10, 250, 499]

GT_PATH = (
    "results/sampling_tra_cont02_10h_500step_metrics/"
    "Tralong_500step/groundTruth.dict"
)

MODEL_PATHS = {
    "SS + cont.": (
        "results/sampling_tra_cont02_10h_500step_metrics/"
        "Tralong_500step/"
        "acdm-r20-tra-cont02-10h-500step.npz"
    ),
    r"MS ($C=P=2$) + cont.": (
        "results/sampling_tra_ms_p2_cont02_fixed_10h_500step_metrics/"
        "Tralong_500step/"
        "acdm-r20-tra-ms-p2-cont02-fixed-10h-500step.npz"
    ),
    r"MS ($C=P=4$) + cont.": (
        "results/sampling_tra_ms_p4_cont02_full_10h_500step_metrics/"
        "Tralong_500step/"
        "acdm-r20-tra-ms-p4-cont02-full-10h-500step.npz"
    ),
}

OUT_DIR = "results/final_figures"

OUT_FIELD_PNG = os.path.join(
    OUT_DIR, "fig_A2a_tra_pressure_spatial.png"
)
OUT_FIELD_PDF = os.path.join(
    OUT_DIR, "fig_A2a_tra_pressure_spatial.pdf"
)

OUT_ERR_PNG = os.path.join(
    OUT_DIR, "fig_A2b_tra_pressure_error.png"
)
OUT_ERR_PDF = os.path.join(
    OUT_DIR, "fig_A2b_tra_pressure_error.pdf"
)


def load_gt():
    x = torch.load(GT_PATH, map_location="cpu")
    data = x["data"]

    if torch.is_tensor(data):
        data = data.cpu().numpy()

    return data[SEQ_INDEX]


def load_prediction(path):
    x = np.load(path)["arr_0"]
    return x[0, 0, SEQ_INDEX]


def clean_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def main():

    os.makedirs(OUT_DIR, exist_ok=True)

    gt = load_gt()

    preds = {
        name: load_prediction(path)
        for name, path in MODEL_PATHS.items()
    }

    gt_p = gt[:, PRESSURE_CHANNEL]

    pred_p = {
        name: x[:, PRESSURE_CHANNEL]
        for name, x in preds.items()
    }

    errors = {
        name: np.abs(x - gt_p)
        for name, x in pred_p.items()
    }


    # Shared pressure scale

    pressure_display = []

    for t in FIELD_TIMESTEPS:
        pressure_display.append(gt_p[t])

        for field in pred_p.values():
            pressure_display.append(field[t])

    pmin = min(np.nanmin(x) for x in pressure_display)
    pmax = max(np.nanmax(x) for x in pressure_display)

    print("Pressure range:", pmin, pmax)


    # Shared error scale

    all_err = np.concatenate([
        errors[name][ERROR_TIMESTEPS].ravel()
        for name in errors
    ])

    # Robust common visual range.
    err_max = np.nanpercentile(all_err, 99.0)

    print("Error vmax (99th percentile):", err_max)

  
    # A2(a): pressure fields

    columns = [
        "Ground truth",
        "SS + cont.",
        r"MS ($C=P=2$) + cont.",
        r"MS ($C=P=4$) + cont.",
    ]

    fields = [
        gt_p,
        pred_p["SS + cont."],
        pred_p[r"MS ($C=P=2$) + cont."],
        pred_p[r"MS ($C=P=4$) + cont."],
    ]

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(14.5, 5.9),
        constrained_layout=True,
    )

    im = None

    for r, t in enumerate(FIELD_TIMESTEPS):

        for c, field in enumerate(fields):

            ax = axes[r, c]

            im = ax.imshow(
                field[t].T,
                origin="lower",
                cmap="viridis",
                vmin=pmin,
                vmax=pmax,
                aspect="equal",
                interpolation="none",
            )

            clean_axis(ax)

            if r == 0:
                ax.set_title(
                    columns[c],
                    fontsize=12,
                    pad=6,
                )

            if c == 0:
                ax.set_ylabel(
                    f"$t={t}$",
                    fontsize=11,
                )

    cbar = fig.colorbar(
        im,
        ax=axes,
        shrink=0.88,
        pad=0.012,
    )

    cbar.set_label(
        "Pressure",
        fontsize=11,
    )

    fig.savefig(
        OUT_FIELD_PNG,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        OUT_FIELD_PDF,
        bbox_inches="tight",
    )

    plt.close(fig)


    # A2(b): absolute errors

    err_columns = [
        "SS + cont.",
        r"MS ($C=P=2$) + cont.",
        r"MS ($C=P=4$) + cont.",
    ]

    err_fields = [
        errors["SS + cont."],
        errors[r"MS ($C=P=2$) + cont."],
        errors[r"MS ($C=P=4$) + cont."],
    ]

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(11.2, 5.9),
        constrained_layout=True,
    )

    im = None

    for r, t in enumerate(ERROR_TIMESTEPS):

        for c, field in enumerate(err_fields):

            ax = axes[r, c]

            im = ax.imshow(
                field[t].T,
                origin="lower",
                cmap="magma",
                vmin=0.0,
                vmax=err_max,
                aspect="equal",
                interpolation="none",
            )

            clean_axis(ax)

            if r == 0:
                ax.set_title(
                    err_columns[c],
                    fontsize=12,
                    pad=6,
                )

            if c == 0:
                ax.set_ylabel(
                    f"$t={t}$",
                    fontsize=11,
                )

    cbar = fig.colorbar(
        im,
        ax=axes,
        shrink=0.88,
        pad=0.012,
    )

    cbar.set_label(
        "Absolute pressure error",
        fontsize=11,
    )

    fig.savefig(
        OUT_ERR_PNG,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        OUT_ERR_PDF,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print("Saved:")
    print(" ", OUT_FIELD_PNG)
    print(" ", OUT_FIELD_PDF)
    print(" ", OUT_ERR_PNG)
    print(" ", OUT_ERR_PDF)


if __name__ == "__main__":
    main()
