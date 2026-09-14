#!/usr/bin/env python3

import os
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


"""
Fig. A1
INC High-Re spatial comparison

 Columns:
   GT / SS baseline / SS+vort / MS P2+vort

 Rows:
   t = 0, 250, 499

 Variable:
   vorticity = dv/dx - du/dy

 Channel convention:
   0 = u
   1 = v
   2 = pressure
   3 = Reynolds number
"""




SEQ_INDEX = 0
TIMESTEPS = [0, 250, 499]

GT_PATH = (
    "results/ablation_diffonly_500step/"
    "highRey/groundTruth.dict"
)

MODEL_PATHS = {
    "SS baseline": (
        "results/ablation_diffonly_500step/highRey/"
        "ablation-diffusion-only-500step.npz"
    ),
    "SS + vort.": (
        "results/ablation_vort_500step/highRey/"
        "ablation-vorticity-500step.npz"
    ),
    "MS P2 + vort.": (
        "results/sampling_inc_ms_p2_vor001_500step/highRey/"
        "acdm-r20-inc-ms-p2-vor001-500step.npz"
    ),
}

OUT_DIR = "results/final_figures"
OUT_PNG = os.path.join(
    OUT_DIR,
    "fig_A1_inc_highre_vorticity.png",
)
OUT_PDF = os.path.join(
    OUT_DIR,
    "fig_A1_inc_highre_vorticity.pdf",
)


def vorticity(field):
    """
    field shape:
        T, C, W, H

    Returns:
        T, W, H
    """

    x = torch.as_tensor(field, dtype=torch.float32)

    # Add batch dimension:
    # B, S, C, W, H
    x = x.unsqueeze(0)

    dv_dx, _ = torch.gradient(
        x[:, :, 1:2],
        dim=(3, 4),
    )

    _, du_dy = torch.gradient(
        x[:, :, 0:1],
        dim=(3, 4),
    )

    vort = dv_dx - du_dy

    return vort[0, :, 0].cpu().numpy()


def load_gt():
    obj = torch.load(
        GT_PATH,
        map_location="cpu",
    )

    data = obj["data"]

    if torch.is_tensor(data):
        data = data.cpu().numpy()

    # shape:
    # seq, time, channel, W, H
    return data[SEQ_INDEX]


def load_prediction(path):
    arr = np.load(path)["arr_0"]

    # shape:
    # model, eval, seq, time, channel, W, H

    return arr[
        0,
        0,
        SEQ_INDEX,
    ]


def main():

    os.makedirs(
        OUT_DIR,
        exist_ok=True,
    )

    gt = load_gt()

    predictions = {
        name: load_prediction(path)
        for name, path in MODEL_PATHS.items()
    }

    print("GT shape:", gt.shape)

    for name, x in predictions.items():
        print(name, x.shape)

    gt_vort = vorticity(gt)

    pred_vort = {
        name: vorticity(x)
        for name, x in predictions.items()
    }

    # --------------------------------------------------------
    # Common symmetric colour scale.
    #
    # Use all displayed panels so every model / timestep
    # is directly comparable.
    # --------------------------------------------------------

    displayed = []

    for t in TIMESTEPS:
        displayed.append(gt_vort[t])

        for name in MODEL_PATHS:
            displayed.append(
                pred_vort[name][t]
            )

    vmax = max(
        np.nanmax(np.abs(x))
        for x in displayed
    )

    vmin = -vmax

    print(
        "Common vorticity range:",
        vmin,
        vmax,
    )

    # Plot

    column_names = [
        "Ground truth",
        "SS baseline",
        "SS + vort.",
        r"MS ($C=P=2$) + vort.",
    ]

    all_fields = [
        gt_vort,
        pred_vort["SS baseline"],
        pred_vort["SS + vort."],
        pred_vort["MS P2 + vort."],
    ]

    fig, axes = plt.subplots(
        nrows=len(TIMESTEPS),
        ncols=4,
        figsize=(14.5, 6.2),
        constrained_layout=True,
    )

    image = None

    for r, t in enumerate(TIMESTEPS):

        for c, field in enumerate(all_fields):

            ax = axes[r, c]

            image = ax.imshow(
                field[t].T,
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                cmap="RdBu_r",
                aspect="equal",
                interpolation="none",
            )

            ax.set_xticks([])
            ax.set_yticks([])

            if r == 0:
                ax.set_title(
                    column_names[c],
                    fontsize=11,
                )

            if c == 0:
                ax.set_ylabel(
                    f"t = {t}",
                    fontsize=11,
                )

    cbar = fig.colorbar(
        image,
        ax=axes,
        orientation="vertical",
        shrink=0.88,
        pad=0.015,
    )

    cbar.set_label(
        r"Vorticity $\omega$",
        fontsize=11,
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
