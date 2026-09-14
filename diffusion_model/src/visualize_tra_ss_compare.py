#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation


ROOTS = {
    "TRA_SS_base":
        Path("results/sampling_tra_baseline_10h_500step_metrics/Tralong_500step"),

    "TRA_SS_cont001":
        Path("results/sampling_tra_cont001_10h_500step_metrics/Tralong_500step"),

    "TRA_SS_cont02":
        Path("results/sampling_tra_cont02_10h_500step_metrics/Tralong_500step"),
}

NPZ = {
    "TRA_SS_base":
        "acdm-r20-tra-baseline-10h-500step.npz",

    "TRA_SS_cont001":
        "acdm-r20-tra-cont001-10h-500step.npz",

    "TRA_SS_cont02":
        "acdm-r20-tra-cont02-10h-500step.npz",
}

OUT = Path("results/visualization_TRA_SS")
OUT.mkdir(parents=True, exist_ok=True)

# One of the two Tralong sequences.
SEQ_IDX = int(os.environ.get("TRA_VIS_SEQ_IDX", "0"))

# Static frames.
STATIC_STEPS = [99, 299, 499]

# Whole 500-step GIF can be heavy.
# GIF_STRIDE=2 gives 250 frames.
GIF_STRIDE = int(os.environ.get("TRA_VIS_GIF_STRIDE", "2"))

# Optional wake crop.
# Current grid = 128 x 64.
X0 = int(os.environ.get("TRA_VIS_X0", "0"))
X1 = int(os.environ.get("TRA_VIS_X1", "128"))
Y0 = int(os.environ.get("TRA_VIS_Y0", "0"))
Y1 = int(os.environ.get("TRA_VIS_Y1", "64"))

# TRA channel convention established by sampling:
# 0=u, 1=v, 2=density, 3=pressure
CHANNELS = {
    "u": 0,
    "v": 1,
    "density": 2,
    "pressure": 3,
}

FIELDS = ["u", "v", "density", "pressure", "vort"]


def load_gt():
    path = ROOTS["TRA_SS_base"] / "groundTruth.dict"
    d = torch.load(path, map_location="cpu")

    x = d["data"]

    print("Raw GT shape:", tuple(x.shape))

    # Current TRA saved groundTruth:
    # (num_sequences, 500, 5, 128, 64)
    #
    # channel 4 is Mach conditioning, not a predicted physical field.
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    if x.ndim != 5:
        raise RuntimeError(
            f"Unexpected ground truth shape {x.shape}. "
            "Expected N,T,C,H,W."
        )

    return x[SEQ_IDX, :, 0:4]


def load_prediction(model):
    path = ROOTS[model] / NPZ[model]
    x = np.load(path)["arr_0"]

    print(model, "raw prediction shape:", x.shape)

    # Current TRA prediction shape:
    # (1, 1, 2, 500, 4, 128, 64)
    #
    # model, eval, sequence, time, channels, x, y
    if x.ndim != 7:
        raise RuntimeError(
            f"Unexpected prediction shape {x.shape} for {model}."
        )

    return x[0, 0, SEQ_IDX]


def crop(x):
    # x: T,C,H,W
    return x[:, :, X0:X1, Y0:Y1]


def vort(x):
    """
    x: T,C,H,W
    channels 0=u, 1=v
    omega = dv/dx - du/dy
    """
    u = x[:, 0]
    v = x[:, 1]

    dv_dx = np.gradient(v, axis=1)
    du_dy = np.gradient(u, axis=2)

    return dv_dx - du_dy


def field_array(x, field):
    if field == "vort":
        return vort(x)

    return x[:, CHANNELS[field]]


def robust_limits(arrays, field):
    """
    Use one common colour scale across GT + all models.
    Percentiles avoid a few extreme pixels dominating the scale.
    """

    vals = np.concatenate(
        [a.reshape(-1) for a in arrays]
    )

    if field in ["vort", "u", "v"]:
        q = np.percentile(np.abs(vals), 99.5)
        if q <= 0:
            q = 1.0
        return -q, q

    lo = np.percentile(vals, 0.5)
    hi = np.percentile(vals, 99.5)

    if abs(hi - lo) < 1e-12:
        hi = lo + 1.0

    return lo, hi


def cmap_for(field):
    if field in ["vort", "u", "v"]:
        return "RdBu_r"
    return "viridis"


gt = crop(load_gt())

preds = {
    model: crop(load_prediction(model))
    for model in ROOTS
}

# Check all shapes match.
print("\nProcessed shapes:")
print("GT:", gt.shape)

for model, x in preds.items():
    print(model, x.shape)

    if x.shape != gt.shape:
        raise RuntimeError(
            f"Shape mismatch: GT={gt.shape}, {model}={x.shape}"
        )


for field in FIELDS:

    print("\n========================================")
    print("FIELD:", field)
    print("========================================")

    arrays = [
        field_array(gt, field),
        field_array(preds["TRA_SS_base"], field),
        field_array(preds["TRA_SS_cont001"], field),
        field_array(preds["TRA_SS_cont02"], field),
    ]

    names = [
        "Ground truth",
        "TRA_SS_base",
        "TRA_SS_cont001",
        "TRA_SS_cont02",
    ]

    vmin, vmax = robust_limits(arrays, field)
    cmap = cmap_for(field)

    print("colour scale:", vmin, vmax)


    # Static snapshots
    # rows = GT/base/cont001/cont02
    # cols = selected timesteps
    fig, axs = plt.subplots(
        nrows=4,
        ncols=len(STATIC_STEPS),
        figsize=(12, 8),
        dpi=180,
        squeeze=False,
    )

    im = None

    for i, (name, arr) in enumerate(zip(names, arrays)):
        for j, t in enumerate(STATIC_STEPS):

            ax = axs[i, j]

            im = ax.imshow(
                arr[t].T,
                origin="lower",
                interpolation="nearest",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )

            if i == 0:
                ax.set_title(f"t = {t + 1}")

            if j == 0:
                ax.set_ylabel(name)

            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"TRA Single-Step Comparison — {field}",
        fontsize=15,
    )

    fig.tight_layout(rect=[0, 0, 0.91, 0.96])

    cax = fig.add_axes([0.925, 0.08, 0.018, 0.82])
    fig.colorbar(im, cax=cax)

    png = OUT / f"TRA_SS_seq{SEQ_IDX}_compare_{field}_snapshots.png"
    pdf = OUT / f"TRA_SS_seq{SEQ_IDX}_compare_{field}_snapshots.pdf"

    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)

    print("saved", png)
    print("saved", pdf)


    # GIF
    # One row, four columns.
    fig, axs = plt.subplots(
        nrows=1,
        ncols=4,
        figsize=(13, 3.4),
        dpi=110,
        squeeze=False,
    )

    axs = axs[0]

    ims = []

    for i, (name, arr) in enumerate(zip(names, arrays)):
        axs[i].set_title(name)
        axs[i].set_xticks([])
        axs[i].set_yticks([])

        im = axs[i].imshow(
            arr[0].T,
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )

        ims.append(im)

    time_text = fig.text(
        0.5,
        0.02,
        "t = 1",
        ha="center",
        fontsize=11,
    )

    fig.suptitle(
        f"TRA Single-Step Comparison — {field}",
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0.06, 0.93, 0.93])

    cax = fig.add_axes([0.94, 0.15, 0.012, 0.68])
    fig.colorbar(ims[-1], cax=cax)

    frame_ids = list(range(0, gt.shape[0], GIF_STRIDE))

    def update(k):
        t = frame_ids[k]

        for im, arr in zip(ims, arrays):
            im.set_data(arr[t].T)

        time_text.set_text(f"t = {t + 1}")

        return ims + [time_text]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_ids),
        interval=80,
        blit=False,
    )

    gif = OUT / f"TRA_SS_seq{SEQ_IDX}_compare_{field}.gif"

    anim.save(
        gif,
        writer=animation.PillowWriter(fps=12),
    )

    plt.close(fig)

    print("saved", gif)


print("\nAll TRA visualisations complete.")
print("Output folder:", OUT)
