#!/usr/bin/env python3

from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



# Configuration

MODELS = {
    "TRA_SS_base": {
        "folder": Path(
            "results/sampling_tra_baseline_10h_500step_metrics/"
            "Tralong_500step"
        ),
        "file": "acdm-r20-tra-baseline-10h-500step.npz",
    },

    "TRA_SS_cont02": {
        "folder": Path(
            "results/sampling_tra_cont02_10h_500step_metrics/"
            "Tralong_500step"
        ),
        "file": "acdm-r20-tra-cont02-10h-500step.npz",
    },

    "TRA_MS_P2_cont02": {
        "folder": Path(
            "results/sampling_tra_ms_p2_cont02_fixed_10h_500step_metrics/"
            "Tralong_500step"
        ),
        "file": "acdm-r20-tra-ms-p2-cont02-fixed-10h-500step.npz",
    },
}

GT_PATH = (
    MODELS["TRA_SS_base"]["folder"] /
    "groundTruth.dict"
)

OUT = Path("results/TRA_wake_region_diagnostic")
OUT.mkdir(parents=True, exist_ok=True)


# Wake ROI
#
# Grid = 128 x 64
#
# Default:
#   x = 20 ... 127
#   y = full domain
#
# Can be overridden from shell.


WAKE_X0 = int(os.environ.get("WAKE_X0", "20"))
WAKE_X1 = int(os.environ.get("WAKE_X1", "128"))

WAKE_Y0 = int(os.environ.get("WAKE_Y0", "0"))
WAKE_Y1 = int(os.environ.get("WAKE_Y1", "64"))


print("Wake ROI:")
print(
    f"  x = [{WAKE_X0}, {WAKE_X1})"
)
print(
    f"  y = [{WAKE_Y0}, {WAKE_Y1})"
)



# Load data

gt_dict = torch.load(
    GT_PATH,
    map_location="cpu"
)

gt = gt_dict["data"]

if torch.is_tensor(gt):
    gt = gt.detach().cpu().numpy()

# GT:
# (2,500,5,128,64)
#
# Physical prediction channels:
# 0 = u
# 1 = v
# 2 = density
# 3 = pressure
#
# channel 4 = Mach / conditioning
gt = gt[:, :, 0:4]

print("GT:", gt.shape)


preds = {}

for model, cfg in MODELS.items():

    path = cfg["folder"] / cfg["file"]

    x = np.load(path)["arr_0"]

    # Saved prediction:
    # (1,1,2,500,4,128,64)
    x = x[0, 0]

    preds[model] = x

    print(model, x.shape)

    if x.shape != gt.shape:
        raise RuntimeError(
            f"Shape mismatch: GT={gt.shape}, "
            f"{model}={x.shape}"
        )


# Metric helpers

FIELD_INDEX = {
    "u": 0,
    "v": 1,
    "density": 2,
    "pressure": 3,
}


def mse(pred, truth):
    return float(
        np.mean(
            (pred - truth) ** 2
        )
    )


def rel_l2(pred, truth):
    numerator = np.sqrt(
        np.sum(
            (pred - truth) ** 2
        )
    )

    denominator = np.sqrt(
        np.sum(
            truth ** 2
        )
    ) + 1e-12

    return float(
        numerator / denominator
    )


def vorticity(x):
    """
    x:
      T,C,X,Y

    Compute on FULL domain before cropping,
    to avoid artificial derivative boundaries at ROI edges.

    omega = dv/dx - du/dy
    """

    u = x[:, 0]
    v = x[:, 1]

    dv_dx = np.gradient(
        v,
        axis=1
    )

    du_dy = np.gradient(
        u,
        axis=2
    )

    return (
        dv_dx - du_dy
    )


def crop_state(x):
    """
    x:
      T,C,X,Y
    """
    return x[
        :,
        :,
        WAKE_X0:WAKE_X1,
        WAKE_Y0:WAKE_Y1
    ]


def crop_scalar(x):
    """
    x:
      T,X,Y
    """
    return x[
        :,
        WAKE_X0:WAKE_X1,
        WAKE_Y0:WAKE_Y1
    ]


def timestep_mse(pred, truth):
    """
    pred/truth:
      T,...
    """
    axes = tuple(
        range(
            1,
            pred.ndim
        )
    )

    return np.mean(
        (pred - truth) ** 2,
        axis=axes
    )


def timestep_rel_l2(pred, truth):
    """
    relL2 independently at every timestep
    """

    axes = tuple(
        range(
            1,
            pred.ndim
        )
    )

    num = np.sqrt(
        np.sum(
            (pred - truth) ** 2,
            axis=axes
        )
    )

    den = np.sqrt(
        np.sum(
            truth ** 2,
            axis=axes
        )
    ) + 1e-12

    return num / den



# Per-sequence + per-region metrics

rows = []

curve_rows = []


for seq in range(gt.shape[0]):

    gt_full = gt[seq]

    gt_wake = crop_state(
        gt_full
    )

    gt_vort_full = vorticity(
        gt_full
    )

    gt_vort_wake = crop_scalar(
        gt_vort_full
    )

    for model, pred_all in preds.items():

        pred_full = pred_all[seq]

        pred_wake = crop_state(
            pred_full
        )

        pred_vort_full = vorticity(
            pred_full
        )

        pred_vort_wake = crop_scalar(
            pred_vort_full
        )


        # Evaluate BOTH full domain and wake region

        for region in [
            "full",
            "wake"
        ]:

            if region == "full":

                p = pred_full
                g = gt_full

                pv = pred_vort_full
                gv = gt_vort_full

            else:

                p = pred_wake
                g = gt_wake

                pv = pred_vort_wake
                gv = gt_vort_wake


            row = {
                "sequence": f"seq{seq}",
                "model": model,
                "region": region,

                "MSE": mse(
                    p,
                    g
                ),

                "relL2": rel_l2(
                    p,
                    g
                ),

                "vorticityMSE": mse(
                    pv,
                    gv
                ),

                "vorticityRelL2": rel_l2(
                    pv,
                    gv
                ),
            }


            # Field-wise errors
            for field, idx in FIELD_INDEX.items():

                row[
                    f"MSE_{field}"
                ] = mse(
                    p[:, idx],
                    g[:, idx]
                )

                row[
                    f"relL2_{field}"
                ] = rel_l2(
                    p[:, idx],
                    g[:, idx]
                )


            rows.append(row)


    
        # Wake-region curves


        wake_mse_t = timestep_mse(
            pred_wake,
            gt_wake
        )

        wake_rel_t = timestep_rel_l2(
            pred_wake,
            gt_wake
        )

        wake_vort_mse_t = timestep_mse(
            pred_vort_wake,
            gt_vort_wake
        )

        wake_vort_rel_t = timestep_rel_l2(
            pred_vort_wake,
            gt_vort_wake
        )


        for t in range(
            len(wake_mse_t)
        ):

            curve_rows.append(
                {
                    "sequence": f"seq{seq}",
                    "model": model,
                    "t": t,

                    "wakeMSE":
                        wake_mse_t[t],

                    "wakeRelL2":
                        wake_rel_t[t],

                    "wakeVorticityMSE":
                        wake_vort_mse_t[t],

                    "wakeVorticityRelL2":
                        wake_vort_rel_t[t],
                }
            )



# Save raw results

df = pd.DataFrame(
    rows
)

curves = pd.DataFrame(
    curve_rows
)


df.to_csv(
    OUT /
    "TRA_full_vs_wake_per_sequence.csv",
    index=False
)

curves.to_csv(
    OUT /
    "TRA_wake_curves_per_sequence.csv",
    index=False
)



# Mean across two sequences

numeric_cols = [
    c
    for c in df.columns
    if c not in [
        "sequence",
        "model",
        "region"
    ]
]


mean_df = (
    df
    .groupby(
        [
            "model",
            "region"
        ],
        as_index=False
    )[numeric_cols]
    .mean()
)


mean_df.to_csv(
    OUT /
    "TRA_full_vs_wake_mean.csv",
    index=False
)


print()
print("=" * 100)
print("MEAN ACROSS BOTH SEQUENCES")
print("=" * 100)

print(
    mean_df[
        [
            "model",
            "region",
            "MSE",
            "relL2",
            "MSE_u",
            "MSE_v",
            "MSE_density",
            "MSE_pressure",
            "vorticityMSE",
            "vorticityRelL2",
        ]
    ].to_string(
        index=False
    )
)



# Wake-only table

wake_mean = mean_df[
    mean_df["region"] == "wake"
].copy()

wake_mean.to_csv(
    OUT /
    "TRA_wake_mean_comparison.csv",
    index=False
)



# Relative change:
# MS P2 vs SS cont02


ss = wake_mean[
    wake_mean["model"]
    ==
    "TRA_SS_cont02"
].iloc[0]

ms = wake_mean[
    wake_mean["model"]
    ==
    "TRA_MS_P2_cont02"
].iloc[0]


comparison_metrics = [
    "MSE",
    "relL2",
    "MSE_u",
    "MSE_v",
    "MSE_density",
    "MSE_pressure",
    "relL2_u",
    "relL2_v",
    "relL2_density",
    "relL2_pressure",
    "vorticityMSE",
    "vorticityRelL2",
]


change_rows = []

for metric in comparison_metrics:

    ss_value = float(
        ss[metric]
    )

    ms_value = float(
        ms[metric]
    )

    change = (
        100.0
        *
        (ms_value - ss_value)
        /
        (ss_value + 1e-30)
    )

    change_rows.append(
        {
            "metric": metric,
            "TRA_SS_cont02":
                ss_value,
            "TRA_MS_P2_cont02":
                ms_value,
            "P2_change_percent":
                change,
        }
    )


change_df = pd.DataFrame(
    change_rows
)

change_df.to_csv(
    OUT /
    "TRA_wake_P2_vs_SS_cont02.csv",
    index=False
)


print()
print("=" * 100)
print("WAKE REGION: P2 vs SS_cont02")
print("Negative percentage = P2 lower error / improvement")
print("=" * 100)

print(
    change_df.to_string(
        index=False
    )
)



# Curves averaged over sequences

curve_mean = (
    curves
    .groupby(
        [
            "model",
            "t"
        ],
        as_index=False
    )[
        [
            "wakeMSE",
            "wakeRelL2",
            "wakeVorticityMSE",
            "wakeVorticityRelL2",
        ]
    ]
    .mean()
)


curve_mean.to_csv(
    OUT /
    "TRA_wake_curves_mean.csv",
    index=False
)


def plot_curve(
    metric,
    ylabel,
    title,
    filename
):

    plt.figure(
        figsize=(8.5, 5.3)
    )

    for model in MODELS:

        d = curve_mean[
            curve_mean["model"]
            ==
            model
        ]

        plt.plot(
            d["t"],
            d[metric],
            label=model,
            linewidth=1.8,
        )

    plt.xlabel(
        "Rollout step"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT / filename,
        dpi=300
    )

    plt.savefig(
        OUT /
        filename.replace(
            ".png",
            ".pdf"
        )
    )

    plt.close()


plot_curve(
    "wakeMSE",
    "Wake-region MSE",
    "TRA 500-step rollout: Wake-region MSE",
    "TRA_wake_MSE_vs_rollout.png",
)

plot_curve(
    "wakeRelL2",
    "Wake-region Relative L2",
    "TRA 500-step rollout: Wake-region Relative L2",
    "TRA_wake_relL2_vs_rollout.png",
)

plot_curve(
    "wakeVorticityMSE",
    "Wake-region Vorticity MSE",
    "TRA 500-step rollout: Wake Vorticity MSE",
    "TRA_wake_vorticity_MSE_vs_rollout.png",
)

plot_curve(
    "wakeVorticityRelL2",
    "Wake-region Vorticity Relative L2",
    "TRA 500-step rollout: Wake Vorticity Relative L2",
    "TRA_wake_vorticity_relL2_vs_rollout.png",
)


print()
print("=" * 100)
print("WAKE REGION DIAGNOSTIC COMPLETE")
print("Output:", OUT)
print("=" * 100)
