#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import torch


MODELS = {
    "TRA_SS_base": Path(
        "results/sampling_tra_baseline_10h_500step_metrics/Tralong_500step/"
        "acdm-r20-tra-baseline-10h-500step.npz"
    ),
    "TRA_SS_cont001": Path(
        "results/sampling_tra_cont001_10h_500step_metrics/Tralong_500step/"
        "acdm-r20-tra-cont001-10h-500step.npz"
    ),
    "TRA_SS_cont02": Path(
        "results/sampling_tra_cont02_10h_500step_metrics/Tralong_500step/"
        "acdm-r20-tra-cont02-10h-500step.npz"
    ),
}

GT_PATH = Path(
    "results/sampling_tra_baseline_10h_500step_metrics/"
    "Tralong_500step/groundTruth.dict"
)

OUT = Path("results/tra_ss_per_sequence")
OUT.mkdir(parents=True, exist_ok=True)



# Load data
gt_dict = torch.load(GT_PATH, map_location="cpu")
gt = gt_dict["data"]

if torch.is_tensor(gt):
    gt = gt.cpu().numpy()

# GT: (2, 500, 5, 128, 64)
# physical channels:
# 0=u, 1=v, 2=density, 3=pressure
gt = gt[:, :, 0:4]

print("GT shape:", gt.shape)

preds = {}

for name, path in MODELS.items():
    x = np.load(path)["arr_0"]

    # prediction:
    # (1, 1, 2, 500, 4, 128, 64)
    x = x[0, 0]

    preds[name] = x
    print(name, x.shape)



# Helpers

def mse(a, b):
    return float(np.mean((a - b) ** 2))


def rel_l2(a, b):
    num = np.sqrt(np.sum((a - b) ** 2))
    den = np.sqrt(np.sum(b ** 2)) + 1e-12
    return float(num / den)


def vorticity(x):
    """
    x: T,C,X,Y
    channel 0=u, 1=v
    omega = dv/dx - du/dy
    """
    u = x[:, 0]
    v = x[:, 1]

    dv_dx = np.gradient(v, axis=1)
    du_dy = np.gradient(u, axis=2)

    return dv_dx - du_dy


def pressure_gradient(x):
    """
    x: T,C,X,Y
    pressure channel = 3
    returns gradient magnitude
    """
    p = x[:, 3]

    dp_dx = np.gradient(p, axis=1)
    dp_dy = np.gradient(p, axis=2)

    return np.sqrt(dp_dx ** 2 + dp_dy ** 2)


def density_gradient(x):
    """
    x: T,C,X,Y
    density channel = 2
    returns gradient magnitude
    """
    rho = x[:, 2]

    drho_dx = np.gradient(rho, axis=1)
    drho_dy = np.gradient(rho, axis=2)

    return np.sqrt(drho_dx ** 2 + drho_dy ** 2)


def continuity_residual(x):
    """
    Discrete continuity residual using the same simplified convention:

        drho/dt + d(rho*u)/dx + d(rho*v)/dy

    dx = dy = dt = 1

    x: T,C,X,Y
    """

    u = x[:, 0]
    v = x[:, 1]
    rho = x[:, 2]

    # t -> t+1 residual, so length T-1
    rho0 = rho[:-1]
    rho1 = rho[1:]

    u0 = u[:-1]
    u1 = u[1:]

    v0 = v[:-1]
    v1 = v[1:]

    drho_dt = rho1 - rho0

    rho_mid = 0.5 * (rho0 + rho1)
    u_mid = 0.5 * (u0 + u1)
    v_mid = 0.5 * (v0 + v1)

    flux_x = rho_mid * u_mid
    flux_y = rho_mid * v_mid

    dflux_dx = np.gradient(flux_x, axis=1)
    dflux_dy = np.gradient(flux_y, axis=2)

    return drho_dt + dflux_dx + dflux_dy



# Per-sequence metrics

rows = []

for seq_idx in range(gt.shape[0]):

    gt_seq = gt[seq_idx]

    print("\n" + "=" * 80)
    print(f"SEQUENCE {seq_idx}")
    print("=" * 80)

    for model, pred_all in preds.items():

        pred = pred_all[seq_idx]

        pred_vort = vorticity(pred)
        gt_vort = vorticity(gt_seq)

        pred_pg = pressure_gradient(pred)
        gt_pg = pressure_gradient(gt_seq)

        pred_rg = density_gradient(pred)
        gt_rg = density_gradient(gt_seq)

        pred_cont = continuity_residual(pred)

        row = {
            "sequence": f"seq{seq_idx}",
            "model": model,

            "predMSE": mse(pred, gt_seq),
            "relL2": rel_l2(pred, gt_seq),

            "predContinuityMSE":
                float(np.mean(pred_cont ** 2)),

            "vorticityMSE":
                mse(pred_vort, gt_vort),

            "pressureGradientMSE":
                mse(pred_pg, gt_pg),

            "densityGradientMSE":
                mse(pred_rg, gt_rg),
        }

        rows.append(row)

        print(
            f"{model:16s} "
            f"predMSE={row['predMSE']:.6e}  "
            f"relL2={row['relL2']:.6f}  "
            f"contMSE={row['predContinuityMSE']:.6e}  "
            f"vortMSE={row['vorticityMSE']:.6e}  "
            f"pGradMSE={row['pressureGradientMSE']:.6e}  "
            f"rhoGradMSE={row['densityGradientMSE']:.6e}"
        )


df = pd.DataFrame(rows)

out_long = OUT / "TRA_SS_per_sequence_metrics.csv"
df.to_csv(out_long, index=False)

print("\nSaved:", out_long)



# Wide tables for easier reading

for seq in ["seq0", "seq1"]:
    d = df[df["sequence"] == seq].copy()

    wide = d.set_index("model")[
        [
            "predMSE",
            "relL2",
            "predContinuityMSE",
            "vorticityMSE",
            "pressureGradientMSE",
            "densityGradientMSE",
        ]
    ]

    out = OUT / f"{seq}_comparison.csv"
    wide.to_csv(out)

    print("\n", seq)
    print(wide.to_string())
    print("Saved:", out)

