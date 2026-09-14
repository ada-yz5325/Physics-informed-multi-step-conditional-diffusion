#!/usr/bin/env python3

import os
import sys
import csv
import json
import math
import time
import copy
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

from turbpred.model import PredictionModel
from turbpred.model_diffusion import DiffusionModel
from turbpred.turbulence_dataset import TurbulenceDataset
from turbpred.data_transformations import Transforms


"""
Formal sampling + evaluation for Transonic ACDM models.

Default model:
    Single-step baseline, C=2, P=1
    runs/2D_Tra/128_acdm-r20_tra_baseline_10h_01/Model.pth

Physical fields:
    u, v, density, pressure

Condition:
    Mach number

Evaluation categories:
1. Accuracy
   - MSE
   - relative L2
   - per-field MSE / relative L2

2. Temporal
   - MSE per timestep
   - relative L2 per timestep
   - temporalIncrementMSE

3. Mass physics
   - discrete continuity residual on the resampled grid
   - predContinuityMAE / MSE
   - gtContinuityMAE / MSE
   - continuity residual per timestep

4. Flow structure
   - vorticity MSE
   - vorticity relative L2

5. Transonic structure
   - pressure-gradient MSE / relative L2
   - density-gradient MSE / relative L2

6. Efficiency
   - inference time
   - diffusion calls
   - reverse denoising steps

IMPORTANT:
The continuity residual here is a discrete comparison metric on the
resampled 128x64 grid. We use dx = dy = dt = 1 in index/rollout space.
It should not be interpreted as a dimensional PDE residual on the
original SU2 mesh.
"""


# Runtime configuration

device = "cuda" if torch.cuda.is_available() else "cpu"

if device != "cuda":
    raise RuntimeError(
        "CUDA is not available. Run this script inside a GPU PBS job."
    )

defaultDataRoot = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data")
)
dataRoot = os.environ.get("DATA_ROOT", defaultDataRoot)

outFolder = os.environ.get(
    "OUT_FOLDER",
    "results/sampling_tra_baseline_10h_metrics",
)

modelPath = os.environ.get(
    "MODEL_PATH",
    "runs/2D_Tra/128_acdm-r20_tra_baseline_10h_01/Model.pth",
)

modelLabel = os.environ.get(
    "MODEL_LABEL",
    "acdm-r20-tra-baseline-10h",
)

numEvals = int(os.environ.get("NUM_EVALS", "1"))
numWorkers = int(os.environ.get("NUM_WORKERS", "0"))

# For current baseline P=1.
# Later P=2 scripts can override OUTPUT_HORIZON=2.
outputHorizon = int(os.environ.get("OUTPUT_HORIZON", "1"))

reverseStepsPerCall = int(
    os.environ.get("REVERSE_STEPS_PER_CALL", "20")
)

datasetSelection = os.environ.get("DATASETS", "").strip()

# Discrete resampled-grid spacings.
DX = float(os.environ.get("PHYSICS_DX", "1.0"))
DY = float(os.environ.get("PHYSICS_DY", "1.0"))
DT = float(os.environ.get("PHYSICS_DT", "1.0"))


print("Runtime configuration:")
print("  device:", device)
print("  dataRoot:", dataRoot)
print("  outFolder:", outFolder)
print("  modelPath:", modelPath)
print("  modelLabel:", modelLabel)
print("  numEvals:", numEvals)
print("  outputHorizon:", outputHorizon)
print("  reverseStepsPerCall:", reverseStepsPerCall)
print("  physics spacing: dx =", DX, "dy =", DY, "dt =", DT)


if not os.path.isfile(modelPath):
    raise FileNotFoundError(
        f"Model not found:\n{modelPath}"
    )


# Test sets

testSpecs = {
    "Traext_500_620": {
        "name": "Test Traext Ma 0.50-0.52 frames 500-620",
        "sims": [0, 1, 2],
        "frame": (500, 620),
        "rollout": 60,
    },
    "Traext_620_740": {
        "name": "Test Traext Ma 0.50-0.52 frames 620-740",
        "sims": [0, 1, 2],
        "frame": (620, 740),
        "rollout": 60,
    },
    "Traint_500_620": {
        "name": "Test Traint Ma 0.66-0.68 frames 500-620",
        "sims": [16, 17, 18],
        "frame": (500, 620),
        "rollout": 60,
    },
    "Traint_620_740": {
        "name": "Test Traint Ma 0.66-0.68 frames 620-740",
        "sims": [16, 17, 18],
        "frame": (620, 740),
        "rollout": 60,
    },
    "Tralong_0_480": {
        "name": "Test Tralong Ma 0.64-0.65 frames 0-480",
        "sims": [14, 15],
        "frame": (0, 480),
        "rollout": 240,
    },
    "Tralong_480_960": {
        "name": "Test Tralong Ma 0.64-0.65 frames 480-960",
        "sims": [14, 15],
        "frame": (480, 960),
        "rollout": 240,
    },
    "Tralong_500step": {
        "name": "Test Tralong Ma 0.64-0.65 500-step rollout",
        "sims": [14, 15],
        "frame": (0, 1000),
        "rollout": 500,
    },
}


if datasetSelection:
    requested = [
        x.strip()
        for x in datasetSelection.split(",")
        if x.strip()
    ]

    unknown = [
        x for x in requested
        if x not in testSpecs
    ]

    if unknown:
        raise ValueError(
            f"Unknown DATASETS entries: {unknown}\n"
            f"Available: {list(testSpecs.keys())}"
        )

    testSpecs = {
        k: testSpecs[k]
        for k in requested
    }


testSets = {}

for shortName, spec in testSpecs.items():

    testSets[shortName] = TurbulenceDataset(
        spec["name"],
        [dataRoot],
        filterTop=["128_tra"],
        filterSim=[spec["sims"]],
        filterFrame=[spec["frame"]],
        sequenceLength=[[spec["rollout"], 2]],
        simFields=["dens", "pres"],
        simParams=["mach"],
        printLevel="sim",
    )


# Diffusion inference config

evalOptions = {
    "samplingMode": "ddpm",
    "posteriorSampling": "random",
    "initialSampling": "random",
    "conditioningIntegration": "noisy",
}


def configure_diffusion_decoder(model, options):

    if isinstance(model.modelDecoder, DiffusionModel):
        modules = [model.modelDecoder]

    elif isinstance(model.modelDecoder, torch.nn.ModuleList):
        modules = [
            m for m in model.modelDecoder
            if isinstance(m, DiffusionModel)
        ]

    else:
        modules = []

    for module in modules:
        module.inferenceSamplingMode = options["samplingMode"]
        module.inferencePosteriorSampling = options["posteriorSampling"]
        module.inferenceInitialSampling = options["initialSampling"]
        module.inferenceConditioningIntegration = options[
            "conditioningIntegration"
        ]


# Helpers

def safe_float(x):
    return float(np.asarray(x))


def relative_l2(pred, gt, mask=None):

    pred = np.asarray(pred)
    gt = np.asarray(gt)

    if mask is not None:
        pred = pred[mask]
        gt = gt[mask]

    numerator = np.sqrt(
        np.sum((pred - gt) ** 2)
    )

    denominator = np.sqrt(
        np.sum(gt ** 2)
    ) + 1e-12

    return numerator / denominator


def denormalize_predictions(predSamples, transformations):
    """
    Expected physical prediction channels:
        0 = u
        1 = v
        2 = density
        3 = pressure
    """

    if predSamples.shape[4] < 4:
        raise ValueError(
            "Expected at least 4 Transonic physical channels, "
            f"got prediction shape {predSamples.shape}"
        )

    fieldIndices = [0, 1, 2, 3]

    mean = np.asarray(
        transformations.normMean[fieldIndices],
        dtype=np.float32,
    )

    std = np.asarray(
        transformations.normStd[fieldIndices],
        dtype=np.float32,
    )

    mean = mean.reshape(
        1, 1, 1, 1, 4, 1, 1
    )

    std = std.reshape(
        1, 1, 1, 1, 4, 1, 1
    )

    predPhys = predSamples[..., :4, :, :]

    return predPhys * std + mean


def load_ground_truth(gtPath):

    sample = torch.load(
        gtPath,
        map_location="cpu",
    )

    data = sample["data"]

    if torch.is_tensor(data):
        data = data.cpu().numpy()

    obsMask = sample.get(
        "obsMask",
        None,
    )

    if torch.is_tensor(obsMask):
        obsMask = obsMask.cpu().numpy()

    # DataLoader collates one obstacle mask per sequence, so for
    # multiple test simulations the stored shape can be [N,H,W].
    # All 128_tra cases use the same cylinder geometry. Reduce the
    # batched masks to one common [H,W] mask after verifying equality.
    if obsMask is not None:
        obsMask = np.asarray(obsMask)

        if obsMask.ndim == 3:
            if not np.all(obsMask == obsMask[0:1]):
                raise ValueError(
                    "Obstacle masks differ between test sequences; "
                    "sequence-wise masking is required."
                )
            obsMask = obsMask[0]

        elif obsMask.ndim != 2:
            raise ValueError(
                f"Unexpected obstacle mask shape: {obsMask.shape}"
            )

    return np.asarray(data), obsMask


def align_prediction_and_ground_truth(predFull, gtData):

    pred = np.asarray(predFull)
    gt = np.asarray(gtData)

    if pred.ndim != 7:
        raise ValueError(
            f"Prediction should be 7D, got {pred.shape}"
        )

    if gt.ndim != 5:
        raise ValueError(
            f"Ground truth should be 5D, got {gt.shape}"
        )

    tPred = pred.shape[3]

    # prediction already denormalized and physical-only
    predPhys = pred[..., :4, :, :]

    # Ground truth order:
    # u, v, density, pressure, Mach
    gtPhys = gt[:, -tPred:, :4, :, :]

    gtPhys = gtPhys[
        np.newaxis,
        np.newaxis,
        ...
    ]

    return predPhys, gtPhys


# Mask processing

def erode_fluid_mask(mask):
    """
    Keep only fluid cells whose four direct neighbours
    are also fluid.

    Input:
        mask = 1 fluid, 0 obstacle

    Output:
        boolean valid-mask with one-cell exclusion
        around solid boundary.
    """

    mask = np.asarray(mask).astype(bool)

    valid = mask.copy()

    valid[1:, :] &= mask[:-1, :]
    valid[:-1, :] &= mask[1:, :]
    valid[:, 1:] &= mask[:, :-1]
    valid[:, :-1] &= mask[:, 1:]

    # Avoid outer-domain derivative boundary.
    valid[0, :] = False
    valid[-1, :] = False
    valid[:, 0] = False
    valid[:, -1] = False

    return valid


# Finite differences

def ddx(a, dx=1.0):
    """
    Central finite difference along spatial x dimension.
    Array shape ends with [..., H, W].

    Dataset convention here:
        H = 128 = x-like dimension
        W = 64  = y-like dimension
    """

    out = np.zeros_like(a)

    out[..., 1:-1, :] = (
        a[..., 2:, :]
        - a[..., :-2, :]
    ) / (2.0 * dx)

    out[..., 0, :] = (
        a[..., 1, :]
        - a[..., 0, :]
    ) / dx

    out[..., -1, :] = (
        a[..., -1, :]
        - a[..., -2, :]
    ) / dx

    return out


def ddy(a, dy=1.0):

    out = np.zeros_like(a)

    out[..., :, 1:-1] = (
        a[..., :, 2:]
        - a[..., :, :-2]
    ) / (2.0 * dy)

    out[..., :, 0] = (
        a[..., :, 1]
        - a[..., :, 0]
    ) / dy

    out[..., :, -1] = (
        a[..., :, -1]
        - a[..., :, -2]
    ) / dy

    return out


def gradient_magnitude(a):

    gx = ddx(a, DX)
    gy = ddy(a, DY)

    return np.sqrt(
        gx ** 2
        + gy ** 2
    )


def vorticity(u, v):

    return (
        ddx(v, DX)
        - ddy(u, DY)
    )


# Masked statistics

def masked_mean(a, validMask):

    # a:
    # [M,E,N,T,H,W]
    # validMask:
    # [H,W]

    mask = validMask.reshape(
        1, 1, 1, 1,
        validMask.shape[0],
        validMask.shape[1],
    )

    values = a[
        np.broadcast_to(
            mask,
            a.shape,
        )
    ]

    return float(
        np.mean(values)
    )


def masked_relative_l2(pred, gt, validMask):

    mask = validMask.reshape(
        1, 1, 1, 1,
        validMask.shape[0],
        validMask.shape[1],
    )

    mask = np.broadcast_to(
        mask,
        pred.shape,
    )

    return relative_l2(
        pred,
        gt,
        mask=mask,
    )


# Continuity residual

def continuity_residual(fields):
    """
    fields:
      [M,E,N,T,4,H,W]

    channels:
      0 u
      1 v
      2 density
      3 pressure

    Discrete continuity:
      d rho / dt
      + d(rho*u)/dx
      + d(rho*v)/dy

    Residual corresponds to intervals between t and t+1,
    so time dimension becomes T-1.
    """

    u = fields[..., 0, :, :]
    v = fields[..., 1, :, :]
    rho = fields[..., 2, :, :]

    if rho.shape[3] < 2:
        return None

    rho_t0 = rho[:, :, :, :-1]
    rho_t1 = rho[:, :, :, 1:]

    drho_dt = (
        rho_t1 - rho_t0
    ) / DT

    # Evaluate spatial flux at interval midpoint.
    u_mid = 0.5 * (
        u[:, :, :, :-1]
        + u[:, :, :, 1:]
    )

    v_mid = 0.5 * (
        v[:, :, :, :-1]
        + v[:, :, :, 1:]
    )

    rho_mid = 0.5 * (
        rho_t0 + rho_t1
    )

    flux_x = rho_mid * u_mid
    flux_y = rho_mid * v_mid

    div_mass_flux = (
        ddx(flux_x, DX)
        + ddy(flux_y, DY)
    )

    return (
        drho_dt
        + div_mass_flux
    )


# Metrics

def compute_metrics(
    predFull,
    gtData,
    obsMask,
    inferenceSeconds,
):

    pred, gt = align_prediction_and_ground_truth(
        predFull,
        gtData,
    )

    if obsMask is None:
        validMask = np.ones(
            pred.shape[-2:],
            dtype=bool,
        )
        validMask[
            [0, -1], :
        ] = False
        validMask[
            :, [0, -1]
        ] = False

    else:
        validMask = erode_fluid_mask(
            obsMask
        )


    # Accuracy

    err = pred - gt

    predMSE = np.mean(
        err ** 2
    )

    predMSE_u = np.mean(
        (pred[..., 0, :, :]
         - gt[..., 0, :, :]) ** 2
    )

    predMSE_v = np.mean(
        (pred[..., 1, :, :]
         - gt[..., 1, :, :]) ** 2
    )

    predMSE_density = np.mean(
        (pred[..., 2, :, :]
         - gt[..., 2, :, :]) ** 2
    )

    predMSE_pressure = np.mean(
        (pred[..., 3, :, :]
         - gt[..., 3, :, :]) ** 2
    )

    relL2 = relative_l2(
        pred,
        gt,
    )

    relL2_u = relative_l2(
        pred[..., 0, :, :],
        gt[..., 0, :, :],
    )

    relL2_v = relative_l2(
        pred[..., 1, :, :],
        gt[..., 1, :, :],
    )

    relL2_density = relative_l2(
        pred[..., 2, :, :],
        gt[..., 2, :, :],
    )

    relL2_pressure = relative_l2(
        pred[..., 3, :, :],
        gt[..., 3, :, :],
    )

    mse_per_timestep = np.mean(
        err ** 2,
        axis=(0, 1, 2, 4, 5, 6),
    )

    relL2_per_timestep = []

    for t in range(
        pred.shape[3]
    ):

        relL2_per_timestep.append(
            relative_l2(
                pred[..., t, :, :, :],
                gt[..., t, :, :, :],
            )
        )

    relL2_per_timestep = np.asarray(
        relL2_per_timestep
    )


    # Temporal increment error

    if pred.shape[3] > 1:

        pred_dt = np.diff(
            pred,
            axis=3,
        )

        gt_dt = np.diff(
            gt,
            axis=3,
        )

        temporalIncrementMSE = np.mean(
            (pred_dt - gt_dt) ** 2
        )

    else:
        temporalIncrementMSE = float("nan")


    # Continuity residual

    predCont = continuity_residual(
        pred
    )

    gtCont = continuity_residual(
        gt
    )

    if predCont is not None:

        predContinuityMAE = masked_mean(
            np.abs(predCont),
            validMask,
        )

        predContinuityMSE = masked_mean(
            predCont ** 2,
            validMask,
        )

        gtContinuityMAE = masked_mean(
            np.abs(gtCont),
            validMask,
        )

        gtContinuityMSE = masked_mean(
            gtCont ** 2,
            validMask,
        )

        continuityMSEExcess = (
            predContinuityMSE
            - gtContinuityMSE
        )

        predContinuityMSE_per_timestep = []

        gtContinuityMSE_per_timestep = []

        for t in range(
            predCont.shape[3]
        ):

            predContinuityMSE_per_timestep.append(
                masked_mean(
                    predCont[:, :, :, t:t+1] ** 2,
                    validMask,
                )
            )

            gtContinuityMSE_per_timestep.append(
                masked_mean(
                    gtCont[:, :, :, t:t+1] ** 2,
                    validMask,
                )
            )

        predContinuityMSE_per_timestep = np.asarray(
            predContinuityMSE_per_timestep
        )

        gtContinuityMSE_per_timestep = np.asarray(
            gtContinuityMSE_per_timestep
        )

    else:
        predContinuityMAE = float("nan")
        predContinuityMSE = float("nan")
        gtContinuityMAE = float("nan")
        gtContinuityMSE = float("nan")
        continuityMSEExcess = float("nan")

        predContinuityMSE_per_timestep = np.asarray([])
        gtContinuityMSE_per_timestep = np.asarray([])


    # Vorticity
    predU = pred[..., 0, :, :]
    predV = pred[..., 1, :, :]

    gtU = gt[..., 0, :, :]
    gtV = gt[..., 1, :, :]

    predVort = vorticity(
        predU,
        predV,
    )

    gtVort = vorticity(
        gtU,
        gtV,
    )

    vortError = (
        predVort - gtVort
    )

    vorticityMSE = masked_mean(
        vortError ** 2,
        validMask,
    )

    vorticityRelL2 = masked_relative_l2(
        predVort,
        gtVort,
        validMask,
    )


    # Pressure gradient

    predP = pred[..., 3, :, :]
    gtP = gt[..., 3, :, :]

    predPGrad = gradient_magnitude(
        predP
    )

    gtPGrad = gradient_magnitude(
        gtP
    )

    pressureGradientMSE = masked_mean(
        (predPGrad - gtPGrad) ** 2,
        validMask,
    )

    pressureGradientRelL2 = masked_relative_l2(
        predPGrad,
        gtPGrad,
        validMask,
    )


    # Density gradient

    predRho = pred[..., 2, :, :]
    gtRho = gt[..., 2, :, :]

    predRhoGrad = gradient_magnitude(
        predRho
    )

    gtRhoGrad = gradient_magnitude(
        gtRho
    )

    densityGradientMSE = masked_mean(
        (predRhoGrad - gtRhoGrad) ** 2,
        validMask,
    )

    densityGradientRelL2 = masked_relative_l2(
        predRhoGrad,
        gtRhoGrad,
        validMask,
    )


    # Rollout stability

    if len(
        mse_per_timestep
    ) > 0:

        firstMSE = float(
            mse_per_timestep[0]
        )

        lastMSE = float(
            mse_per_timestep[-1]
        )

        rolloutStabilityRatio = (
            lastMSE
            / (firstMSE + 1e-12)
        )

    else:
        firstMSE = float("nan")
        lastMSE = float("nan")
        rolloutStabilityRatio = float("nan")

    if len(
        mse_per_timestep
    ) > 1:

        rolloutStabilitySlope = float(
            np.polyfit(
                np.arange(
                    len(
                        mse_per_timestep
                    )
                ),
                mse_per_timestep,
                deg=1,
            )[0]
        )

    else:
        rolloutStabilitySlope = float("nan")


    # Efficiency

    nModels = predFull.shape[0]
    nEvals = predFull.shape[1]
    nSeq = predFull.shape[2]
    tPred = predFull.shape[3]

    diffusionCalls = (
        nModels
        * nEvals
        * nSeq
        * int(
            math.ceil(
                tPred
                / max(
                    outputHorizon,
                    1,
                )
            )
        )
    )

    reverseDenoisingSteps = (
        diffusionCalls
        * reverseStepsPerCall
    )

    metrics = {
        # Accuracy
        "predMSE": safe_float(predMSE),
        "predMSE_u": safe_float(predMSE_u),
        "predMSE_v": safe_float(predMSE_v),
        "predMSE_density": safe_float(predMSE_density),
        "predMSE_pressure": safe_float(predMSE_pressure),

        "relL2": safe_float(relL2),
        "relL2_u": safe_float(relL2_u),
        "relL2_v": safe_float(relL2_v),
        "relL2_density": safe_float(relL2_density),
        "relL2_pressure": safe_float(relL2_pressure),

        # Temporal
        "temporalIncrementMSE": safe_float(
            temporalIncrementMSE
        ),

        # Mass physics
        "predContinuityMAE": safe_float(
            predContinuityMAE
        ),
        "predContinuityMSE": safe_float(
            predContinuityMSE
        ),
        "gtContinuityMAE": safe_float(
            gtContinuityMAE
        ),
        "gtContinuityMSE": safe_float(
            gtContinuityMSE
        ),
        "continuityMSEExcess": safe_float(
            continuityMSEExcess
        ),

        # Flow structure
        "vorticityMSE": safe_float(
            vorticityMSE
        ),
        "vorticityRelL2": safe_float(
            vorticityRelL2
        ),

        # Transonic spatial structure
        "pressureGradientMSE": safe_float(
            pressureGradientMSE
        ),
        "pressureGradientRelL2": safe_float(
            pressureGradientRelL2
        ),
        "densityGradientMSE": safe_float(
            densityGradientMSE
        ),
        "densityGradientRelL2": safe_float(
            densityGradientRelL2
        ),

        # Rollout stability
        "rolloutStabilityFirstMSE": safe_float(
            firstMSE
        ),
        "rolloutStabilityLastMSE": safe_float(
            lastMSE
        ),
        "rolloutStabilityRatio": safe_float(
            rolloutStabilityRatio
        ),
        "rolloutStabilitySlope": safe_float(
            rolloutStabilitySlope
        ),

        # Efficiency
        "inferenceTimeSec": safe_float(
            inferenceSeconds
        ),
        "inferenceTimeMin": safe_float(
            inferenceSeconds / 60.0
        ),
        "numDiffusionCalls_estimated": int(
            diffusionCalls
        ),
        "numReverseDenoisingSteps_estimated": int(
            reverseDenoisingSteps
        ),
        "reverseStepsPerCall": int(
            reverseStepsPerCall
        ),
        "outputHorizon": int(
            outputHorizon
        ),
        "rolloutLength": int(
            tPred
        ),
        "numTrainedModels": int(
            nModels
        ),
        "numEvals": int(
            nEvals
        ),
        "numSequences": int(
            nSeq
        ),

        # Metric metadata
        "physicsDX": DX,
        "physicsDY": DY,
        "physicsDT": DT,
        "physicsMetricConvention":
            "discrete residual on resampled 128x64 grid",
    }

    curves = {
        "mse_per_timestep":
            mse_per_timestep,

        "relL2_per_timestep":
            relL2_per_timestep,

        "predContinuityMSE_per_timestep":
            predContinuityMSE_per_timestep,

        "gtContinuityMSE_per_timestep":
            gtContinuityMSE_per_timestep,
    }

    return metrics, curves


# Save outputs

def save_metrics(
    testSetOutPath,
    currentModelName,
    metrics,
    curves,
):

    jsonPath = os.path.join(
        testSetOutPath,
        currentModelName
        + "_metrics.json",
    )

    csvPath = os.path.join(
        testSetOutPath,
        currentModelName
        + "_metrics.csv",
    )

    with open(
        jsonPath,
        "w",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    with open(
        csvPath,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                metrics.keys()
            ),
        )

        writer.writeheader()
        writer.writerow(
            metrics
        )

    curveSpecs = {
        "mse_per_timestep":
            "predMSE",

        "relL2_per_timestep":
            "relativeL2",

        "predContinuityMSE_per_timestep":
            "predContinuityMSE",

        "gtContinuityMSE_per_timestep":
            "gtContinuityMSE",
    }

    for curveName, valueName in curveSpecs.items():

        curve = curves[
            curveName
        ]

        path = os.path.join(
            testSetOutPath,
            currentModelName
            + "_"
            + curveName
            + ".csv",
        )

        with open(
            path,
            "w",
            newline="",
        ) as f:

            writer = csv.writer(
                f
            )

            writer.writerow(
                [
                    "t",
                    valueName,
                ]
            )

            for t, value in enumerate(
                curve
            ):
                writer.writerow(
                    [
                        t,
                        float(value),
                    ]
                )

    print("Saved metrics:")
    print(" ", jsonPath)
    print(" ", csvPath)


# Prepare output folders + ground truth

os.makedirs(
    outFolder,
    exist_ok=True,
)

for shortName, testSet in testSets.items():

    testSetOutPath = os.path.join(
        outFolder,
        shortName,
    )

    os.makedirs(
        testSetOutPath,
        exist_ok=True,
    )

    shutil.copy(
        sys.argv[0],
        os.path.join(
            testSetOutPath,
            "sample_models_tra_10h.py",
        ),
    )

    gtPath = os.path.join(
        testSetOutPath,
        "groundTruth.dict",
    )

    if os.path.isfile(
        gtPath
    ):
        print(
            "Skipping existing ground truth:",
            gtPath,
        )
        continue

    testSet.transform = lambda x: x

    sampler = SequentialSampler(
        testSet
    )

    loader = DataLoader(
        testSet,
        sampler=sampler,
        batch_size=len(
            testSet
        ),
        drop_last=False,
    )

    sample = next(
        iter(loader)
    )

    torch.save(
        sample,
        gtPath,
    )

    print(
        "Saved ground truth:",
        gtPath,
    )

    print(
        "Ground truth shape:",
        tuple(
            sample[
                "data"
            ].shape
        ),
    )


# Load model

print("\n========================================")
print("Loading model")
print("========================================")
print(modelPath)

model = PredictionModel.load(
    modelPath,
    useGPU=True,
)

model.eval()

configure_diffusion_decoder(
    model,
    evalOptions,
)


# Sample

for shortName, testSet in testSets.items():

    print("\n========================================")
    print("DATASET:", shortName)
    print("========================================")

    testSetOutPath = os.path.join(
        outFolder,
        shortName,
    )

    predPath = os.path.join(
        testSetOutPath,
        modelLabel + ".npz",
    )

    gtPath = os.path.join(
        testSetOutPath,
        "groundTruth.dict",
    )

    inferenceSeconds = float("nan")

    # If predictions already exist, the original sampling time can
    # be supplied so efficiency metrics are preserved on re-evaluation.
    inferenceTimeMinOverride = os.environ.get(
        "INFERENCE_TIME_MIN_OVERRIDE",
        "",
    ).strip()

    if inferenceTimeMinOverride:
        inferenceSeconds = (
            float(inferenceTimeMinOverride) * 60.0
        )

    # Prediction

    if os.path.isfile(
        predPath
    ):

        print(
            "Loading existing prediction:",
            predPath,
        )

        predFull = np.load(
            predPath
        )["arr_0"]

    else:

        timerStart = time.perf_counter()

        p_d_test = copy.deepcopy(
            model.p_d
        )

        p_d_test.augmentations = [
            "normalize"
        ]

        p_d_test.sequenceLength = (
            testSet.sequenceLength
        )

        p_d_test.randSeqOffset = False

        transformations = Transforms(
            p_d_test
        )

        testSet.transform = transformations

        sampler = SequentialSampler(
            testSet
        )

        loader = DataLoader(
            testSet,
            sampler=sampler,
            batch_size=1,
            drop_last=False,
            num_workers=numWorkers,
            pin_memory=True,
        )

        predEvals = []

        with torch.no_grad():

            for run in range(
                numEvals
            ):

                print(
                    f"Sampling eval "
                    f"{run+1}/{numEvals}"
                )

                predSamples = []

                for s, sample in enumerate(
                    loader
                ):

                    data = sample[
                        "data"
                    ].to(
                        device
                    )

                    if type(
                        sample[
                            "simParameters"
                        ]
                    ) is not dict:

                        simParameters = sample[
                            "simParameters"
                        ].to(
                            device
                        )

                    else:
                        simParameters = None

                    prediction, _, _ = model(
                        data,
                        simParameters,
                    )

                    # Expected:
                    # [B,T,C,H,W]
                    prediction = (
                        prediction
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )

                    predSamples.append(
                        prediction
                        .cpu()
                        .numpy()
                    )

                predSamples = np.concatenate(
                    predSamples,
                    axis=2,
                )

                predSamples = denormalize_predictions(
                    predSamples,
                    transformations,
                )

                predEvals.append(
                    predSamples
                )

        predFull = np.concatenate(
            predEvals,
            axis=1,
        )

        # Add model dimension if needed.
        if predFull.ndim == 6:
            predFull = predFull[
                np.newaxis,
                ...
            ]

        np.savez_compressed(
            predPath,
            predFull,
        )

        timerEnd = time.perf_counter()

        inferenceSeconds = (
            timerEnd
            - timerStart
        )

        print(
            "Saved prediction:",
            predPath,
        )

        print(
            "Prediction shape:",
            predFull.shape,
        )

        print(
            "Inference time:",
            inferenceSeconds / 60.0,
            "min",
        )


    # Metrics

    gtData, obsMask = load_ground_truth(
        gtPath
    )

    metrics, curves = compute_metrics(
        predFull,
        gtData,
        obsMask,
        inferenceSeconds,
    )

    save_metrics(
        testSetOutPath,
        modelLabel,
        metrics,
        curves,
    )

    print(
        "\nMetric summary for",
        shortName,
    )

    summaryKeys = [
        "predMSE",
        "predMSE_u",
        "predMSE_v",
        "predMSE_density",
        "predMSE_pressure",

        "relL2",
        "relL2_u",
        "relL2_v",
        "relL2_density",
        "relL2_pressure",

        "temporalIncrementMSE",

        "predContinuityMAE",
        "predContinuityMSE",
        "gtContinuityMAE",
        "gtContinuityMSE",
        "continuityMSEExcess",

        "vorticityMSE",
        "vorticityRelL2",

        "pressureGradientMSE",
        "pressureGradientRelL2",

        "densityGradientMSE",
        "densityGradientRelL2",

        "rolloutStabilityFirstMSE",
        "rolloutStabilityLastMSE",
        "rolloutStabilityRatio",
        "rolloutStabilitySlope",

        "inferenceTimeMin",
        "numDiffusionCalls_estimated",
        "numReverseDenoisingSteps_estimated",
    ]

    for key in summaryKeys:
        print(
            f"  {key}: "
            f"{metrics[key]}"
        )


print(
    "\nTransonic sampling and evaluation complete."
)
