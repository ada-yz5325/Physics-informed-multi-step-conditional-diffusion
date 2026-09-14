import os
import time
import sys
import shutil
import copy
import glob
import json
import csv
import math

import torch
from torch.utils.data import DataLoader, SequentialSampler
import numpy as np

from turbpred.model import PredictionModel
from turbpred.model_diffusion import DiffusionModel
from turbpred.turbulence_dataset import TurbulenceDataset
from turbpred.data_transformations import Transforms


"""
100-step sampling + evaluation script for the baseline physics-aware ACDM model on 128_inc.

This script uses an existing trained baseline checkpoint. It does not retrain the model.

Main functions:
  1. Load the trained single-step autoregressive conditional diffusion baseline.
  2. Run 100-step rollout prediction on low-Re and high-Re test cases.
  3. Save prediction and ground truth.
  4. Compute evaluation metrics:
       - predMSE, predMSE_u, predMSE_v, predMSE_pres
       - relative L2 error: overall, u, v, pressure
       - relative L2 over rollout timestep
       - divergence error
       - vorticity error
       - temporal continuity error
       - rollout stability metrics
       - inference time
       - estimated number of diffusion calls and reverse denoising steps

Expected prediction output shape:
  [trained_models, num_evals, dataset_sequences, sequence_length, channels, H, W]

Channel convention:
  channel 0 = u velocity
  channel 1 = v velocity
  channel 2 = pressure
  optional channel 3 = Reynolds number conditioning channel
"""


# Basic runtime settings

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

if device != "cuda":
    raise RuntimeError(
        "CUDA is not available. Run this script inside a GPU PBS job, not on the login node."
    )


defaultDataRoot = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data")
)
dataRoot = os.environ.get("DATA_ROOT", defaultDataRoot)

outFolder = os.environ.get(
    "OUT_FOLDER",
    "results/sampling_puv_phys_10h_100step_relL2",
)

modelFolder = os.environ.get("MODEL_FOLDER", "models/2D_Inc")

# Either set MODEL_PATH exactly, or let MODEL_GLOB find the trained baseline model.
# Examples:
#   export MODEL_PATH=models/2D_Inc/128_acdm-r20_puv_phys_10h_00/Model.pth
#   export MODEL_GLOB="128_acdm-r20_puv_phys_10h*/Model.pth"
modelPathEnv = os.environ.get("MODEL_PATH", "").strip()
modelGlob = os.environ.get("MODEL_GLOB", "128_acdm-r20_puv_phys_10h*/Model.pth")
modelLabel = os.environ.get("MODEL_LABEL", "acdm-r20-puv-phys-10h-100step")

numEvals = int(os.environ.get("NUM_EVALS", "1"))

# Main change for this script: default rollout length is 100.
rolloutLength = int(os.environ.get("ROLLOUT_LENGTH", "100"))

frameStart = int(os.environ.get("TEST_FRAME_START", "1000"))
frameEnd = frameStart + rolloutLength * 2

numWorkers = int(os.environ.get("NUM_WORKERS", "0"))

# Single-step baseline: OUTPUT_HORIZON=1.
outputHorizon = int(os.environ.get("OUTPUT_HORIZON", "1"))

# For acdm-r20, there are 20 reverse denoising steps per diffusion call.
reverseStepsPerCall = int(os.environ.get("REVERSE_STEPS_PER_CALL", "20"))

# If FORCE_RESAMPLE=1, ignore existing prediction .npz and sample again.
forceResample = os.environ.get("FORCE_RESAMPLE", "0").strip() == "1"

print("Runtime configuration:")
print("  device:", device)
print("  dataRoot:", dataRoot)
print("  outFolder:", outFolder)
print("  modelFolder:", modelFolder)
print("  modelGlob:", modelGlob)
print("  modelLabel:", modelLabel)
print("  numEvals:", numEvals)
print("  rolloutLength:", rolloutLength)
print("  frame range:", (frameStart, frameEnd))
print("  outputHorizon:", outputHorizon)
print("  reverseStepsPerCall:", reverseStepsPerCall)
print("  forceResample:", forceResample)


# Resolve model path(s)
if modelPathEnv:
    resolvedModelPaths = [modelPathEnv]
else:
    resolvedModelPaths = sorted(
        glob.glob(os.path.join(modelFolder, modelGlob)),
        key=os.path.getmtime,
        reverse=True,
    )

if not resolvedModelPaths:
    raise FileNotFoundError(
        "No trained Model.pth was found.\n"
        f"Checked modelFolder={modelFolder!r}, modelGlob={modelGlob!r}.\n"
        "On HPC, run:\n"
        "  find models/2D_Inc -name 'Model.pth'\n"
        "or:\n"
        "  find runs/2D_Inc -name 'Model.pth'\n"
        "Then set MODEL_PATH or MODEL_FOLDER/MODEL_GLOB accordingly."
    )

print("Using model path(s):")
for p in resolvedModelPaths:
    print("  ", p)


# Test datasets
testSets = {
    "lowRey": TurbulenceDataset(
        "Test Low Reynolds 100-200",
        [dataRoot],
        filterTop=["128_inc"],
        filterSim=[[82, 84, 86, 88, 90]],
        filterFrame=[(frameStart, frameEnd)],
        sequenceLength=[[rolloutLength, 2]],
        simFields=["pres"],
        simParams=["rey"],
        printLevel="sim",
    ),
    "highRey": TurbulenceDataset(
        "Test High Reynolds 900-1000",
        [dataRoot],
        filterTop=["128_inc"],
        filterSim=[[0, 2, 4, 6, 8]],
        filterFrame=[(frameStart, frameEnd)],
        sequenceLength=[[rolloutLength, 2]],
        simFields=["pres"],
        simParams=["rey"],
        printLevel="sim",
    ),
}

models = {
    modelLabel: (
        resolvedModelPaths,
        {"lowRey": 1, "highRey": 1},
        {
            "numEvals": numEvals,
            "sequentialEvalRuns": {"lowRey": True, "highRey": True},
            "samplingMode": "ddpm",
            "posteriorSampling": "random",
            "initialSampling": "random",
            "conditioningIntegration": "noisy",
        },
    )
}


# Helper functions

def configure_diffusion_decoder(model: PredictionModel, evalOptions: dict) -> None:
    """Set diffusion inference options on the decoder."""
    if isinstance(model.modelDecoder, DiffusionModel):
        modules = [model.modelDecoder]
    elif isinstance(model.modelDecoder, torch.nn.ModuleList):
        modules = [m for m in model.modelDecoder if isinstance(m, DiffusionModel)]
    else:
        modules = []

    for module in modules:
        if "samplingMode" in evalOptions:
            module.inferenceSamplingMode = evalOptions["samplingMode"]
        if "posteriorSampling" in evalOptions:
            module.inferencePosteriorSampling = evalOptions["posteriorSampling"]
        if "initialSampling" in evalOptions:
            module.inferenceInitialSampling = evalOptions["initialSampling"]
        if "conditioningIntegration" in evalOptions:
            module.inferenceConditioningIntegration = evalOptions["conditioningIntegration"]


def denormalize_predictions(predSamples: np.ndarray, testTransformations: Transforms) -> np.ndarray:
    """
    Undo normalisation for predicted channels.

    Predicted channel convention:
      3 channels: [u, v, pressure]
      4 channels: [u, v, pressure, Reynolds number]

    Transform channel index convention in this codebase is typically:
      u -> 0
      v -> 1
      pressure -> 3
      Reynolds number -> 4
    """
    predChannels = predSamples.shape[4]

    if predChannels == 3:
        fieldIndices = [0, 1, 3]
    elif predChannels == 4:
        fieldIndices = [0, 1, 3, 4]
    else:
        raise ValueError(
            f"Unexpected number of predicted channels: {predChannels}. "
            "Expected 3 ([u,v,p]) or 4 ([u,v,p,Re])."
        )

    normMean = np.asarray(testTransformations.normMean[fieldIndices])
    normStd = np.asarray(testTransformations.normStd[fieldIndices])

    normMean = np.expand_dims(normMean, axis=(0, 1, 2, 3, 5, 6))
    normStd = np.expand_dims(normStd, axis=(0, 1, 2, 3, 5, 6))

    return (predSamples * normStd) + normMean


def load_ground_truth_data(gtPath: str) -> np.ndarray:
    """Load saved groundTruth.dict and return raw data array."""
    gt = torch.load(gtPath, map_location="cpu")
    data = gt["data"]
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    return np.asarray(data)


def align_prediction_and_ground_truth(predFull: np.ndarray, gtData: np.ndarray):
    """
    Align prediction and ground truth arrays.

    predFull shape:
      [M, E, N, Tpred, Cpred, H, W]

    gtData shape:
      [N, Tgt, Cgt, H, W]

    We use the last Tpred frames of gtData, because the dataset includes
    historical conditioning frames and target rollout frames.
    Only u, v, pressure are compared.
    """
    pred = np.asarray(predFull)
    gt = np.asarray(gtData)

    if pred.ndim != 7:
        raise ValueError(f"Expected predFull with 7 dims, got shape {pred.shape}")
    if gt.ndim != 5:
        raise ValueError(f"Expected gtData with 5 dims, got shape {gt.shape}")

    tPred = pred.shape[3]
    cPred = pred.shape[4]
    cGt = gt.shape[2]

    if cPred < 3 or cGt < 3:
        raise ValueError(f"Need at least 3 channels for u,v,p. pred={pred.shape}, gt={gt.shape}")

    pred_uvp = pred[..., :3, :, :]

    # Use last Tpred frames as prediction target.
    gt_rollout = gt[:, -tPred:, :3, :, :]

    # Broadcast gt over trained-model and evaluation dimensions.
    gt_uvp = gt_rollout[None, None, ...]

    return pred_uvp, gt_uvp


def spatial_divergence(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute approximate 2D divergence du/dx + dv/dy on the grid."""
    dudx = np.gradient(u, axis=-2)
    dvdy = np.gradient(v, axis=-1)
    return dudx + dvdy


def spatial_vorticity(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute approximate 2D scalar vorticity dv/dx - du/dy."""
    dvdx = np.gradient(v, axis=-2)
    dudy = np.gradient(u, axis=-1)
    return dvdx - dudy


def safe_float(x):
    x = float(np.asarray(x))
    if math.isnan(x) or math.isinf(x):
        return x
    return x


def relative_l2(error: np.ndarray, target: np.ndarray, repeat_factor: int = 1) -> float:
    """
    Compute relative L2 = ||error||_2 / ||target||_2.

    repeat_factor is used when target has broadcast dimensions while error has
    model/evaluation dimensions.
    """
    eps = 1e-12
    numerator = np.sqrt(np.sum(error ** 2))
    denominator = np.sqrt(np.sum(target ** 2) * max(repeat_factor, 1))
    return safe_float(numerator / (denominator + eps))


def compute_metrics(predFull: np.ndarray, gtData: np.ndarray, inferenceSeconds: float):
    """Compute all project evaluation metrics from prediction and ground truth."""
    pred, gt = align_prediction_and_ground_truth(predFull, gtData)

    # Channel split: [u, v, pressure]
    pred_u = pred[..., 0, :, :]
    pred_v = pred[..., 1, :, :]
    pred_p = pred[..., 2, :, :]

    gt_u = gt[..., 0, :, :]
    gt_v = gt[..., 1, :, :]
    gt_p = gt[..., 2, :, :]

    err = pred - gt

    # MSE over rollout timestep.
    mse_per_timestep = np.mean(err ** 2, axis=(0, 1, 2, 4, 5, 6))

    # Field prediction error.
    predMSE = np.mean(err ** 2)
    predMSE_u = np.mean((pred_u - gt_u) ** 2)
    predMSE_v = np.mean((pred_v - gt_v) ** 2)
    predMSE_pres = np.mean((pred_p - gt_p) ** 2)

    # Relative L2 error.
    # pred shape: [M, E, N, T, C, H, W]
    # gt shape:   [1, 1, N, T, C, H, W]
    repeatFactor = pred.shape[0] * pred.shape[1]

    relL2 = relative_l2(err, gt, repeat_factor=repeatFactor)
    relL2_u = relative_l2(pred_u - gt_u, gt_u, repeat_factor=repeatFactor)
    relL2_v = relative_l2(pred_v - gt_v, gt_v, repeat_factor=repeatFactor)
    relL2_pres = relative_l2(pred_p - gt_p, gt_p, repeat_factor=repeatFactor)

    eps = 1e-12
    relL2_per_timestep = np.sqrt(
        np.sum(err ** 2, axis=(0, 1, 2, 4, 5, 6))
    ) / (
        np.sqrt(
            np.sum(gt ** 2, axis=(0, 1, 2, 4, 5, 6)) * repeatFactor
        ) + eps
    )

    # Physical consistency.
    pred_div = spatial_divergence(pred_u, pred_v)
    predDivMean = np.mean(np.abs(pred_div))
    predDivMSE = np.mean(pred_div ** 2)

    pred_vort = spatial_vorticity(pred_u, pred_v)
    gt_vort = spatial_vorticity(gt_u, gt_v)
    predVortMSE = np.mean((pred_vort - gt_vort) ** 2)

    # Temporal behaviour.
    pred_dt = np.diff(pred, axis=3)
    gt_dt = np.diff(gt, axis=3)

    temporalContinuityError = (
        np.mean((pred_dt - gt_dt) ** 2)
        if pred.shape[3] > 1
        else float("nan")
    )
    predTemporalSmoothness = (
        np.mean(pred_dt ** 2)
        if pred.shape[3] > 1
        else float("nan")
    )
    gtTemporalSmoothness = (
        np.mean(gt_dt ** 2)
        if pred.shape[3] > 1
        else float("nan")
    )

    # Rollout stability.
    firstMSE = mse_per_timestep[0] if len(mse_per_timestep) > 0 else float("nan")
    lastMSE = mse_per_timestep[-1] if len(mse_per_timestep) > 0 else float("nan")
    maxMSE = np.max(mse_per_timestep) if len(mse_per_timestep) > 0 else float("nan")

    rolloutStabilityRatio = lastMSE / (firstMSE + 1e-12)

    if len(mse_per_timestep) > 1:
        rolloutStabilitySlope = np.polyfit(
            np.arange(len(mse_per_timestep)),
            mse_per_timestep,
            deg=1,
        )[0]
    else:
        rolloutStabilitySlope = float("nan")

    # Relative L2 rollout stability.
    firstRelL2 = relL2_per_timestep[0] if len(relL2_per_timestep) > 0 else float("nan")
    lastRelL2 = relL2_per_timestep[-1] if len(relL2_per_timestep) > 0 else float("nan")
    maxRelL2 = np.max(relL2_per_timestep) if len(relL2_per_timestep) > 0 else float("nan")

    # Block-boundary continuity for future multi-step model.
    # For single-step ACDM, OUTPUT_HORIZON=1 and this is not applicable.
    if outputHorizon > 1 and pred.shape[3] > outputHorizon:
        boundary_indices = list(range(outputHorizon, pred.shape[3], outputHorizon))
        boundary_errors = []
        for t in boundary_indices:
            pred_jump = pred[..., t, :, :, :] - pred[..., t - 1, :, :, :]
            gt_jump = gt[..., t, :, :, :] - gt[..., t - 1, :, :, :]
            boundary_errors.append(np.mean((pred_jump - gt_jump) ** 2))
        blockBoundaryContinuityError = (
            np.mean(boundary_errors)
            if boundary_errors
            else float("nan")
        )
    else:
        blockBoundaryContinuityError = float("nan")

    # Efficiency estimates.
    nModels = predFull.shape[0]
    nEvals = predFull.shape[1]
    nSeq = predFull.shape[2]
    tPred = predFull.shape[3]

    diffusionCalls = (
        nModels
        * nEvals
        * nSeq
        * int(math.ceil(tPred / max(outputHorizon, 1)))
    )
    reverseDenoisingSteps = diffusionCalls * reverseStepsPerCall

    metrics = {
        "predMSE": safe_float(predMSE),
        "predMSE_u": safe_float(predMSE_u),
        "predMSE_v": safe_float(predMSE_v),
        "predMSE_pres": safe_float(predMSE_pres),

        "relL2": safe_float(relL2),
        "relL2_u": safe_float(relL2_u),
        "relL2_v": safe_float(relL2_v),
        "relL2_pres": safe_float(relL2_pres),
        "relL2First": safe_float(firstRelL2),
        "relL2Last": safe_float(lastRelL2),
        "relL2Max": safe_float(maxRelL2),

        "predDivMean": safe_float(predDivMean),
        "predDivMSE": safe_float(predDivMSE),
        "predVortMSE": safe_float(predVortMSE),

        "temporalContinuityError": safe_float(temporalContinuityError),
        "predTemporalSmoothness": safe_float(predTemporalSmoothness),
        "gtTemporalSmoothness": safe_float(gtTemporalSmoothness),

        "rolloutStabilityFirstMSE": safe_float(firstMSE),
        "rolloutStabilityLastMSE": safe_float(lastMSE),
        "rolloutStabilityMaxMSE": safe_float(maxMSE),
        "rolloutStabilityRatio": safe_float(rolloutStabilityRatio),
        "rolloutStabilitySlope": safe_float(rolloutStabilitySlope),

        "blockBoundaryContinuityError": safe_float(blockBoundaryContinuityError),

        "inferenceTimeSec": safe_float(inferenceSeconds),
        "inferenceTimeMin": safe_float(inferenceSeconds / 60.0)
        if not math.isnan(inferenceSeconds)
        else float("nan"),

        "numDiffusionCalls_estimated": int(diffusionCalls),
        "numReverseDenoisingSteps_estimated": int(reverseDenoisingSteps),
        "reverseStepsPerCall": int(reverseStepsPerCall),
        "outputHorizon": int(outputHorizon),
        "rolloutLength": int(tPred),
        "numTrainedModels": int(nModels),
        "numEvals": int(nEvals),
        "numSequences": int(nSeq),
    }

    return metrics, mse_per_timestep, relL2_per_timestep


def save_metrics(
    testSetOutPath: str,
    modelName: str,
    metrics: dict,
    mse_per_timestep: np.ndarray,
    relL2_per_timestep: np.ndarray,
) -> None:
    """Save metrics as JSON, one-row CSV, timestep-MSE CSV, and timestep-relative-L2 CSV."""
    jsonPath = os.path.join(testSetOutPath, modelName + "_metrics.json")
    csvPath = os.path.join(testSetOutPath, modelName + "_metrics.csv")
    mseCurvePath = os.path.join(testSetOutPath, modelName + "_mse_per_timestep.csv")
    relL2CurvePath = os.path.join(testSetOutPath, modelName + "_relL2_per_timestep.csv")

    with open(jsonPath, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(csvPath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    with open(mseCurvePath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "predMSE"])
        for t, value in enumerate(mse_per_timestep):
            writer.writerow([t, float(value)])

    with open(relL2CurvePath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "relativeL2"])
        for t, value in enumerate(relL2_per_timestep):
            writer.writerow([t, float(value)])

    print("Saved metrics:")
    print("  ", jsonPath)
    print("  ", csvPath)
    print("  ", mseCurvePath)
    print("  ", relL2CurvePath)


# Main logic
os.makedirs(outFolder, exist_ok=True)

# Save ground truth once per test set.
for shortTestSetName, testSet in testSets.items():
    testSetOutPath = os.path.join(outFolder, shortTestSetName)
    os.makedirs(testSetOutPath, exist_ok=True)

    try:
        shutil.copy(
            sys.argv[0],
            os.path.join(testSetOutPath, "sample_models_inc_with_metrics.py"),
        )
    except Exception as e:
        print("Warning: could not copy current script:", e)

    gtPath = os.path.join(testSetOutPath, "groundTruth.dict")

    if os.path.isfile(gtPath) and not forceResample:
        print(f"Skipping existing ground truth: {gtPath}")
        continue

    testSet.transform = lambda x: x
    testSampler = SequentialSampler(testSet)
    testLoader = DataLoader(
        testSet,
        sampler=testSampler,
        batch_size=len(testSet),
        drop_last=False,
    )

    sample = next(iter(testLoader))
    torch.save(sample, gtPath)

    print(f"Saved ground truth to {gtPath}")
    print("Ground truth data shape:", tuple(sample["data"].shape))


# Generate predictions and compute metrics.
for currentModelName, modelData in models.items():
    print("\n\n--------------------------------------------------")
    print(f"Processing model: {currentModelName}")
    print("--------------------------------------------------")

    modelPaths, batchDict, evalOptions = modelData

    for shortTestSetName, testSet in testSets.items():
        print(f"\nDATASET: {shortTestSetName}")

        testSetOutPath = os.path.join(outFolder, shortTestSetName)
        os.makedirs(testSetOutPath, exist_ok=True)

        predPath = os.path.join(testSetOutPath, currentModelName + ".npz")
        gtPath = os.path.join(testSetOutPath, "groundTruth.dict")

        if os.path.isfile(predPath) and not forceResample:
            print(f"Loading existing prediction: {predPath}")
            predFull = np.load(predPath)["arr_0"]
            inferenceSeconds = float("nan")
        else:
            predFull = []
            timerFullStart = time.perf_counter()

            for modelPath in modelPaths:
                print(f"Loading model: {modelPath}")
                model = PredictionModel.load(modelPath, useGPU=(device == "cuda"))
                model.eval()
                configure_diffusion_decoder(model, evalOptions)

                batchSize = batchDict[shortTestSetName]

                p_d_test = copy.deepcopy(model.p_d)
                p_d_test.augmentations = ["normalize"]
                p_d_test.sequenceLength = testSet.sequenceLength
                p_d_test.randSeqOffset = False

                testTransformations = Transforms(p_d_test)
                testSet.transform = testTransformations

                testSampler = SequentialSampler(testSet)
                testLoader = DataLoader(
                    testSet,
                    sampler=testSampler,
                    batch_size=batchSize,
                    drop_last=False,
                    num_workers=numWorkers,
                    pin_memory=True,
                )

                with torch.no_grad():
                    predEvals = []
                    numEvalsOption = evalOptions["numEvals"]
                    print(f"\tSampling sequentially for {numEvalsOption} evaluation run(s)...")

                    timerEvalsStart = time.perf_counter()

                    for run in range(numEvalsOption):
                        predSamples = []

                        for s, sample in enumerate(testLoader, 0):
                            print(
                                f"\tDataset={shortTestSetName}, eval={run + 1}/{numEvalsOption}, "
                                f"batch={s + 1}/{len(testLoader)}"
                            )

                            data = sample["data"].to(device)
                            simParameters = (
                                sample["simParameters"].to(device)
                                if type(sample["simParameters"]) is not dict
                                else None
                            )

                            prediction, _, _ = model(data, simParameters)

                            # Add model and evaluation dimensions.
                            prediction = prediction.unsqueeze(0).unsqueeze(0)
                            predSamples.append(prediction.cpu().numpy())

                        predSamples = np.concatenate(predSamples, axis=2)
                        predSamples = denormalize_predictions(
                            predSamples,
                            testTransformations,
                        )
                        predEvals.append(predSamples)

                        timerEvalsEnd = time.perf_counter()
                        print(
                            "\t[%2.2f min] Eval run %d/%d done."
                            % (
                                (timerEvalsEnd - timerEvalsStart) / 60,
                                run + 1,
                                numEvalsOption,
                            )
                        )

                    predEvals = np.concatenate(predEvals, axis=1)
                    predFull.append(predEvals)

            predFull = np.concatenate(predFull, axis=0)
            np.savez_compressed(predPath, predFull)

            timerFullEnd = time.perf_counter()
            inferenceSeconds = timerFullEnd - timerFullStart

            print(
                "\n[%2.2f min] Saved prediction to %s with shape %s\n"
                % (inferenceSeconds / 60, predPath, predFull.shape)
            )

        # Compute metrics.
        gtData = load_ground_truth_data(gtPath)
        metrics, mse_per_timestep, relL2_per_timestep = compute_metrics(
            predFull,
            gtData,
            inferenceSeconds,
        )
        save_metrics(
            testSetOutPath,
            currentModelName,
            metrics,
            mse_per_timestep,
            relL2_per_timestep,
        )

        print("\nMetric summary for", shortTestSetName)
        for key in [
            "predMSE",
            "predMSE_pres",
            "predMSE_u",
            "predMSE_v",
            "relL2",
            "relL2_pres",
            "relL2_u",
            "relL2_v",
            "relL2First",
            "relL2Last",
            "relL2Max",
            "predDivMean",
            "predDivMSE",
            "predVortMSE",
            "temporalContinuityError",
            "rolloutStabilityFirstMSE",
            "rolloutStabilityLastMSE",
            "rolloutStabilityMaxMSE",
            "rolloutStabilityRatio",
            "blockBoundaryContinuityError",
            "inferenceTimeMin",
            "numDiffusionCalls_estimated",
            "numReverseDenoisingSteps_estimated",
        ]:
            print(f"  {key}: {metrics[key]}")


print("\n\n100-step sampling and metric evaluation complete.")