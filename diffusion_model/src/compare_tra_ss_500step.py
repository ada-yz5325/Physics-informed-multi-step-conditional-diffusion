import os
import pandas as pd
import matplotlib.pyplot as plt

MODELS = {
    "TRA_SS_base": {
        "dir": "results/sampling_tra_baseline_10h_500step_metrics/Tralong_500step",
        "prefix": "acdm-r20-tra-baseline-10h-500step",
    },
    "TRA_SS_cont001": {
        "dir": "results/sampling_tra_cont001_10h_500step_metrics/Tralong_500step",
        "prefix": "acdm-r20-tra-cont001-10h-500step",
    },
    "TRA_SS_cont02": {
        "dir": "results/sampling_tra_cont02_10h_500step_metrics/Tralong_500step",
        "prefix": "acdm-r20-tra-cont02-10h-500step",
    },
}

OUT = "results/tra_ss_500step_comparison"
os.makedirs(OUT, exist_ok=True)


# Load summary metrics
rows = []

for model, cfg in MODELS.items():
    path = os.path.join(cfg["dir"], cfg["prefix"] + "_metrics.csv")
    df = pd.read_csv(path)
    row = df.iloc[0].to_dict()
    row["model"] = model
    rows.append(row)

summary = pd.DataFrame(rows).set_index("model")

wanted = [
    "predMSE",
    "relL2",
    "predMSE_u",
    "predMSE_v",
    "predMSE_density",
    "predMSE_pressure",
    "relL2_u",
    "relL2_v",
    "relL2_density",
    "relL2_pressure",
    "temporalIncrementMSE",
    "rolloutStabilityLastMSE",
    "rolloutStabilitySlope",
    "predContinuityMAE",
    "predContinuityMSE",
    "gtContinuityMAE",
    "gtContinuityMSE",
    "vorticityMSE",
    "vorticityRelL2",
    "pressureGradientMSE",
    "pressureGradientRelL2",
    "densityGradientMSE",
    "densityGradientRelL2",
    "inferenceTimeMin",
    "numDiffusionCalls_estimated",
    "numReverseDenoisingSteps_estimated",
]

summary_out = summary[[x for x in wanted if x in summary.columns]].copy()

summary_out = summary_out.rename(columns={
    "predMSE_u": "MSE_u",
    "predMSE_v": "MSE_v",
    "predMSE_density": "MSE_density",
    "predMSE_pressure": "MSE_pressure",
    "rolloutStabilityLastMSE": "finalStepMSE",
    "numDiffusionCalls_estimated": "numDiffusionCalls",
    "numReverseDenoisingSteps_estimated": "numReverseDenoisingSteps",
})

summary_out.to_csv(
    os.path.join(OUT, "TRA_SS_500step_all_metrics.csv")
)

print("\nTRA SS 500-step comparison:")
print(summary_out.to_string())


# MSE vs rollout
plt.figure(figsize=(10, 6))

for model, cfg in MODELS.items():
    path = os.path.join(
        cfg["dir"],
        cfg["prefix"] + "_mse_per_timestep.csv"
    )
    df = pd.read_csv(path)
    plt.plot(df["t"], df["predMSE"], label=model)

plt.xlabel("Rollout step")
plt.ylabel("MSE")
plt.title("TRA Single-Step Models: MSE vs Rollout Step")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(OUT, "TRA_SS_MSE_vs_rollout.png"),
    dpi=300
)
plt.close()


# Relative L2 vs rollout
plt.figure(figsize=(10, 6))

for model, cfg in MODELS.items():
    path = os.path.join(
        cfg["dir"],
        cfg["prefix"] + "_relL2_per_timestep.csv"
    )
    df = pd.read_csv(path)
    plt.plot(df["t"], df["relativeL2"], label=model)

plt.xlabel("Rollout step")
plt.ylabel("Relative L2")
plt.title("TRA Single-Step Models: Relative L2 vs Rollout Step")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(OUT, "TRA_SS_relL2_vs_rollout.png"),
    dpi=300
)
plt.close()


# Generic grouped bar chart
def barplot(metrics, title, ylabel, filename):
    data = summary[metrics].T

    ax = data.plot(
        kind="bar",
        figsize=(11, 6),
        width=0.75
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, filename), dpi=300)
    plt.close()

# Overall accuracy
barplot(
    ["predMSE", "relL2"],
    "TRA Single-Step Models: Overall Accuracy",
    "Error",
    "TRA_SS_overall_accuracy.png"
)

# Field MSE
barplot(
    [
        "predMSE_u",
        "predMSE_v",
        "predMSE_density",
        "predMSE_pressure",
    ],
    "TRA Single-Step Models: Per-Field MSE",
    "MSE",
    "TRA_SS_field_MSE.png"
)

# Field relative L2
barplot(
    [
        "relL2_u",
        "relL2_v",
        "relL2_density",
        "relL2_pressure",
    ],
    "TRA Single-Step Models: Per-Field Relative L2",
    "Relative L2",
    "TRA_SS_field_relL2.png"
)

# Continuity physics
barplot(
    [
        "predContinuityMAE",
        "predContinuityMSE",
    ],
    "TRA Single-Step Models: Predicted Continuity Residual",
    "Residual error",
    "TRA_SS_continuity.png"
)

# Vorticity
barplot(
    [
        "vorticityMSE",
        "vorticityRelL2",
    ],
    "TRA Single-Step Models: Vorticity Structure",
    "Error",
    "TRA_SS_vorticity.png"
)

# Pressure/density gradients
barplot(
    [
        "pressureGradientMSE",
        "densityGradientMSE",
    ],
    "TRA Single-Step Models: Gradient MSE",
    "MSE",
    "TRA_SS_gradient_MSE.png"
)

barplot(
    [
        "pressureGradientRelL2",
        "densityGradientRelL2",
    ],
    "TRA Single-Step Models: Gradient Relative L2",
    "Relative L2",
    "TRA_SS_gradient_relL2.png"
)

# Temporal scalar metrics
barplot(
    [
        "temporalIncrementMSE",
        "rolloutStabilityLastMSE",
    ],
    "TRA Single-Step Models: Temporal Behaviour",
    "MSE",
    "TRA_SS_temporal.png"
)

print("\nSaved comparison outputs to:")
print(OUT)
