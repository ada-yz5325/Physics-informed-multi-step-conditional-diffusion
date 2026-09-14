#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("results")

MODELS = {
    "INC_SS_base":
        ROOT / "ablation_diffonly_500step",

    "INC_SS_div001":
        ROOT / "ablation_div_500step",

    "INC_SS_vor001":
        ROOT / "ablation_vort_500step",

    "INC_SS_phys001":
        ROOT / "ablation_all_500step",

    "INC_SS_phys0003":
        ROOT / "sampling_puv_phys003_10h_500step_relL2",

    "INC_SS_vor001_div0003":
        ROOT / "sampling_inc_vor001_div0003_10h_500step",
}

REGIMES = ["lowRey", "highRey"]

OUT = ROOT / "comparison_INC_SS_final"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def find_one(folder, pattern):
    matches = sorted(folder.glob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"No file matching {pattern} in {folder}"
        )

    if len(matches) > 1:
        print(
            f"WARNING: multiple files matched {pattern} in {folder}; "
            f"using {matches[0].name}"
        )

    return matches[0]


def load_metrics(folder):
    path = find_one(folder, "*_metrics.csv")

    print("Loading:", path)

    df = pd.read_csv(path)

    if len(df) == 1:
        return df.iloc[0].to_dict()

    if {"metric", "value"}.issubset(df.columns):
        return dict(zip(df["metric"], df["value"]))

    if df.shape[1] == 2:
        return dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

    raise RuntimeError(
        f"Unknown metrics CSV format: {path}"
    )


def load_curve(folder, pattern, expected_value_col):
    path = find_one(folder, pattern)
    df = pd.read_csv(path)

    if "t" in df.columns and expected_value_col in df.columns:
        return (
            df["t"].to_numpy(),
            df[expected_value_col].to_numpy()
        )

    numeric = df.select_dtypes(include=[np.number])

    if numeric.shape[1] < 2:
        raise RuntimeError(
            f"Could not identify curve columns in {path}"
        )

    return (
        numeric.iloc[:, 0].to_numpy(),
        numeric.iloc[:, -1].to_numpy()
    )


# Metric aliases

ALIASES = {

    "MSE_u": [
        "predMSE_u",
        "MSE_u",
    ],

    "MSE_v": [
        "predMSE_v",
        "MSE_v",
    ],

    "MSE_pressure": [
        "predMSE_pres",
        "predMSE_pressure",
        "MSE_pressure",
    ],

    "relL2_pressure": [
        "relL2_pres",
        "relL2_pressure",
    ],

    "finalStepMSE": [
        "rolloutStabilityLastMSE",
        "finalStepMSE",
        "final-step MSE",
    ],

    "vorticityMSE": [
        "predVortMSE",
        "vorticityMSE",
    ],

    "vorticityRelL2": [
        "predVortRelL2",
        "vorticityRelL2",
    ],

    "numDiffusionCalls": [
        "numDiffusionCalls_estimated",
        "numDiffusionCalls",
    ],

    "numReverseDenoisingSteps": [
        "numReverseDenoisingSteps_estimated",
        "numReverseDenoisingSteps",
    ],
}


def metric_value(d, metric):
    if metric in d:
        return d[metric]

    for alias in ALIASES.get(metric, []):
        if alias in d:
            return d[alias]

    return np.nan




# Final metric organization


CATEGORIES = {

    "Accuracy": [
        "predMSE",
        "relL2",
        "MSE_u",
        "MSE_v",
        "MSE_pressure",
        "relL2_u",
        "relL2_v",
        "relL2_pressure",
    ],

    "Temporal": [
        "temporalIncrementMSE",
        "rolloutStabilitySlope",
        "finalStepMSE",
    ],

    "Mass physics": [
        "predDivMean",
        "predDivMSE",
        "gtDivMean",
        "gtDivMSE",
    ],

    "Flow structure": [
        "vorticityMSE",
        "vorticityRelL2",
    ],

    "Efficiency": [
        "inferenceTimeMin",
        "numDiffusionCalls",
        "numReverseDenoisingSteps",
    ],
}


def make_barplot(
    metrics_by_model,
    regime,
    metric_names,
    title,
    ylabel,
    filename
):
    data = {}

    for model in MODELS:
        data[model] = [
            metric_value(
                metrics_by_model[model],
                m
            )
            for m in metric_names
        ]

    df = pd.DataFrame(
        data,
        index=metric_names
    )

    # Remove rows completely unavailable in old metric files.
    df = df.dropna(axis=0, how="all")

    if len(df) == 0:
        print("Skipping empty plot:", filename)
        return

    ax = df.plot(
        kind="bar",
        figsize=(12, 6),
        width=0.78,
    )

    ax.set_title(
        f"{title} ({regime})"
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")

    ax.grid(
        axis="y",
        alpha=0.25
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.legend(
        title="Model",
        fontsize=8
    )

    plt.tight_layout()

    path = OUT / filename
    plt.savefig(
        path,
        dpi=300
    )
    plt.close()

    print("Saved:", path)



# Main comparison

for regime in REGIMES:

    print()
    print("=" * 100)
    print("REGIME:", regime)
    print("=" * 100)

    metrics_by_model = {}

    for model, root in MODELS.items():

        folder = root / regime

        if not folder.exists():
            raise FileNotFoundError(
                f"\nMissing result folder for {model}:\n"
                f"{folder}\n"
            )

        metrics_by_model[model] = load_metrics(folder)



    # Full quantitative table

    rows = []

    for category, metrics in CATEGORIES.items():

        for metric in metrics:

            row = {
                "Category": category,
                "Metric": metric,
            }

            for model in MODELS:
                row[model] = metric_value(
                    metrics_by_model[model],
                    metric
                )

            rows.append(row)

    comparison = pd.DataFrame(rows)

    comparison_path = (
        OUT /
        f"{regime}_INC_SS_final_metrics.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False
    )

    print()
    print("FULL COMPARISON")
    print(comparison.to_string(index=False))
    print()
    print("Saved:", comparison_path)



    # Relative change vs baseline
    #
    # negative = error decreased = improvement

    baseline = metrics_by_model["INC_SS_base"]

    relative_rows = []

    for category, metric_names in CATEGORIES.items():

        for metric in metric_names:

            base = metric_value(
                baseline,
                metric
            )

            try:
                base = float(base)
            except (TypeError, ValueError):
                continue

            if (
                not np.isfinite(base)
                or abs(base) < 1e-15
            ):
                continue

            row = {
                "Category": category,
                "Metric": metric,
                "INC_SS_base": 0.0,
            }

            for model in list(MODELS.keys())[1:]:

                value = metric_value(
                    metrics_by_model[model],
                    metric
                )

                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = np.nan

                if np.isfinite(value):
                    row[model] = (
                        100.0 *
                        (value - base) /
                        base
                    )
                else:
                    row[model] = np.nan

            relative_rows.append(row)

    relative = pd.DataFrame(
        relative_rows
    )

    relative_path = (
        OUT /
        f"{regime}_relative_change_vs_base_percent.csv"
    )

    relative.to_csv(
        relative_path,
        index=False
    )

    print("Saved:", relative_path)



    # MSE vs rollout

    plt.figure(
        figsize=(10, 6)
    )

    for model, root in MODELS.items():

        x, y = load_curve(
            root / regime,
            "*_mse_per_timestep.csv",
            "predMSE"
        )

        plt.plot(
            x,
            y,
            linewidth=1.7,
            label=model
        )

    plt.xlabel(
        "Rollout step"
    )
    plt.ylabel(
        "MSE"
    )

    plt.title(
        f"INC Single-Step Models: "
        f"MSE vs Rollout Step ({regime})"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend(
        fontsize=8
    )

    plt.tight_layout()

    path = (
        OUT /
        f"{regime}_MSE_vs_rollout.png"
    )

    plt.savefig(
        path,
        dpi=300
    )
    plt.close()

    print("Saved:", path)



    # Relative L2 vs rollout
    plt.figure(
        figsize=(10, 6)
    )

    for model, root in MODELS.items():

        x, y = load_curve(
            root / regime,
            "*_relL2_per_timestep.csv",
            "relativeL2"
        )

        plt.plot(
            x,
            y,
            linewidth=1.7,
            label=model
        )

    plt.xlabel(
        "Rollout step"
    )
    plt.ylabel(
        "Relative L2"
    )

    plt.title(
        f"INC Single-Step Models: "
        f"Relative L2 vs Rollout Step ({regime})"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend(
        fontsize=8
    )

    plt.tight_layout()

    path = (
        OUT /
        f"{regime}_relL2_vs_rollout.png"
    )

    plt.savefig(
        path,
        dpi=300
    )
    plt.close()

    print("Saved:", path)



    # Accuracy plots

    make_barplot(
        metrics_by_model,
        regime,
        ["predMSE", "relL2"],
        "INC Overall Accuracy",
        "Error",
        f"{regime}_overall_accuracy.png"
    )

    make_barplot(
        metrics_by_model,
        regime,
        [
            "MSE_u",
            "MSE_v",
            "MSE_pressure",
        ],
        "INC Per-Field MSE",
        "MSE",
        f"{regime}_field_MSE.png"
    )

    make_barplot(
        metrics_by_model,
        regime,
        [
            "relL2_u",
            "relL2_v",
            "relL2_pressure",
        ],
        "INC Per-Field Relative L2",
        "Relative L2",
        f"{regime}_field_relL2.png"
    )


  
    # Mass physics

    make_barplot(
        metrics_by_model,
        regime,
        [
            "predDivMean",
            "predDivMSE",
        ],
        "INC Predicted Divergence",
        "Divergence residual",
        f"{regime}_divergence.png"
    )



    # Vorticity structure
    make_barplot(
        metrics_by_model,
        regime,
        [
            "vorticityMSE",
            "vorticityRelL2",
        ],
        "INC Vorticity Structure",
        "Error",
        f"{regime}_vorticity.png"
    )



    # Temporal

    make_barplot(
        metrics_by_model,
        regime,
        [
            "temporalIncrementMSE",
            "finalStepMSE",
        ],
        "INC Temporal Behaviour",
        "MSE",
        f"{regime}_temporal.png"
    )

    make_barplot(
        metrics_by_model,
        regime,
        [
            "rolloutStabilitySlope",
        ],
        "INC Rollout Stability Slope",
        "Slope",
        f"{regime}_rollout_stability_slope.png"
    )



    # Efficiency
    make_barplot(
        metrics_by_model,
        regime,
        [
            "numDiffusionCalls",
            "numReverseDenoisingSteps",
        ],
        "INC Inference Workload",
        "Count",
        f"{regime}_efficiency_workload.png"
    )


print()
print("=" * 100)
print("FINAL INC SIX-MODEL COMPARISON COMPLETE")
print("Output folder:")
print(OUT)
print("=" * 100)
