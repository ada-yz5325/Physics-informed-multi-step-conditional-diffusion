#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path


BASE_SCRIPT = "src/plot_data_vid_10h.py"

RESULTS_DIR = "results/sampling_puv_phys_10h_500step_relL2"
MODEL_LABEL = "acdm-r20-puv-phys-10h-500step"
OUTPUT_DIR = "results/flow_visualizations_baseline_10h_500step_allfields"

FIELDS = "u,v,pres,speed,vort,div"
DATASETS = "lowRey,highRey"
SEQ_INDICES = "0"


def main():
    if not Path(BASE_SCRIPT).is_file():
        raise FileNotFoundError(f"Cannot find base plotting script: {BASE_SCRIPT}")

    if not Path(RESULTS_DIR).is_dir():
        raise FileNotFoundError(f"Cannot find results directory: {RESULTS_DIR}")

    env = os.environ.copy()
    env["RESULTS_DIR"] = RESULTS_DIR
    env["MODEL_LABEL"] = MODEL_LABEL
    env["OUTPUT_DIR"] = OUTPUT_DIR

    env["DATASETS"] = DATASETS
    env["FIELDS"] = FIELDS
    env["SEQ_INDICES"] = SEQ_INDICES
    env["ALL_SEQUENCES"] = "0"

    env["SPATIAL_ZOOM"] = ""
    env["MODEL_IDX"] = "0"
    env["EVAL_IDX"] = "0"
    env["FPS"] = "5"
    env["DPI"] = "160"
    env["TRANSPOSE"] = "1"

    print("=" * 80)
    print("Plotting baseline P=1 10h, 500-step rollout")
    print(f"  RESULTS_DIR = {env['RESULTS_DIR']}")
    print(f"  MODEL_LABEL = {env['MODEL_LABEL']}")
    print(f"  OUTPUT_DIR  = {env['OUTPUT_DIR']}")
    print(f"  DATASETS    = {env['DATASETS']}")
    print(f"  FIELDS      = {env['FIELDS']}")
    print("=" * 80)

    subprocess.run([sys.executable, BASE_SCRIPT], env=env, check=True)

    print("\n500-step baseline P=1 10h visualisation complete.")


if __name__ == "__main__":
    main()
