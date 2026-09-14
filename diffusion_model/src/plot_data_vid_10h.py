"""
Visualise full-domain rollout flow-field evolution for the 10h physics-aware ACDM PUV model.

Designed for outputs produced by sample_models_inc_10h.py, for example:

    results/sampling_puv_phys_10h_metrics/lowRey/groundTruth.dict
    results/sampling_puv_phys_10h_metrics/lowRey/acdm-r20-puv-phys-10h.npz

Default behaviour in this version:
  - Uses the 10h sampling result folder.
  - Uses the 10h model label.
  - Processes both lowRey and highRey.
  - Processes all main fields: u, v, pres, speed, vort, div.
  - Uses the full 128 x 64 spatial domain by default, not a cropped subdomain.
  - Saves one MP4 animation, one PNG selected-frame figure, and one PDF selected-frame figure
    for each dataset-field-sequence combination.

Default output count with one sequence:
  2 datasets x 6 fields x 3 files = 36 files.

Optional environment variables:
  RESULTS_DIR      default: results/sampling_puv_phys_10h_metrics
  MODEL_LABEL      default: acdm-r20-puv-phys-10h
  OUTPUT_DIR       default: results/flow_visualizations_10h_full_domain
  DATASETS         default: lowRey,highRey
  FIELDS           default: u,v,pres,speed,vort,div
  SEQ_INDICES      default: 0
  ALL_SEQUENCES    default: 0. Set to 1 to plot every sequence in the saved result.
  SPATIAL_ZOOM     default: 0:128,0:64, i.e. full domain for 128 x 64 data.
                   Set to empty string "" to use the full loaded shape automatically.
  TRANSPOSE        default: 1, follows the original display orientation.
  FPS              default: 5
  DPI              default: 140

Examples:
  python src/plot_data_vid_puv_phys_10h_full.py

  DATASETS=lowRey,highRey FIELDS=vort,pres python src/plot_data_vid_puv_phys_10h_full.py

  ALL_SEQUENCES=1 FIELDS=vort python src/plot_data_vid_puv_phys_10h_full.py

  SPATIAL_ZOOM="" python src/plot_data_vid_puv_phys_10h_full.py
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_list(name: str, default: str) -> List[str]:
    value = os.environ.get(name, default)
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_zoom(value: str, shape_hw: Tuple[int, int]) -> Tuple[slice, slice]:
    """Parse SPATIAL_ZOOM='x0:x1,y0:y1' into spatial slices."""
    h, w = shape_hw
    if value is None or value.strip() == "":
        return slice(0, h), slice(0, w)

    try:
        x_part, y_part = value.split(",")
        x0, x1 = [int(v) for v in x_part.split(":")]
        y0, y1 = [int(v) for v in y_part.split(":")]
    except Exception as exc:
        raise ValueError(
            f"Invalid SPATIAL_ZOOM='{value}'. Expected format like '0:128,0:64'."
        ) from exc

    x0 = max(0, min(h, x0))
    x1 = max(0, min(h, x1))
    y0 = max(0, min(w, y0))
    y1 = max(0, min(w, y1))

    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid SPATIAL_ZOOM='{value}' for shape H={h}, W={w}.")

    return slice(x0, x1), slice(y0, y1)


def robust_limits(values: np.ndarray, symmetric: bool = False, q: float = 99.0) -> Tuple[float, float]:
    """Robust colour limits using percentiles."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0

    if symmetric:
        vmax = float(np.percentile(np.abs(finite), q))
        vmax = max(vmax, 1e-12)
        return -vmax, vmax

    vmin = float(np.percentile(finite, 100.0 - q))
    vmax = float(np.percentile(finite, q))
    if abs(vmax - vmin) < 1e-12:
        centre = 0.5 * (vmin + vmax)
        return centre - 1.0, centre + 1.0
    return vmin, vmax


def get_npz_array(path: Path) -> np.ndarray:
    loaded = np.load(path)
    if "arr_0" in loaded:
        return loaded["arr_0"]
    if len(loaded.files) == 1:
        return loaded[loaded.files[0]]
    raise KeyError(f"Could not find arr_0 in {path}. Keys: {loaded.files}")


def torch_load_dict(path: Path):
    """Load a torch-saved dictionary robustly across PyTorch versions."""
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def load_ground_truth(dataset_folder: Path) -> np.ndarray:
    gt_path = dataset_folder / "groundTruth.dict"
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing ground truth file: {gt_path}")

    gt_dict = torch_load_dict(gt_path)
    if "data" not in gt_dict:
        raise KeyError(f"{gt_path} does not contain key 'data'. Keys: {list(gt_dict.keys())}")

    gt = gt_dict["data"]
    gt = gt.cpu().numpy() if isinstance(gt, torch.Tensor) else np.asarray(gt)

    # Expected shape: [sequence, time, channel, H, W]
    if gt.ndim != 5:
        raise ValueError(f"Expected ground truth shape [S,T,C,H,W], got {gt.shape}")
    return gt


def load_prediction(dataset_folder: Path, model_label: str) -> np.ndarray:
    pred_path = dataset_folder / f"{model_label}.npz"
    if not pred_path.exists():
        available = [p.name for p in dataset_folder.glob("*.npz")]
        raise FileNotFoundError(
            f"Missing prediction file: {pred_path}\nAvailable npz files: {available}"
        )

    pred = get_npz_array(pred_path)

    # Expected shape: [model, eval, sequence, time, channel, H, W]
    if pred.ndim != 7:
        raise ValueError(f"Expected prediction shape [M,E,S,T,C,H,W], got {pred.shape}")
    return pred


def compute_field(data_stch: np.ndarray, field: str) -> np.ndarray:
    """
    data_stch shape: [S, T, C, H, W]. Return scalar field [S, T, H, W].

    Current project channel convention:
      channel 0 = u velocity
      channel 1 = v velocity
      channel 2 = pressure
      channel 3 = Reynolds parameter / conditioning parameter
    """
    if data_stch.ndim != 5:
        raise ValueError(f"Expected [S,T,C,H,W], got {data_stch.shape}")

    field = field.lower()
    channels = data_stch.shape[2]
    if channels < 3:
        raise ValueError(f"Expected at least 3 channels [u,v,p], got C={channels}")

    u = data_stch[:, :, 0]
    v = data_stch[:, :, 1]
    p = data_stch[:, :, 2]

    if field == "u":
        return u
    if field == "v":
        return v
    if field in ["p", "pres", "pressure"]:
        return p
    if field == "speed":
        return np.sqrt(u * u + v * v)

    # Gradients are computed over spatial axes H and W.
    # Same convention as the original code:
    #   vorticity = d(v)/dH - d(u)/dW
    du_dh, du_dw = np.gradient(u, axis=(2, 3))
    dv_dh, dv_dw = np.gradient(v, axis=(2, 3))

    if field in ["vort", "vorticity"]:
        return dv_dh - du_dw
    if field in ["div", "divergence"]:
        return du_dh + dv_dw

    raise ValueError("Unsupported FIELD='%s'. Use: u, v, pres, speed, vort, div." % field)


def display_orientation(arr_thw: np.ndarray, transpose: bool) -> np.ndarray:
    """Convert [T,H,W] into display arrays. Original scripts used permute(0,2,1)."""
    if transpose:
        return np.transpose(arr_thw, (0, 2, 1))
    return arr_thw


def make_animation(
    gt_txy: np.ndarray,
    pred_txy: np.ndarray,
    dataset: str,
    field: str,
    seq_idx: int,
    output_path: Path,
    fps: int,
    dpi: int,
) -> None:
    err_txy = np.abs(pred_txy - gt_txy)

    value_data = np.concatenate([gt_txy.reshape(-1), pred_txy.reshape(-1)])
    symmetric = field.lower() in ["u", "v", "vort", "vorticity", "div", "divergence"]
    vmin, vmax = robust_limits(value_data, symmetric=symmetric, q=99.0)
    emin, emax = 0.0, robust_limits(err_txy, symmetric=False, q=99.0)[1]

    cmap_main = "coolwarm" if symmetric else "viridis"
    cmap_err = "magma"

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(11, 3.6), dpi=dpi, squeeze=False)
    axs = axs[0]

    titles = ["Ground truth", "Prediction", "Absolute error"]
    for ax, title in zip(axs, titles):
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    im_gt = axs[0].imshow(gt_txy[0], interpolation="nearest", cmap=cmap_main, vmin=vmin, vmax=vmax)
    im_pr = axs[1].imshow(pred_txy[0], interpolation="nearest", cmap=cmap_main, vmin=vmin, vmax=vmax)
    im_er = axs[2].imshow(err_txy[0], interpolation="nearest", cmap=cmap_err, vmin=emin, vmax=emax)

    cb1 = fig.colorbar(im_pr, ax=axs[:2].tolist(), fraction=0.035, pad=0.02)
    cb1.set_label(field)
    cb2 = fig.colorbar(im_er, ax=axs[2], fraction=0.046, pad=0.04)
    cb2.set_label("|prediction - truth|")

    suptitle = fig.suptitle(f"{dataset} | seq={seq_idx} | {field} | t=0", fontsize=12)

    def update(t: int):
        im_gt.set_array(gt_txy[t])
        im_pr.set_array(pred_txy[t])
        im_er.set_array(err_txy[t])
        suptitle.set_text(f"{dataset} | seq={seq_idx} | {field} | t={t}")
        return [im_gt, im_pr, im_er, suptitle]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=range(gt_txy.shape[0]),
        interval=1000.0 / max(fps, 1),
        blit=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        anim.save(str(output_path), writer=writer, dpi=dpi)
        print(f"Saved animation: {output_path}")
    except Exception as exc:
        fallback = output_path.with_suffix(".gif")
        print(f"MP4 saving failed ({exc}). Falling back to GIF: {fallback}")
        writer = animation.PillowWriter(fps=fps)
        anim.save(str(fallback), writer=writer, dpi=dpi)
        print(f"Saved animation: {fallback}")

    plt.close(fig)


def make_selected_frame_figure(
    gt_txy: np.ndarray,
    pred_txy: np.ndarray,
    dataset: str,
    field: str,
    seq_idx: int,
    output_prefix: Path,
    selected_times: Optional[List[int]] = None,
    dpi: int = 180,
) -> None:
    err_txy = np.abs(pred_txy - gt_txy)
    total_t = gt_txy.shape[0]

    if selected_times is None:
        selected_times = [0, 10, 20, 30, 40, total_t - 1]
    selected_times = [t for t in selected_times if 0 <= t < total_t]
    if not selected_times:
        selected_times = list(range(min(total_t, 6)))

    value_data = np.concatenate([gt_txy.reshape(-1), pred_txy.reshape(-1)])
    symmetric = field.lower() in ["u", "v", "vort", "vorticity", "div", "divergence"]
    vmin, vmax = robust_limits(value_data, symmetric=symmetric, q=99.0)
    emin, emax = 0.0, robust_limits(err_txy, symmetric=False, q=99.0)[1]

    cmap_main = "coolwarm" if symmetric else "viridis"
    cmap_err = "magma"

    fig, axs = plt.subplots(
        nrows=3,
        ncols=len(selected_times),
        figsize=(2.0 * len(selected_times), 5.8),
        dpi=dpi,
        squeeze=False,
    )

    row_labels = ["Ground truth", "Prediction", "|Error|"]
    last_main = None
    last_err = None

    for col, t in enumerate(selected_times):
        axs[0, col].set_title(f"t={t}")
        images = [
            (gt_txy[t], cmap_main, vmin, vmax),
            (pred_txy[t], cmap_main, vmin, vmax),
            (err_txy[t], cmap_err, emin, emax),
        ]
        for row, (img, cmap, lo, hi) in enumerate(images):
            ax = axs[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(row_labels[row])
            im = ax.imshow(img, interpolation="nearest", cmap=cmap, vmin=lo, vmax=hi)
            if row < 2:
                last_main = im
            else:
                last_err = im

    fig.suptitle(f"{dataset} | seq={seq_idx} | {field} | selected rollout frames", fontsize=12)
    fig.tight_layout(pad=0.5, w_pad=0.15, h_pad=0.25)
    fig.subplots_adjust(right=0.88, top=0.90)

    if last_main is not None:
        cax1 = fig.add_axes([0.895, 0.38, 0.018, 0.45])
        cb1 = fig.colorbar(last_main, cax=cax1)
        cb1.set_label(field)

    if last_err is not None:
        cax2 = fig.add_axes([0.895, 0.10, 0.018, 0.20])
        cb2 = fig.colorbar(last_err, cax=cax2)
        cb2.set_label("|err|")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(str(png_path), bbox_inches="tight")
    fig.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved selected-frame figure: {png_path}")
    print(f"Saved selected-frame figure: {pdf_path}")


def parse_seq_indices(value: str, total_sequences: int) -> List[int]:
    if value.strip().lower() in ["all", "*"]:
        return list(range(total_sequences))

    indices = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0 or idx >= total_sequences:
            raise IndexError(f"SEQ index {idx} out of range for total_sequences={total_sequences}")
        indices.append(idx)

    if not indices:
        indices = [0]
    return indices


def process_one_dataset(
    results_dir: Path,
    dataset: str,
    model_label: str,
    fields: List[str],
    model_idx: int,
    eval_idx: int,
    seq_indices_setting: str,
    all_sequences: bool,
    fps: int,
    dpi: int,
    transpose: bool,
    output_dir: Path,
    spatial_zoom: str,
) -> None:
    dataset_folder = results_dir / dataset

    print("\n" + "=" * 70)
    print(f"Processing dataset: {dataset}")
    print("=" * 70)

    gt = load_ground_truth(dataset_folder)
    pred_full = load_prediction(dataset_folder, model_label)

    print(f"Loaded ground truth shape: {gt.shape}")
    print(f"Loaded prediction shape:   {pred_full.shape}")

    if model_idx >= pred_full.shape[0]:
        raise IndexError(f"MODEL_IDX={model_idx} out of range for prediction shape {pred_full.shape}")
    if eval_idx >= pred_full.shape[1]:
        raise IndexError(f"EVAL_IDX={eval_idx} out of range for prediction shape {pred_full.shape}")

    pred = pred_full[model_idx, eval_idx]  # [S,T,C,H,W]

    total_sequences = min(gt.shape[0], pred.shape[0])
    if all_sequences:
        seq_indices = list(range(total_sequences))
    else:
        seq_indices = parse_seq_indices(seq_indices_setting, total_sequences)

    print(f"Sequence indices to plot: {seq_indices}")
    print(f"Fields to plot: {fields}")

    for field in fields:
        gt_field = compute_field(gt, field)
        pred_field = compute_field(pred, field)

        t_len = min(gt_field.shape[1], pred_field.shape[1])
        gt_field = gt_field[:, :t_len]
        pred_field = pred_field[:, :t_len]

        for seq_idx in seq_indices:
            gt_seq = gt_field[seq_idx]      # [T,H,W]
            pred_seq = pred_field[seq_idx]  # [T,H,W]

            x_slice, y_slice = parse_zoom(spatial_zoom, gt_seq.shape[1:])
            gt_seq = gt_seq[:, x_slice, y_slice]
            pred_seq = pred_seq[:, x_slice, y_slice]

            print(
                f"Plotting dataset={dataset}, field={field}, seq={seq_idx}, "
                f"spatial zoom x={x_slice.start}:{x_slice.stop}, y={y_slice.start}:{y_slice.stop}"
            )
            print(f"Selected field shape before display orientation: {gt_seq.shape}")

            gt_disp = display_orientation(gt_seq, transpose=transpose)
            pred_disp = display_orientation(pred_seq, transpose=transpose)
            print(f"Selected field shape for display: {gt_disp.shape}")

            base_name = f"{dataset}_{model_label}_seq{seq_idx}_{field}_full_domain"
            video_path = output_dir / dataset / field / f"{base_name}.mp4"
            frame_prefix = output_dir / dataset / field / f"{base_name}_selected_frames"

            make_animation(
                gt_txy=gt_disp,
                pred_txy=pred_disp,
                dataset=dataset,
                field=field,
                seq_idx=seq_idx,
                output_path=video_path,
                fps=fps,
                dpi=dpi,
            )

            make_selected_frame_figure(
                gt_txy=gt_disp,
                pred_txy=pred_disp,
                dataset=dataset,
                field=field,
                seq_idx=seq_idx,
                output_prefix=frame_prefix,
                dpi=max(160, dpi),
            )


def main() -> None:
    results_dir = Path(os.environ.get("RESULTS_DIR", "results/sampling_puv_phys_10h_metrics"))
    model_label = os.environ.get("MODEL_LABEL", "acdm-r20-puv-phys-10h")

    datasets = env_list("DATASETS", "lowRey,highRey")
    fields = env_list("FIELDS", "u,v,pres,speed,vort,div")

    model_idx = env_int("MODEL_IDX", 0)
    eval_idx = env_int("EVAL_IDX", 0)
    seq_indices_setting = os.environ.get("SEQ_INDICES", "0")
    all_sequences = os.environ.get("ALL_SEQUENCES", "0") == "1"

    fps = env_int("FPS", 5)
    dpi = env_int("DPI", 140)
    transpose = os.environ.get("TRANSPOSE", "1") != "0"
    output_dir = Path(os.environ.get("OUTPUT_DIR", "results/flow_visualizations_10h_full_domain"))

    # Full 128 x 64 domain by default. If your loaded data has a different size,
    # use SPATIAL_ZOOM="" to automatically use the full loaded shape.
    spatial_zoom = os.environ.get("SPATIAL_ZOOM", "0:128,0:64")

    print("Runtime configuration:")
    print(f"  results_dir:    {results_dir}")
    print(f"  model_label:    {model_label}")
    print(f"  datasets:       {datasets}")
    print(f"  fields:         {fields}")
    print(f"  model_idx:      {model_idx}")
    print(f"  eval_idx:       {eval_idx}")
    print(f"  seq_indices:    {'ALL' if all_sequences else seq_indices_setting}")
    print(f"  transpose:      {transpose}")
    print(f"  spatial_zoom:   {spatial_zoom if spatial_zoom else 'full loaded shape'}")
    print(f"  output_dir:     {output_dir}")
    print(f"  fps:            {fps}")
    print(f"  dpi:            {dpi}")

    for dataset in datasets:
        process_one_dataset(
            results_dir=results_dir,
            dataset=dataset,
            model_label=model_label,
            fields=fields,
            model_idx=model_idx,
            eval_idx=eval_idx,
            seq_indices_setting=seq_indices_setting,
            all_sequences=all_sequences,
            fps=fps,
            dpi=dpi,
            transpose=transpose,
            output_dir=output_dir,
            spatial_zoom=spatial_zoom,
        )

    print("\nVisualisation complete.")
    print(f"Outputs saved under: {output_dir}")


if __name__ == "__main__":
    main()
