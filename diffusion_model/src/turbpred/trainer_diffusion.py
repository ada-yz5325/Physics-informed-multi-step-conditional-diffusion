import time
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from turbpred.model import PredictionModel
from turbpred.loss import PredictionLoss
from turbpred.loss_history import LossHistory
from turbpred.params import TrainingParams, LossParams


def _zero(device: torch.device) -> torch.Tensor:
    return torch.zeros(1, device=device)


def _prediction_part(x: torch.Tensor) -> torch.Tensor:
    """
    Use prediction frames if the sequence contains reconstruction + prediction frames.

    For diffusion training, the clean target usually has sequence length 1, so this
    returns x unchanged. It is kept for compatibility with the existing LossHistory
    / PredictionLoss conventions.
    """
    if x.ndim == 5 and x.shape[1] > 1:
        return x[:, 1:]
    return x


def _is_multistep_model(model: PredictionModel) -> bool:
    return bool(getattr(model.p_d, "multiStep", False))


def _future_steps(model: PredictionModel) -> int:
    if _is_multistep_model(model):
        return int(getattr(model.p_d, "futureSteps", 1))
    return 1


def _state_channels(model: PredictionModel) -> int:
    """
    One physical state channel count.

    In this codebase velocity is counted by p_d.dimension, while simFields adds
    fields such as pressure and simParams adds parameters such as Reynolds number.
    For the current p,u,v,rey setting this is 2 + 1 + 1 = 4.
    """
    return model.p_d.dimension + len(model.p_d.simFields) + len(model.p_d.simParams)


def _reshape_clean_target_for_physics(x: torch.Tensor, model: PredictionModel) -> torch.Tensor:
    """
    Convert clean diffusion target/prediction to B S C W H for physics losses.

    Single-step baseline:
        B, 1, C, W, H -> unchanged

    Multi-step P=2 channel-concatenated target:
        B, 1, P*C, W, H -> B, P, C, W, H

    This is deliberately backward-compatible. If the tensor does not match the
    expected multi-step layout, it is returned unchanged rather than crashing the
    original baseline.
    """
    if x.ndim == 4:
        x = x.unsqueeze(1)

    if x.ndim != 5:
        return x

    if not _is_multistep_model(model):
        return x

    future_steps = _future_steps(model)
    if future_steps <= 1:
        return x

    state_channels = _state_channels(model)

    # Expected multi-step clean target format from model.py:
    #     B, 1, future_steps * state_channels, W, H
    if x.shape[1] == 1 and x.shape[2] == future_steps * state_channels:
        b, _, _, w, h = x.shape
        return x.reshape(b, future_steps, state_channels, w, h)

    # Already reshaped or otherwise compatible.
    return x


def _divergence_2d(x: torch.Tensor) -> torch.Tensor:
    """
    Compute 2D divergence for tensors shaped B S C W H.

    Assumes channel 0 = u and channel 1 = v.
    """
    if x.ndim != 5 or x.shape[2] < 2:
        return _zero(x.device)

    du_dx, _ = torch.gradient(x[:, :, 0:1], dim=(3, 4))
    _, dv_dy = torch.gradient(x[:, :, 1:2], dim=(3, 4))
    return du_dx + dv_dy


def _vorticity_2d(x: torch.Tensor) -> torch.Tensor:
    """
    Compute 2D scalar vorticity for tensors shaped B S C W H.

    omega = dv/dx - du/dy
    Assumes channel 0 = u and channel 1 = v.
    """
    if x.ndim != 5 or x.shape[2] < 2:
        return _zero(x.device)

    dv_dx, _ = torch.gradient(x[:, :, 1:2], dim=(3, 4))
    _, du_dy = torch.gradient(x[:, :, 0:1], dim=(3, 4))
    return dv_dx - du_dy


class TrainerDiffusion(object):
    model: PredictionModel
    trainLoader: DataLoader
    optimizer: Optimizer
    trainHistory: LossHistory
    writer: SummaryWriter
    p_t: TrainingParams
    p_l: Optional[LossParams]

    def __init__(
            self,
            model: PredictionModel,
            trainLoader: DataLoader,
            optimizer: Optimizer,
            trainHistory: LossHistory,
            writer: SummaryWriter,
            p_t: TrainingParams,
            p_l: Optional[LossParams] = None):
        self.model = model
        self.trainLoader = trainLoader
        self.optimizer = optimizer
        self.trainHistory = trainHistory
        self.writer = writer
        self.p_t = p_t
        self.p_l = p_l

    def _physics_losses(self, prediction) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute physics losses from clean-field estimates returned by the diffusion model.

        Expected training-time prediction tuple:
            prediction[0] = true diffusion noise
            prediction[1] = predicted diffusion noise
            prediction[2] = estimated clean target field
            prediction[3] = true clean target field

        Single-step clean target shape:
            B, 1, C, W, H

        Multi-step P=2 clean target shape from compatible model.py:
            B, 1, P*C, W, H

        This function reshapes multi-step clean targets back to:
            B, P, C, W, H

        Then divergence and vorticity losses are averaged over all future frames.
        """
        device = prediction[0].device
        div_loss = _zero(device)
        vort_loss = _zero(device)

        if self.p_l is None:
            return div_loss, vort_loss

        reg_div = float(getattr(self.p_l, "regDiv", 0.0))
        reg_vort = float(getattr(self.p_l, "regVort", 0.0))

        if reg_div <= 0 and reg_vort <= 0:
            return div_loss, vort_loss

        if not isinstance(prediction, (tuple, list)) or len(prediction) < 4:
            return div_loss, vort_loss

        clean_pred = prediction[2]
        clean_true = prediction[3]

        if not torch.is_tensor(clean_pred) or not torch.is_tensor(clean_true):
            return div_loss, vort_loss

        clean_pred = _reshape_clean_target_for_physics(clean_pred, self.model)
        clean_true = _reshape_clean_target_for_physics(clean_true, self.model)

        if clean_pred.ndim == 4:
            clean_pred = clean_pred.unsqueeze(1)
        if clean_true.ndim == 4:
            clean_true = clean_true.unsqueeze(1)

        if clean_pred.ndim != 5 or clean_true.ndim != 5:
            return div_loss, vort_loss

        clean_pred = _prediction_part(clean_pred)
        clean_true = _prediction_part(clean_true)

        if reg_div > 0:
            div_pred = _divergence_2d(clean_pred)
            div_loss = torch.abs(div_pred).mean()

        if reg_vort > 0:
            vort_pred = _vorticity_2d(clean_pred)
            vort_true = _vorticity_2d(clean_true)
            vort_loss = F.mse_loss(vort_pred, vort_true)

        return div_loss, vort_loss

    def _continuity_loss_transonic(
            self,
            data: torch.Tensor,
            prediction,
            obs_mask=None) -> torch.Tensor:
        """
        Compressible-flow mass-conservation loss on the resampled grid.

        Discrete continuity equation:

            d rho / dt
            + d(rho*u)/dx
            + d(rho*v)/dy = 0

        For the single-step C=2, P=1 baseline:

            data[:, 0] = older conditioning state
            data[:, 1] = last conditioning state t
            prediction[2] = predicted clean target t+1

        The residual is evaluated consistently with the Transonic sampling
        metrics using index/rollout-space spacings by default:

            dx = dy = dt = 1.

        IMPORTANT:
        Training tensors are normalized. The physical channels are therefore
        denormalized to the original 128_tra data scale before calculating the
        residual.

        Channel order:
            0 = u
            1 = v
            2 = density
            3 = pressure
            4 = Mach
        """

        device = prediction[0].device

        if not isinstance(prediction, (tuple, list)) or len(prediction) < 4:
            return _zero(device)

        clean_pred = prediction[2]

        if clean_pred is None or not torch.is_tensor(clean_pred):
            return _zero(device)

        clean_pred = _reshape_clean_target_for_physics(
            clean_pred,
            self.model,
        )

        if clean_pred.ndim == 4:
            clean_pred = clean_pred.unsqueeze(1)

        if clean_pred.ndim != 5:
            return _zero(device)

        # This first implementation is specifically for the single-step
        # Transonic baseline experiment.
        if clean_pred.shape[1] < 1 or clean_pred.shape[2] < 3:
            return _zero(device)

        # Number of historical states used by the direct-DDPM architecture.
        prev_steps = self.model._arch_prev_steps()

        if data.ndim != 5 or data.shape[1] < prev_steps:
            return _zero(device)

        # Last historical conditioning state = state at t.
        last_history = data[:, prev_steps - 1 : prev_steps]

        if last_history.shape[2] < 3 or clean_pred.shape[2] < 3:
            return _zero(device)


        # Multi-step continuity.
        #
        # For P predicted states:
        #
        #   transition 1: x_t       -> xhat_{t+1}
        #   transition 2: xhat_{t+1} -> xhat_{t+2}
        #   ...
        #   transition P: xhat_{t+P-1} -> xhat_{t+P}
        #
        # Single-step P=1 therefore keeps the original behaviour.


        # Previous states for each predicted transition.
        #
        # P=2:
        #   prev_states[:,0] = last history state x_t
        #   prev_states[:,1] = predicted H1
        prev_states = torch.cat(
            (
                last_history,
                clean_pred[:, :-1],
            ),
            dim=1,
        )

        next_states = clean_pred

        if (
            os.environ.get("MULTISTEP_CONT_DIAGNOSTIC", "0") == "1"
            and not hasattr(self, "_multistep_cont_diag_printed")
        ):
            self._multistep_cont_diag_printed = True
            print(
                "MULTISTEP CONTINUITY diagnostic:"
                f" last_history={tuple(last_history.shape)}"
                f" clean_pred={tuple(clean_pred.shape)}"
                f" prev_states={tuple(prev_states.shape)}"
                f" next_states={tuple(next_states.shape)}"
            )

        if prev_states.shape != next_states.shape:
            raise RuntimeError(
                "Continuity multi-step state mismatch: "
                f"prev={tuple(prev_states.shape)}, "
                f"next={tuple(next_states.shape)}"
            )


        # Denormalize u, v and density.
        #
        # traMixed:
        #   u:       mean  0.560642, std 0.216987
        #   v:       mean -0.000129, std 0.216987
        #   density: mean  0.903352, std 0.145391

        dtype = next_states.dtype

        means = torch.tensor(
            [0.560642, -0.000129, 0.903352],
            device=device,
            dtype=dtype,
        ).view(1, 1, 3, 1, 1)

        stds = torch.tensor(
            [0.216987, 0.216987, 0.145391],
            device=device,
            dtype=dtype,
        ).view(1, 1, 3, 1, 1)

        prev_phys = (
            prev_states[:, :, 0:3] * stds + means
        )

        next_phys = (
            next_states[:, :, 0:3] * stds + means
        )

        u0 = prev_phys[:, :, 0:1]
        v0 = prev_phys[:, :, 1:2]
        rho0 = prev_phys[:, :, 2:3]

        u1 = next_phys[:, :, 0:1]
        v1 = next_phys[:, :, 1:2]
        rho1 = next_phys[:, :, 2:3]

        # Keep exactly the same convention as the evaluation metric.
        dx = float(os.environ.get("CONTINUITY_DX", "1.0"))
        dy = float(os.environ.get("CONTINUITY_DY", "1.0"))
        dt = float(os.environ.get("CONTINUITY_DT", "1.0"))

        # Temporal density derivative.
        drho_dt = (rho1 - rho0) / dt

        # Midpoint mass flux between t and t+1.
        rho_mid = 0.5 * (rho0 + rho1)
        u_mid = 0.5 * (u0 + u1)
        v_mid = 0.5 * (v0 + v1)

        flux_x = rho_mid * u_mid
        flux_y = rho_mid * v_mid

        # Central finite differences.
        dflux_dx = torch.zeros_like(flux_x)
        dflux_dy = torch.zeros_like(flux_y)

        dflux_dx[:, :, :, 1:-1, :] = (
            flux_x[:, :, :, 2:, :]
            - flux_x[:, :, :, :-2, :]
        ) / (2.0 * dx)

        dflux_dy[:, :, :, :, 1:-1] = (
            flux_y[:, :, :, :, 2:]
            - flux_y[:, :, :, :, :-2]
        ) / (2.0 * dy)

        residual = (
            drho_dt
            + dflux_dx
            + dflux_dy
        )


        # Valid fluid mask.
        #
        # Dataset mask:
        #   1 = fluid
        #   0 = obstacle
        #
        # Remove the solid plus one neighboring cell because derivative
        # stencils crossing the solid boundary generate artificial gradients.
        # Also exclude the outer grid boundary.
   

        if obs_mask is not None:

            if torch.is_tensor(obs_mask):
                mask = obs_mask.to(device)
            else:
                mask = torch.as_tensor(
                    obs_mask,
                    device=device,
                )

            # DataLoader normally gives B,H,W.
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

            if mask.ndim == 3:
                valid = mask.bool().clone()

                base = mask.bool()

                valid[:, 1:, :] &= base[:, :-1, :]
                valid[:, :-1, :] &= base[:, 1:, :]
                valid[:, :, 1:] &= base[:, :, :-1]
                valid[:, :, :-1] &= base[:, :, 1:]

                valid[:, 0, :] = False
                valid[:, -1, :] = False
                valid[:, :, 0] = False
                valid[:, :, -1] = False

                valid = valid[:, None, None, :, :]

                valid = valid.expand_as(residual)

                if torch.any(valid):
                    return torch.mean(
                        residual[valid] ** 2
                    )

        # Fallback: exclude only outer derivative boundary.
        interior = residual[:, :, :, 1:-1, 1:-1]

        if interior.numel() == 0:
            return _zero(device)

        return torch.mean(
            interior ** 2
        )


    # run one epoch of training

    def _horizon_weighted_diffusion_loss(
            self,
            noise: torch.Tensor,
            predictedNoise: torch.Tensor,
            epoch: int = None) -> torch.Tensor:
        """
        Diffusion Smooth-L1 loss with optional true multi-step horizon weighting.

        Multi-step tensors in this codebase use time-as-channels:

            B, 1, (C_history + P*C_state), H, W

        For example, INC C=2, P=2, stateChannels=4:

            conditioning: 8 channels
            H1 target:    4 channels
            H2 target:    4 channels

        HORIZON_T2_WEIGHT controls the second prediction horizon only.
        Conditioning channels and H1 retain weight 1.0.

        If HORIZON_T2_WEIGHT=1.0, this is exactly the ordinary
        unweighted Smooth-L1 diffusion loss.
        """

        t2_weight = float(
            os.environ.get("HORIZON_T2_WEIGHT", "1.0")
        )

        # Ordinary diffusion training.
        if abs(t2_weight - 1.0) <= 1e-12:
            return F.smooth_l1_loss(
                noise,
                predictedNoise,
            )

        if noise.shape != predictedNoise.shape:
            raise ValueError(
                "noise and predictedNoise shapes differ: "
                f"{tuple(noise.shape)} vs "
                f"{tuple(predictedNoise.shape)}"
            )

        if noise.ndim != 5:
            raise ValueError(
                "Expected diffusion tensor B,S,C,H,W, got "
                f"{tuple(noise.shape)}"
            )

        future_steps = int(
            getattr(self.model.p_d, "futureSteps", 1)
        )

        history_steps = int(
            getattr(self.model.p_d, "historySteps", 2)
        )

        state_channels = (
            self.model.p_d.dimension
            + len(self.model.p_d.simFields)
            + len(self.model.p_d.simParams)
        )

        # HW experiment currently defined for P=2.
        if future_steps != 2:
            raise ValueError(
                "HORIZON_T2_WEIGHT != 1 currently requires "
                f"futureSteps=2, got {future_steps}"
            )

        cond_channels = history_steps * state_channels
        target_channels = future_steps * state_channels

        expected_channels = cond_channels + target_channels

        if noise.shape[2] != expected_channels:
            raise ValueError(
                "Unexpected diffusion channel count for HW: "
                f"got {noise.shape[2]}, expected "
                f"{expected_channels} "
                f"(history={history_steps}, "
                f"future={future_steps}, "
                f"stateChannels={state_channels})"
            )

        per_elem = F.smooth_l1_loss(
            noise,
            predictedNoise,
            reduction="none",
        )

        # Channel weights:
        # conditioning = 1
        # H1           = 1
        # H2           = t2_weight
        weights = torch.ones(
            (1, 1, expected_channels, 1, 1),
            device=noise.device,
            dtype=per_elem.dtype,
        )

        h2_start = cond_channels + state_channels
        h2_end = h2_start + state_channels

        weights[:, :, h2_start:h2_end] = t2_weight

        weighted_loss = (per_elem * weights).mean()

        # Optional one-line sanity check.
        if os.environ.get("HW_DIAGNOSTIC", "0") == "1":
            if not hasattr(self, "_hw_diag_printed"):
                self._hw_diag_printed = True

                h1_start = cond_channels
                h1_end = h1_start + state_channels

                print(
                    "TRUE HW diagnostic:"
                    f" shape={tuple(noise.shape)}"
                    f" historySteps={history_steps}"
                    f" futureSteps={future_steps}"
                    f" stateChannels={state_channels}"
                    f" condChannels={cond_channels}"
                    f" H1=[{h1_start}:{h1_end}] weight=1.0"
                    f" H2=[{h2_start}:{h2_end}]"
                    f" weight={t2_weight}"
                )

        return weighted_loss

    def _temporal_consistency_loss(self, prediction) -> torch.Tensor:
        """
        Temporal consistency loss for P=2.

        prediction[2] = estimated clean prediction
        prediction[3] = true clean target

        L_temp = MSE((pred[t+2] - pred[t+1]), (gt[t+2] - gt[t+1]))
        """
        device = prediction[0].device
        weight = float(os.environ.get("TEMPORAL_LOSS_WEIGHT", "0.0"))

        if weight <= 0:
            return _zero(device)

        if len(prediction) < 4:
            return _zero(device)

        clean_pred = prediction[2]
        clean_true = prediction[3]

        if clean_pred is None or clean_true is None:
            return _zero(device)

        if not torch.is_tensor(clean_pred) or not torch.is_tensor(clean_true):
            return _zero(device)

        # Multi-step diffusion stores future frames concatenated along
        # the channel dimension as B,1,P*C,W,H. Restore them to
        # B,P,C,W,H before computing temporal differences.
        clean_pred = _reshape_clean_target_for_physics(
            clean_pred,
            self.model,
        )
        clean_true = _reshape_clean_target_for_physics(
            clean_true,
            self.model,
        )

        if clean_pred.dim() == 4:
            clean_pred = clean_pred.unsqueeze(1)
        if clean_true.dim() == 4:
            clean_true = clean_true.unsqueeze(1)

        if clean_pred.dim() != 5 or clean_true.dim() != 5:
            return _zero(device)

        if clean_pred.shape[1] < 2 or clean_true.shape[1] < 2:
            return _zero(device)

        pred_delta = clean_pred[:, 1] - clean_pred[:, 0]
        true_delta = clean_true[:, 1] - clean_true[:, 0]

        return F.mse_loss(pred_delta, true_delta)


    def trainingStep(self, epoch: int):
        assert (len(self.trainLoader) > 0), "Not enough samples for one batch!"
        timerStart = time.perf_counter()
        timerEnd = 0

        # Epoch-average diagnostics
        sum_diffusion = 0.0

        sum_div_raw = 0.0
        sum_vort_raw = 0.0
        sum_div_weighted = 0.0
        sum_vort_weighted = 0.0

        sum_continuity_raw = 0.0
        sum_continuity_weighted = 0.0

        sum_total = 0.0
        num_diag_batches = 0

        self.model.train()
        for s, sample in enumerate(self.trainLoader, 0):
            self.optimizer.zero_grad()

            device = torch.device("cuda" if self.model.useGPU else "cpu")
            data = sample["data"].to(device)
            simParameters = sample["simParameters"].to(device) if type(sample["simParameters"]) is not dict else None

            prediction, _, _ = self.model(data, simParameters)
            noise, predictedNoise = prediction[0], prediction[1]

            diffusion_loss = self._horizon_weighted_diffusion_loss(noise, predictedNoise, epoch)

    
            # Optional multi-step per-channel diffusion diagnostic.
            if (
                os.environ.get("P5_CHANNEL_DIAGNOSTIC", "0") == "1"
                and s == 0
            ):
                per_elem = F.smooth_l1_loss(
                    predictedNoise,
                    noise,
                    reduction="none",
                )

                # Expected tensor:
                # B, S, C, H, W
                if per_elem.ndim != 5:
                    print(
                        "P5 CHANNEL DIAGNOSTIC: unexpected tensor shape:",
                        tuple(per_elem.shape),
                    )
                else:
                    per_ch = per_elem.mean(dim=(0, 1, 3, 4))

                    state_names = [
                        "u",
                        "v",
                        "density",
                        "pressure",
                        "mach",
                    ]

                    history_steps_diag = int(
                        getattr(self.model.p_d, "historySteps", 2)
                    )
                    future_steps_diag = int(
                        getattr(self.model.p_d, "futureSteps", 1)
                    )
                    state_channels_diag = (
                        self.model.p_d.dimension
                        + len(self.model.p_d.simFields)
                        + len(self.model.p_d.simParams)
                    )

                    cond_channels_diag = (
                        history_steps_diag
                        * state_channels_diag
                    )

                    print()
                    print("=" * 100)
                    print("MULTISTEP PER-CHANNEL DIFFUSION LOSS DIAGNOSTIC")
                    print("=" * 100)
                    print("noise shape:", tuple(noise.shape))
                    print("predictedNoise shape:", tuple(predictedNoise.shape))
                    print("historySteps:", history_steps_diag)
                    print("futureSteps:", future_steps_diag)
                    print("stateChannels:", state_channels_diag)
                    print("conditioning channels:", cond_channels_diag)
                    print()

                    # Conditioning channels
                    print("CONDITIONING")
                    for h in range(history_steps_diag):
                        vals = []
                        for f in range(state_channels_diag):
                            idx = h * state_channels_diag + f

                            name = (
                                state_names[f]
                                if f < len(state_names)
                                else f"field{f}"
                            )

                            vals.append(
                                f"{name}={per_ch[idx].item():.6e}"
                            )

                        print(
                            f"C{h+1}: "
                            + "  ".join(vals)
                        )

                    print()
                    print("PREDICTION TARGETS")

                    for h in range(future_steps_diag):
                        vals = []

                        for f in range(state_channels_diag):
                            idx = (
                                cond_channels_diag
                                + h * state_channels_diag
                                + f
                            )

                            if idx >= len(per_ch):
                                vals.append(
                                    f"channel{idx}=OUT_OF_RANGE"
                                )
                                continue

                            name = (
                                state_names[f]
                                if f < len(state_names)
                                else f"field{f}"
                            )

                            vals.append(
                                f"{name}={per_ch[idx].item():.6e}"
                            )

                        print(
                            f"H{h+1}: "
                            + "  ".join(vals)
                        )

                    print("=" * 100)
                    print()

            div_loss, vort_loss = self._physics_losses(prediction)
            reg_div = float(getattr(self.p_l, "regDiv", 0.0)) if self.p_l is not None else 0.0
            reg_vort = float(getattr(self.p_l, "regVort", 0.0)) if self.p_l is not None else 0.0

            lossRegDiv = reg_div * div_loss
            lossRegVort = reg_vort * vort_loss

            # Optional diagnostic for checking physics-loss scales.
            if (
                os.environ.get("PHYSICS_LOSS_DIAGNOSTIC", "0") == "1"
                and s == 0
            ):
                weighted_physics = lossRegDiv + lossRegVort

                physics_ratio = (
                    weighted_physics.detach()
                    / (diffusion_loss.detach() + 1e-12)
                )

                print(
                    "Physics loss diagnostic:"
                    f" diffusion={diffusion_loss.detach().item():.6e}"
                    f" raw_div={div_loss.detach().item():.6e}"
                    f" lambda_div={reg_div:.6e}"
                    f" weighted_div={lossRegDiv.detach().item():.6e}"
                    f" raw_vort={vort_loss.detach().item():.6e}"
                    f" lambda_vort={reg_vort:.6e}"
                    f" weighted_vort={lossRegVort.detach().item():.6e}"
                    f" physics_over_diffusion={physics_ratio.item():.6e}"
                )

            temporal_loss = self._temporal_consistency_loss(prediction)
            temporal_weight = float(os.environ.get("TEMPORAL_LOSS_WEIGHT", "0.0"))
            lossRegTemporal = temporal_weight * temporal_loss

            # Compressible-flow continuity loss.
            # Disabled by default for backward compatibility.
            continuity_weight = float(
                os.environ.get("CONTINUITY_LOSS_WEIGHT", "0.0")
            )

            if continuity_weight > 0:
                continuity_loss = self._continuity_loss_transonic(
                    data,
                    prediction,
                    sample.get("obsMask", None),
                )
            else:
                continuity_loss = _zero(device)

            lossRegContinuity = (
                continuity_weight * continuity_loss
            )

            loss = (
                diffusion_loss
                + lossRegDiv
                + lossRegVort
                + lossRegTemporal
                + lossRegContinuity
            )

            # One diagnostic per epoch, useful for choosing lambda.
            if s == 0 and continuity_weight > 0:
                print(
                    "Continuity loss diagnostic:"
                    f" diffusion={diffusion_loss.detach().item():.6e}"
                    f" raw_cont={continuity_loss.detach().item():.6e}"
                    f" lambda={continuity_weight:.6e}"
                    f" weighted_cont={lossRegContinuity.detach().item():.6e}"
                    f" total={loss.detach().item():.6e}"
                )

            # Accumulate epoch-average loss diagnostics.
            sum_diffusion += diffusion_loss.detach().item()

            sum_div_raw += div_loss.detach().item()
            sum_vort_raw += vort_loss.detach().item()
            sum_div_weighted += lossRegDiv.detach().item()
            sum_vort_weighted += lossRegVort.detach().item()

            sum_continuity_raw += continuity_loss.detach().item()
            sum_continuity_weighted += lossRegContinuity.detach().item()

            sum_total += loss.detach().item()
            num_diag_batches += 1

            loss.backward()
            self.optimizer.step()

            timerEnd = time.perf_counter()

            lossParts = {
                "lossFull": loss,
                "lossDiffusion": diffusion_loss,
                "lossRegDiv": lossRegDiv,
                "lossRegVort": lossRegVort,

                # Keep these keys for compatibility with the existing LossHistory.
                "lossRecMSE": diffusion_loss,
                "lossRecLSIM": _zero(device),
                "lossPredMSE": _zero(device),
                "lossPredLSIM": _zero(device),
            }
            lossSeq = {
                "MSE": torch.zeros(4, device=device),
                "LSIM": torch.zeros(4, device=device),
            }

            self.trainHistory.updateBatch(lossParts, lossSeq, s, (timerEnd - timerStart) / 60.0)

        timerEnd = time.perf_counter()
        # Epoch-average continuity-loss diagnostic.
        if num_diag_batches > 0 and continuity_weight > 0:
            mean_diff = sum_diffusion / num_diag_batches
            mean_cont_raw = sum_continuity_raw / num_diag_batches
            mean_cont_weighted = sum_continuity_weighted / num_diag_batches
            mean_total = sum_total / num_diag_batches

            # Ratio of epoch-mean weighted continuity loss
            # to epoch-mean diffusion loss.
            cont_over_diff = (
                mean_cont_weighted / (mean_diff + 1e-12)
            )

            print(
                "Epoch continuity diagnostic:"
                f" epoch={epoch}"
                f" batches={num_diag_batches}"
                f" mean_diffusion={mean_diff:.6e}"
                f" mean_raw_cont={mean_cont_raw:.6e}"
                f" mean_weighted_cont={mean_cont_weighted:.6e}"
                f" continuity_over_diffusion={cont_over_diff:.6e}"
                f" mean_total={mean_total:.6e}"
            )

        # Epoch-average Inc physics-loss diagnostic.
        if (
            num_diag_batches > 0
            and os.environ.get("PHYSICS_EPOCH_DIAGNOSTIC", "0") == "1"
        ):
            mean_diff = sum_diffusion / num_diag_batches

            mean_div_raw = sum_div_raw / num_diag_batches
            mean_vort_raw = sum_vort_raw / num_diag_batches

            mean_div_weighted = (
                sum_div_weighted / num_diag_batches
            )
            mean_vort_weighted = (
                sum_vort_weighted / num_diag_batches
            )

            mean_physics_weighted = (
                mean_div_weighted
                + mean_vort_weighted
            )

            physics_over_diff = (
                mean_physics_weighted
                / (mean_diff + 1e-12)
            )

            print(
                "Epoch physics diagnostic:"
                f" epoch={epoch}"
                f" batches={num_diag_batches}"
                f" mean_diffusion={mean_diff:.6e}"
                f" mean_raw_div={mean_div_raw:.6e}"
                f" mean_weighted_div={mean_div_weighted:.6e}"
                f" mean_raw_vort={mean_vort_raw:.6e}"
                f" mean_weighted_vort={mean_vort_weighted:.6e}"
                f" physics_over_diffusion={physics_over_diff:.6e}"
            )

        self.trainHistory.updateEpoch((timerEnd - timerStart) / 60.0)

        self.trainHistory.prepareAndClearForNextEpoch()


class TesterDiffusion(object):
    model: PredictionModel
    testLoader: DataLoader
    criterion: PredictionLoss
    testHistory: LossHistory
    p_t: TrainingParams

    def __init__(self, model: PredictionModel, testLoader: DataLoader, criterion: PredictionLoss,
                    testHistory: LossHistory, p_t: TrainingParams):
        self.model = model
        self.testLoader = testLoader
        self.criterion = criterion
        self.testHistory = testHistory
        self.p_t = p_t

    # run one epoch of testing
    def testStep(self, epoch: int):
        if epoch % self.testHistory.epochStep != self.testHistory.epochStep - 1:
            return

        assert (len(self.testLoader) > 0), "Not enough samples for one batch!"
        timerStart = time.perf_counter()
        timerEnd = 0

        self.model.eval()
        with torch.no_grad():
            for s, sample in enumerate(self.testLoader, 0):
                device = torch.device("cuda" if self.model.useGPU else "cpu")
                data = sample["data"].to(device)
                simParameters = sample["simParameters"].to(device) if type(sample["simParameters"]) is not dict else None
                if "obsMask" in sample:
                    obsMask = sample["obsMask"].to(device)
                    obsMask = torch.unsqueeze(torch.unsqueeze(obsMask, 1), 2)
                else:
                    obsMask = None

                prediction, _, _ = self.model(data, simParameters)

                if obsMask is not None:
                    _, lossParts, lossSeq = self.criterion(
                        prediction * obsMask,
                        data * obsMask,
                        None,
                        None,
                        weighted=False,
                        noLSIM=False,
                    )
                else:
                    _, lossParts, lossSeq = self.criterion(
                        prediction,
                        data,
                        None,
                        None,
                        weighted=False,
                        noLSIM=False,
                    )

                timerEnd = time.perf_counter()
                self.testHistory.updateBatch(lossParts, lossSeq, s, (timerEnd - timerStart) / 60.0)

            timerEnd = time.perf_counter()
        self.testHistory.updateEpoch((timerEnd - timerStart) / 60.0)

        if obsMask is not None:
            maskedPred = prediction * obsMask
            maskedData = data * obsMask
        else:
            maskedPred = prediction
            maskedData = data

        # Keep TensorBoard image/video writing disabled.
        # This avoids Pillow Image.ANTIALIAS errors and does not affect metrics.
        # self.testHistory.writePredictionExample(maskedPred, maskedData)
        # self.testHistory.writeSequenceLoss(lossSeq)

        self.testHistory.prepareAndClearForNextEpoch()
