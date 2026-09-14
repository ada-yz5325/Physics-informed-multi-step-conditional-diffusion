import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from turbpred.model_diffusion_blocks import (
    Unet,
    linear_beta_schedule,
    quadratic_beta_schedule,
    sigmoid_beta_schedule,
    cosine_beta_schedule,
)
from turbpred.params import DataParams, ModelParamsDecoder
from turbpred.model_dfpnet import DfpNetTimeEmbedding


### DIFFUSION MODEL WITH CONDITIONING
class DiffusionModel(nn.Module):
    p_d: DataParams
    p_md: ModelParamsDecoder

    def __init__(self, p_d: DataParams, p_md: ModelParamsDecoder, dimension: int, condChannels: int):
        super(DiffusionModel, self).__init__()

        self.p_d = p_d
        self.p_md = p_md
        self.dimension = dimension

        self.timesteps = self.p_md.diffSteps

        # sampling settings
        self.inferencePosteriorSampling = "random"      # "random" or "same", ddpm only
        self.inferenceInitialSampling = "random"        # "random" or "same"
        self.inferenceConditioningIntegration = self.p_md.diffCondIntegration  # "noisy" or "clean"
        self.inferenceSamplingMode = "ddpm" if "ddpm" in self.p_md.arch else "ddim"

        # optional denoising trace for visualisation
        # This is used only during inference when SAVE_DENOISING_TRACE=1.
        self.denoisingTrace = []

        if self.p_md.diffSchedule == "linear":
            betas = linear_beta_schedule(timesteps=self.timesteps)
        elif self.p_md.diffSchedule == "quadratic":
            betas = quadratic_beta_schedule(timesteps=self.timesteps)
        elif self.p_md.diffSchedule == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps=self.timesteps)
        elif self.p_md.diffSchedule == "cosine":
            betas = cosine_beta_schedule(timesteps=self.timesteps)
        else:
            raise ValueError("Unknown variance schedule")

        betas = betas.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        alphas = 1.0 - betas
        alphasCumprod = torch.cumprod(alphas, axis=0)
        alphasCumprodPrev = F.pad(alphasCumprod[:-1], (0, 0, 0, 0, 0, 0, 1, 0), value=1.0)
        sqrtRecipAlphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrtAlphasCumprod = torch.sqrt(alphasCumprod)
        sqrtOneMinusAlphasCumprod = torch.sqrt(1.0 - alphasCumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posteriorVariance = betas * (1.0 - alphasCumprodPrev) / (1.0 - alphasCumprod)
        sqrtPosteriorVariance = torch.sqrt(posteriorVariance)

        self.register_buffer("betas", betas)
        self.register_buffer("sqrtRecipAlphas", sqrtRecipAlphas)
        self.register_buffer("sqrtAlphasCumprod", sqrtAlphasCumprod)
        self.register_buffer("sqrtOneMinusAlphasCumprod", sqrtOneMinusAlphasCumprod)
        self.register_buffer("sqrtPosteriorVariance", sqrtPosteriorVariance)

        baseChannels = self.p_d.dimension + len(self.p_d.simFields) + len(self.p_d.simParams)

        futureSteps = int(getattr(self.p_d, "futureSteps", 1))
        isMultiStep = bool(getattr(self.p_d, "multiStep", False))

        targetChannels = baseChannels * futureSteps if isMultiStep and futureSteps > 1 else baseChannels
        totalChannels = condChannels + targetChannels

        if "dfp" in self.p_md.arch:
            self.unet = DfpNetTimeEmbedding(
                inChannels=totalChannels,
                outChannels=totalChannels,
                blockChannels=self.p_md.decWidth,
            )
        else:
            self.unet = Unet(
                dim=self.p_d.dataSize[0],
                channels=totalChannels,
                dim_mults=(1, 1, 1),
                use_convnext=True,
                convnext_mult=1,
            )

    # ------------------------------------------------------------------
    # Denoising trace helpers
    # ------------------------------------------------------------------

    def _denoising_trace_enabled(self) -> bool:
        """
        Whether to store intermediate reverse-denoising states.

        Enable with:
            export SAVE_DENOISING_TRACE=1
        """
        return os.environ.get("SAVE_DENOISING_TRACE", "0").strip() == "1"

    def _get_trace_steps(self):
        """
        Get diffusion steps to save.

        Example:
            export DENOISING_TRACE_STEPS="20,15,10,5,0"

        Note:
            For diffSteps=20, the reverse loop uses i = 19, 18, ..., 0.
            Here step=20 is used to represent the initial pure-noise target
            before any denoising update.
        """
        default_steps = f"{self.timesteps},{int(self.timesteps * 0.75)},{int(self.timesteps * 0.5)},{int(self.timesteps * 0.25)},0"
        raw = os.environ.get("DENOISING_TRACE_STEPS", default_steps)

        steps = []
        for item in raw.split(","):
            item = item.strip()
            if item == "":
                continue
            try:
                steps.append(int(item))
            except ValueError:
                pass

        steps = sorted(set(steps), reverse=True)
        return steps

    def _reset_denoising_trace(self):
        self.denoisingTrace = []

    def _append_denoising_trace(self, step: int, tensor: torch.Tensor):
        """
        Store a CPU copy of the current target sample.

        Stored tensor shape before final reshape:
            [B*S, C, W, H]

        The sampling script can later reshape or visualise it.
        """
        self.denoisingTrace.append(
            {
                "step": int(step),
                "sample": tensor.detach().cpu(),
            }
        )

    def get_denoising_trace(self):
        """
        Return the currently stored denoising trace.

        Each entry is:
            {
              "step": int,
              "sample": torch.Tensor on CPU
            }
        """
        return self.denoisingTrace

    # input shape (both inputs): B S C W H (D) -> output shape (both outputs): B S nC W H (D)
    def forward(self, conditioning: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        if self.dimension == 3:
            raise NotImplementedError()

        device = "cuda" if data.is_cuda else "cpu"
        seqLen = data.shape[1]

        # combine batch and sequence dimension for decoder processing
        d = torch.reshape(data, (-1, data.shape[2], data.shape[3], data.shape[4]))
        cond = torch.reshape(conditioning, (-1, conditioning.shape[2], conditioning.shape[3], conditioning.shape[4]))

        # TRAINING
        if self.training:

            # Keep the clean target field before adding noise.
            # This is returned for optional physics-aware training losses.
            dClean = d

            # forward diffusion process that adds noise to data
            if self.p_md.diffCondIntegration == "noisy":
                dAll = torch.concat((cond, dClean), dim=1)
                noise = torch.randn_like(dAll, device=device)
                t = torch.randint(0, self.timesteps, (dAll.shape[0],), device=device).long()
                dNoisy = self.sqrtAlphasCumprod[t] * dAll + self.sqrtOneMinusAlphasCumprod[t] * noise

            elif self.p_md.diffCondIntegration == "clean":
                dNoise = torch.randn_like(dClean, device=device)
                t = torch.randint(0, self.timesteps, (dClean.shape[0],), device=device).long()
                dTargetNoisy = self.sqrtAlphasCumprod[t] * dClean + self.sqrtOneMinusAlphasCumprod[t] * dNoise

                noise = torch.concat((cond, dNoise), dim=1)
                dNoisy = torch.concat((cond, dTargetNoisy), dim=1)

            else:
                raise ValueError("Unknown conditioning integration mode")

            # noise prediction with network
            predictedNoise = self.unet(dNoisy, t)

            # Estimate x_0 from predicted noise.
            # Physics losses should be computed on the target part of this clean-field estimate,
            # not on the predicted noise itself.
            x0PredAll = (
                dNoisy - self.sqrtOneMinusAlphasCumprod[t] * predictedNoise
            ) / self.sqrtAlphasCumprod[t]

            condChannels = cond.shape[1]
            x0PredTarget = x0PredAll[:, condChannels:]

            # unstack batch and sequence dimension again
            noise = torch.reshape(
                noise,
                (-1, seqLen, conditioning.shape[2] + data.shape[2], data.shape[3], data.shape[4]),
            )
            predictedNoise = torch.reshape(
                predictedNoise,
                (-1, seqLen, conditioning.shape[2] + data.shape[2], data.shape[3], data.shape[4]),
            )
            x0PredTarget = torch.reshape(
                x0PredTarget,
                (-1, seqLen, data.shape[2], data.shape[3], data.shape[4]),
            )
            x0Target = torch.reshape(
                dClean,
                (-1, seqLen, data.shape[2], data.shape[3], data.shape[4]),
            )

            return noise, predictedNoise, x0PredTarget, x0Target

        # INFERENCE
        else:
            traceEnabled = self._denoising_trace_enabled()
            traceSteps = self._get_trace_steps() if traceEnabled else []

            if traceEnabled:
                self._reset_denoising_trace()

            # conditioned reverse diffusion process
            if self.inferenceInitialSampling == "random":
                dNoise = torch.randn_like(d, device=device)
                cNoise = torch.randn_like(cond, device=device)
            else:
                dNoise = torch.randn(
                    (1, d.shape[1], d.shape[2], d.shape[3]),
                    device=device,
                ).expand(d.shape[0], -1, -1, -1)
                cNoise = torch.randn(
                    (1, cond.shape[1], cond.shape[2], cond.shape[3]),
                    device=device,
                ).expand(cond.shape[0], -1, -1, -1)

            # Save initial pure-noise target before reverse denoising.
            # Use step=self.timesteps, e.g. 20 for a 20-step diffusion model.
            if traceEnabled and self.timesteps in traceSteps:
                self._append_denoising_trace(self.timesteps, dNoise)

            sampleStride = 1

            for i in reversed(range(0, self.timesteps, sampleStride)):
                t = i * torch.ones(cond.shape[0], device=device).long()

                # compute conditioned part with normal forward diffusion
                if self.inferenceConditioningIntegration == "noisy":
                    condNoisy = self.sqrtAlphasCumprod[t] * cond + self.sqrtOneMinusAlphasCumprod[t] * cNoise
                else:
                    condNoisy = cond

                dNoiseCond = torch.concat((condNoisy, dNoise), dim=1)

                # backward diffusion process that removes noise to create data
                predictedNoiseCond = self.unet(dNoiseCond, t)

                # use model noise predictor to predict mean
                modelMean = self.sqrtRecipAlphas[t] * (
                    dNoiseCond
                    - self.betas[t] * predictedNoiseCond / self.sqrtOneMinusAlphasCumprod[t]
                )

                # discard prediction of conditioning
                dNoise = modelMean[:, cond.shape[1]:modelMean.shape[1]]

                if i != 0 and self.inferenceSamplingMode == "ddpm":
                    if self.inferencePosteriorSampling == "random":
                        # sample randomly only for non-final prediction
                        dNoise = dNoise + self.sqrtPosteriorVariance[t] * torch.randn_like(dNoise)
                    else:
                        # sample with same seed only for non-final prediction
                        postNoise = torch.randn(
                            (1, dNoise.shape[1], dNoise.shape[2], dNoise.shape[3]),
                            device=device,
                        ).expand(dNoise.shape[0], -1, -1, -1)
                        dNoise = dNoise + self.sqrtPosteriorVariance[t] * postNoise

                # Save intermediate target after this denoising update.
                # For diffSteps=20, possible i values are 19,...,0.
                # If user asks for step 15, 10, 5, 0, these are saved here.
                if traceEnabled and i in traceSteps:
                    self._append_denoising_trace(i, dNoise)

            # unstack batch and sequence dimension again
            dNoise = torch.reshape(dNoise, (-1, seqLen, data.shape[2], data.shape[3], data.shape[4]))

            return dNoise
        