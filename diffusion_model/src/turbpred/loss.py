import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

from lsim.distance_model import DistanceModel as LSIM_Model

from turbpred.params import LossParams


# input shape: B S C W H -> output shape: B S C
def loss_lsim(lsimModel: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # normalize to [0,255] for each sequence
    xMin = torch.amin(x, dim=(1, 3, 4), keepdim=True)
    xMax = torch.amax(x, dim=(1, 3, 4), keepdim=True)
    yMin = torch.amin(y, dim=(1, 3, 4), keepdim=True)
    yMax = torch.amax(y, dim=(1, 3, 4), keepdim=True)

    # avoid division by zero when a field/channel is constant
    eps = 1e-8
    ref = 255 * ((x - xMin) / (xMax - xMin + eps))
    oth = 255 * ((y - yMin) / (yMax - yMin + eps))

    # compare each sequence and channel individually by moving them to batch dimension
    # and adding a dummy channel dimension and lsim parameter dimension
    sizeBatch, sizeSeq, sizeChannel = ref.shape[0], ref.shape[1], ref.shape[2]
    ref = torch.reshape(ref, (-1, 1, 1, ref.shape[3], ref.shape[4]))
    oth = torch.reshape(oth, (-1, 1, 1, oth.shape[3], oth.shape[4]))

    # clone channel dimension as lsim compares 3-channel data
    ref = ref.expand(-1, -1, 3, -1, -1)
    oth = oth.expand(-1, -1, 3, -1, -1)

    inDict = {"reference": ref, "other": oth}
    distance = lsimModel(inDict)

    # move results from each original channel and sequence back into a
    # new channel and sequence dimension
    distance = torch.reshape(distance, (sizeBatch, sizeSeq, sizeChannel))

    return distance


def channel_mse_2d(prediction: torch.Tensor, groundTruth: torch.Tensor, channel: int) -> torch.Tensor:
    """
    Per-sequence MSE for one channel.
    Input shape: B S C W H
    Output shape: S
    """
    if prediction.shape[2] <= channel:
        return torch.zeros(prediction.shape[1], device=prediction.device)

    return F.mse_loss(
        prediction[:, :, channel:channel + 1],
        groundTruth[:, :, channel:channel + 1],
        reduction="none"
    ).mean((0, 2, 3, 4))


def divergence_2d(x: torch.Tensor) -> torch.Tensor:
    """
    2D divergence.
    Input shape: B S C W H

    Assumes:
        channel 0 = u
        channel 1 = v

    Output shape: B S 1 W H
    """
    if x.shape[2] < 2:
        return torch.zeros(
            x.shape[0],
            x.shape[1],
            1,
            x.shape[3],
            x.shape[4],
            device=x.device,
            dtype=x.dtype,
        )

    du_dx, _ = torch.gradient(x[:, :, 0:1], dim=(3, 4))
    _, dv_dy = torch.gradient(x[:, :, 1:2], dim=(3, 4))

    return du_dx + dv_dy


def vorticity_2d(x: torch.Tensor) -> torch.Tensor:
    """
    2D scalar vorticity.
    omega = dv/dx - du/dy

    Input shape: B S C W H

    Assumes:
        channel 0 = u
        channel 1 = v

    Output shape: B S 1 W H
    """
    if x.shape[2] < 2:
        return torch.zeros(
            x.shape[0],
            x.shape[1],
            1,
            x.shape[3],
            x.shape[4],
            device=x.device,
            dtype=x.dtype,
        )

    dv_dx, _ = torch.gradient(x[:, :, 1:2], dim=(3, 4))
    _, du_dy = torch.gradient(x[:, :, 0:1], dim=(3, 4))

    return dv_dx - du_dy


class PredictionLoss(nn.modules.loss._Loss):

    def __init__(self, p_l: LossParams, dimension: int, simFields: List[str], useGPU: bool):
        super(PredictionLoss, self).__init__()
        self.p_l = p_l
        self.dimension = dimension
        self.simFields = simFields
        self.useGPU = useGPU

        self.lsim = None

        # Only load LSiM when an LSiM loss is actually enabled.
        if self.dimension == 2 and (self.p_l.recLSIM > 0 or self.p_l.predLSIM > 0):
            self.lsim = LSIM_Model(baseType="lsim", isTrain=False, useGPU=self.useGPU)
            self.lsim.load("src/lsim/models/LSiM.pth")
            self.lsim.eval()

            # freeze lsim weights
            for param in self.lsim.parameters():
                param.requires_grad = False

    def forward(
        self,
        prediction: torch.Tensor,
        groundTruth: torch.Tensor,
        latentSpace: torch.Tensor,
        vaeMeanVar: Tuple[torch.Tensor, torch.Tensor],
        weighted: bool = True,
        fadePredWeight: float = 1,
        noLSIM: bool = False,
        ignorePredLSIMSteps: int = 0
    ) -> Tuple[torch.Tensor, dict, dict]:

        assert not ((self.p_l.predMSE > 0) and prediction.shape[1] <= 1), \
            "Sequence length too small for prediction errors!"

        device = prediction.device

   
        # Initialise loss terms

        lossRecMSE = torch.zeros(1, device=device)
        lossPredMSE = torch.zeros(1, device=device)
        lossRecLSIM = torch.zeros(1, device=device)
        lossPredLSIM = torch.zeros(1, device=device)

        lossRegMeanStd = torch.zeros(1, device=device)
        lossRegDiv = torch.zeros(1, device=device)
        lossRegVort = torch.zeros(1, device=device)
        lossRegVaeKLDiv = torch.zeros(1, device=device)
        lossRegLatStep = torch.zeros(1, device=device)

        # Extra evaluation metrics
        lossPredMSE_u = torch.zeros(1, device=device)
        lossPredMSE_v = torch.zeros(1, device=device)
        lossPredMSE_pres = torch.zeros(1, device=device)
        lossPredDivMean = torch.zeros(1, device=device)
        lossPredVortMSE = torch.zeros(1, device=device)
        lossTemporalMSE = torch.zeros(1, device=device)

  
        # MSE sequence loss
   
        if self.dimension == 2:
            seqMSE = F.mse_loss(prediction, groundTruth, reduction="none").mean((0, 2, 3, 4))

        elif self.dimension == 3:
            seqMSE = F.mse_loss(prediction, groundTruth, reduction="none").mean((0, 3, 4, 5))

            if self.p_l.extraMSEvelZ > 0:
                seqMSE[:, 2:3] = self.p_l.extraMSEvelZ * seqMSE[:, 2:3]

            seqMSE = torch.mean(seqMSE, dim=1)

        else:
            raise ValueError(f"Unsupported dimension: {self.dimension}")

        if weighted:
            if self.p_l.recMSE > 0:
                lossRecMSE = self.p_l.recMSE * seqMSE[0]

            if self.p_l.predMSE > 0 and fadePredWeight > 0:
                lossPredMSE = self.p_l.predMSE * fadePredWeight * torch.mean(seqMSE[1:])
        else:
            lossRecMSE = seqMSE[0]
            lossPredMSE = torch.mean(seqMSE[1:])

    
        # LSIM loss
   
        seqLSIM = torch.zeros(prediction.shape[1], device=device)

        if self.dimension == 2:
            numFields = self.dimension + len(self.simFields)

            if not noLSIM and self.lsim is not None:
                if weighted:
                    if self.p_l.predLSIM > 0:
                        if self.p_l.recLSIM > 0:
                            seqLSIM = loss_lsim(
                                self.lsim,
                                prediction[:, :, 0:numFields],
                                groundTruth[:, :, 0:numFields],
                            ).mean((0, 2))
                        else:
                            predLSIMStart = 1 + ignorePredLSIMSteps
                            predSeqLSIM = loss_lsim(
                                self.lsim,
                                prediction[:, predLSIMStart:, 0:numFields],
                                groundTruth[:, predLSIMStart:, 0:numFields],
                            ).mean((0, 2))

                            seqLSIM = torch.cat([
                                torch.zeros(predLSIMStart, device=device),
                                predSeqLSIM,
                            ])

                    elif self.p_l.recLSIM > 0:
                        recSeqLSIM = loss_lsim(
                            self.lsim,
                            prediction[:, 0:1, 0:numFields],
                            groundTruth[:, 0:1, 0:numFields],
                        ).mean((0, 2))

                        seqLSIM = torch.cat([
                            recSeqLSIM,
                            torch.zeros(prediction.shape[1] - 1, device=device),
                        ])

                    if self.p_l.recLSIM > 0:
                        lossRecLSIM = self.p_l.recLSIM * seqLSIM[0]

                    if self.p_l.predLSIM > 0 and fadePredWeight > 0:
                        lossPredLSIM = self.p_l.predLSIM * fadePredWeight * torch.mean(seqLSIM[1:])

                else:
                    seqLSIM = loss_lsim(
                        self.lsim,
                        prediction[:, :, 0:numFields],
                        groundTruth[:, :, 0:numFields],
                    ).mean((0, 2))

                    lossRecLSIM = seqLSIM[0]
                    lossPredLSIM = torch.mean(seqLSIM[1:])


        # Mean/std regularisation

        if weighted and self.p_l.regMeanStd > 0:
            if self.dimension == 2:
                meanDiff = torch.abs(groundTruth.mean((3, 4)) - prediction.mean((3, 4)))
                stdDiff = torch.abs(groundTruth.std((3, 4)) - prediction.std((3, 4)))

            elif self.dimension == 3:
                meanDiff = torch.abs(groundTruth.mean((3, 4, 5)) - prediction.mean((3, 4, 5)))
                stdDiff = torch.abs(groundTruth.std((3, 4, 5)) - prediction.std((3, 4, 5)))

            else:
                raise ValueError(f"Unsupported dimension: {self.dimension}")

            meanStd = meanDiff.mean() + stdDiff.mean()
            lossRegMeanStd = self.p_l.regMeanStd * meanStd

     
        # Extra 2D evaluation metrics and physics-aware loss terms
        # Assumes simFields=["vel", "pres"]:
        # channel 0 = u
        # channel 1 = v
        # channel 2 = pressure

        if self.dimension == 2 and prediction.shape[1] > 1:
            seqMSE_u = channel_mse_2d(prediction, groundTruth, 0)
            seqMSE_v = channel_mse_2d(prediction, groundTruth, 1)
            seqMSE_pres = channel_mse_2d(prediction, groundTruth, 2)

            lossPredMSE_u = torch.mean(seqMSE_u[1:])
            lossPredMSE_v = torch.mean(seqMSE_v[1:])
            lossPredMSE_pres = torch.mean(seqMSE_pres[1:])

            divPred = divergence_2d(prediction)
            lossPredDivMean = torch.abs(divPred[:, 1:]).mean()

            vortPred = vorticity_2d(prediction)
            vortTrue = vorticity_2d(groundTruth)
            lossPredVortMSE = F.mse_loss(vortPred[:, 1:], vortTrue[:, 1:], reduction="mean")

            predDt = prediction[:, 1:] - prediction[:, :-1]
            trueDt = groundTruth[:, 1:] - groundTruth[:, :-1]
            lossTemporalMSE = F.mse_loss(predDt, trueDt, reduction="mean")

            if weighted and self.p_l.regDiv > 0:
                lossRegDiv = self.p_l.regDiv * lossPredDivMean

            regVort = getattr(self.p_l, "regVort", 0.0)
            if weighted and regVort > 0:
                lossRegVort = regVort * lossPredVortMSE

   
        # 3D divergence regularisation
        if weighted and self.dimension == 3 and self.p_l.regDiv > 0:
            vx_dx, _, _ = torch.gradient(prediction[:, :, 0:1], dim=(3, 4, 5))
            _, vy_dy, _ = torch.gradient(prediction[:, :, 1:2], dim=(3, 4, 5))
            _, _, vz_dz = torch.gradient(prediction[:, :, 2:3], dim=(3, 4, 5))

            div = vx_dx + vy_dy + vz_dz
            lossRegDiv = self.p_l.regDiv * torch.abs(div[:, 1:]).mean()


        # KL divergence regularisation for VAE
        if weighted and self.p_l.regVae > 0 and vaeMeanVar[0] is not None and vaeMeanVar[1] is not None:
            vaeMean = vaeMeanVar[0]
            vaeLogVar = vaeMeanVar[1]
            lossRegVaeKLDiv = -0.5 * torch.mean(
                1 + vaeLogVar - vaeMean.pow(2) - vaeLogVar.exp()
            )
            lossRegVaeKLDiv = self.p_l.regVae * lossRegVaeKLDiv


        # Latent space step regularisation
        if weighted and self.p_l.regLatStep > 0 and latentSpace.shape[1] > 1:
            latFirst = latentSpace[:, 0:latentSpace.shape[1] - 1]
            latSecond = latentSpace[:, 1:latentSpace.shape[1]]
            lossRegLatStep = self.p_l.regLatStep * torch.mean(torch.abs(latFirst - latSecond))


        # Final loss
        loss = (
            lossRecMSE
            + lossRecLSIM
            + lossPredMSE
            + lossPredLSIM
            + lossRegMeanStd
            + lossRegDiv
            + lossRegVort
            + lossRegVaeKLDiv
            + lossRegLatStep
        )

        lossParts = {
            "lossFull": loss,

            "lossRecMSE": lossRecMSE,
            "lossPredMSE": lossPredMSE,

            "lossPredMSE_u": lossPredMSE_u,
            "lossPredMSE_v": lossPredMSE_v,
            "lossPredMSE_pres": lossPredMSE_pres,

            "lossRecLSIM": lossRecLSIM,
            "lossPredLSIM": lossPredLSIM,

            "lossRegMeanStd": lossRegMeanStd,

            "lossRegDiv": lossRegDiv,
            "lossPredDivMean": lossPredDivMean,

            "lossRegVort": lossRegVort,
            "lossPredVortMSE": lossPredVortMSE,

            "lossTemporalMSE": lossTemporalMSE,

            "lossRegVaeKLDiv": lossRegVaeKLDiv,
            "lossRegLatStep": lossRegLatStep,
        }

        # Keep this simple to avoid breaking LossHistory.
        lossSeq = {
            "MSE": seqMSE,
            "LSIM": seqLSIM,
        }

        return loss, lossParts, lossSeq
