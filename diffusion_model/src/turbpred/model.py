import numpy as np
import logging, os
import torch
import torch.nn as nn

from neuralop.models import FNO

from turbpred.model_encoder import EncoderModelSkip, DecoderModelSkip
from turbpred.model_latent_transformer import LatentModelTransformerEnc, LatentModelTransformerDec, LatentModelTransformer, LatentModelTransformerMGN, LatentModelTransformerMGNParamEmb
from turbpred.model_diffusion import DiffusionModel
from turbpred.model_diffusion_blocks import Unet
from turbpred.params import DataParams, TrainingParams, LossParams, ModelParamsEncoder, ModelParamsDecoder, ModelParamsLatent

from turbpred.model_dfpnet import DfpNet
from turbpred.model_resnet import DilatedResNet
from turbpred.model_refiner import PDERefiner

class PredictionModel(nn.Module):
    p_d: DataParams
    p_t: TrainingParams
    p_l: LossParams
    p_me: ModelParamsEncoder
    p_md: ModelParamsDecoder
    p_ml: ModelParamsLatent
    useGPU: bool

    def __init__(self, p_d:DataParams, p_t:TrainingParams, p_l:LossParams, p_me:ModelParamsEncoder, p_md:ModelParamsDecoder,
                p_ml:ModelParamsLatent, pretrainPath:str="", useGPU:bool=True):
        super(PredictionModel, self).__init__()

        self.p_d = p_d
        self.p_t = p_t
        self.p_l = p_l
        self.p_me = p_me
        self.p_md = p_md
        self.p_ml = p_ml
        self.useGPU = useGPU

        if (self.p_me and self.p_me.pretrained) or (self.p_md and self.p_md.pretrained) or (self.p_ml and self.p_ml.pretrained):
            if pretrainPath:
                loadedPretrainedWeightDict = torch.load(pretrainPath)

        # ENCODER
        if self.p_me:
            if self.p_me.arch == "skip":
                self.modelEncoder = EncoderModelSkip(p_d, p_me, p_ml, p_d.dimension)
            else:
                raise ValueError("Unknown encoder architecture!")

            # load pretraining weights
            if pretrainPath and self.p_me.pretrained:
                self.modelEncoder.load_state_dict(loadedPretrainedWeightDict["stateDictEncoder"])

            # freeze weights
            if self.p_me.frozen:
                for param in self.modelEncoder.parameters():
                    param.requires_grad = False
        else:
            self.modelEncoder = None


        # DECODER
        if self.p_md:
            if self.p_md.arch == "skip":
                self.modelDecoder = DecoderModelSkip(p_d, p_me, p_md, p_ml, p_d.dimension)

            elif self.p_md.arch in ["unet", "unet+Prev", "unet+2Prev", "unet+3Prev",
                                    "dil_resnet", "dil_resnet+Prev", "dil_resnet+2Prev", "dil_resnet+3Prev",
                                    "resnet", "resnet+Prev", "resnet+2Prev", "resnet+3Prev",
                                    "fno", "fno+Prev", "fno+2Prev", "fno+3Prev",
                                    "dfp", "dfp+Prev", "dfp+2Prev", "dfp+3Prev",]:
                if "+Prev" in self.p_md.arch:
                    prevSteps = 2
                elif "+2Prev" in self.p_md.arch:
                    prevSteps = 3
                elif "+3Prev" in self.p_md.arch:
                    prevSteps = 4
                else:
                    prevSteps = 1

                inChannels = prevSteps * (self.p_d.dimension + len(self.p_d.simFields) + len(self.p_d.simParams))
                outChannels = self.p_d.dimension + len(self.p_d.simFields) + len(self.p_d.simParams)

                if "unet" in self.p_md.arch:
                    self.modelDecoder = Unet(dim=self.p_d.dataSize[0], out_dim=outChannels, channels=inChannels,
                                        dim_mults=(1,1,1), use_convnext=True, convnext_mult=1, with_time_emb=False)

                elif "resnet" in self.p_md.arch:
                    self.modelDecoder = DilatedResNet(inFeatures=inChannels, outFeatures=outChannels, blocks=4, features=self.p_md.decWidth, dilate="dil_" in self.p_md.arch)

                elif "fno" in self.p_md.arch:
                    self.modelDecoder = FNO(n_modes=(self.p_md.fnoModes[0],self.p_md.fnoModes[1]), hidden_channels=self.p_md.decWidth, in_channels=inChannels, out_channels=outChannels, n_layers=4)

                elif "dfp" in self.p_md.arch:
                    self.modelDecoder = DfpNet(inChannels=inChannels, outChannels=outChannels, blockChannels=self.p_md.decWidth)

                else:
                    raise ValueError("Unknown decoder architecture")


            elif self.p_md.arch in ["decode-ddpm", "decode-ddim", "direct-ddpm+First", "direct-ddim+First",
                                    "direct-ddpm", "direct-ddim", "direct-ddpm+Prev", "direct-ddim+Prev",
                                    "direct-ddpm+2Prev", "direct-ddim+2Prev", "direct-ddpm+3Prev", "direct-ddim+3Prev", 
                                    "dfp-ddpm", "dfp-ddpm+Prev", "dfp-ddpm+2Prev", "dfp-ddpm+3Prev",
                                    "direct-ddpm+Enc", "direct-ddim+Enc", "hybrid-ddpm+Lat", "hybrid-ddim+Lat"]:
                stateChannels = self._state_channels()
                futureSteps = self._future_steps()
                isMultiStep = self._is_multistep()

                if self.p_md.arch in ["decode-ddpm", "decode-ddim"]:
                    condChannels = self.p_me.latentSize + len(self.p_d.simParams)
                elif self.p_md.arch in ["direct-ddpm", "direct-ddim", "dfp-ddpm"]:
                    condChannels = stateChannels
                elif self.p_md.arch in ["direct-ddpm+First", "direct-ddim+First", "direct-ddpm+Prev", "direct-ddim+Prev", "dfp-ddpm+Prev"]:
                    condChannels = 2 * stateChannels
                elif self.p_md.arch in ["direct-ddpm+2Prev", "direct-ddim+2Prev", "dfp-ddpm+2Prev"]:
                    condChannels = 3 * stateChannels
                elif self.p_md.arch in ["direct-ddpm+3Prev", "direct-ddim+3Prev", "dfp-ddpm+3Prev"]:
                    condChannels = 4 * stateChannels
                elif self.p_md.arch in ["direct-ddpm+Enc", "direct-ddim+Enc", "hybrid-ddpm+Lat", "hybrid-ddim+Lat"]:
                    condChannels = stateChannels + (self.p_me.latentSize + len(self.p_d.simParams))

                # Multi-step mode is only enabled explicitly by setting:
                #     p_d.multiStep = True
                #     p_d.historySteps = C
                #     p_d.futureSteps = P
                #
                # This keeps all original single-step baseline scripts backward-compatible.
                if isMultiStep and self.p_md.arch in [
                    "direct-ddpm", "direct-ddim", "direct-ddpm+Prev", "direct-ddim+Prev",
                    "direct-ddpm+2Prev", "direct-ddim+2Prev", "direct-ddpm+3Prev", "direct-ddim+3Prev",
                    "dfp-ddpm", "dfp-ddpm+Prev", "dfp-ddpm+2Prev", "dfp-ddpm+3Prev"
                ]:
                    condChannels = self._history_steps() * stateChannels

                # DiffusionModel internally adds len(simFields)+len(simParams) to the
                # passed dimension-like argument. For multi-step channel concatenation,
                # we need target channels = futureSteps * stateChannels.
                diffusionDimension = self.p_d.dimension
                if isMultiStep and futureSteps > 1:
                    diffusionDimension = futureSteps * stateChannels - len(self.p_d.simFields) - len(self.p_d.simParams)

                self.modelDecoder = DiffusionModel(p_d, p_md, diffusionDimension, condChannels=condChannels)

            elif self.p_md.arch == "refiner":
                condChannels = self.p_d.dimension + len(self.p_d.simFields) + len(self.p_d.simParams)
                self.modelDecoder = PDERefiner(p_d, p_md, condChannels=condChannels)


            elif self.p_md.arch in ["skip+finetune-ddpm", "skip+finetune-ddim"]:
                self.modelDecoder = nn.ModuleList([
                    DecoderModelSkip(p_d, p_me, p_md, p_ml, p_d.dimension),
                    DiffusionModel(p_d, p_md, p_d.dimension, condChannels=self.p_d.dimension + len(self.p_d.simFields) + len(self.p_d.simParams))
                ])

            elif self.p_md.arch in ["skip+hybrid-ddpm", "skip+hybrid-ddim"]:
                self.modelDecoder = nn.ModuleList([
                    DecoderModelSkip(p_d, p_me, p_md, p_ml, p_d.dimension),
                    DiffusionModel(p_d, p_md, p_d.dimension, condChannels=2 * (self.p_d.dimension + len(self.p_d.simFields) + len(self.p_d.simParams)))
                ])

            else:
                raise ValueError("Unknown decoder architecture!")

            # load pretraining weights
            if pretrainPath and self.p_md.pretrained:
                if self.p_md.arch in ["skip+finetune-ddpm", "skip+finetune-ddim", "skip+hybrid-ddpm", "skip+hybrid-ddim"]:
                    self.modelDecoder[0].load_state_dict(loadedPretrainedWeightDict["stateDictDecoder"])
                else:
                    self.modelDecoder.load_state_dict(loadedPretrainedWeightDict["stateDictDecoder"])

            # freeze weights
            if self.p_md.frozen:
                if self.p_md.arch in ["skip+finetune-ddpm", "skip+finetune-ddim", "skip+hybrid-ddpm", "skip+hybrid-ddim"]:
                    for param in self.modelDecoder[0].parameters():
                        param.requires_grad = False
                else:
                    for param in self.modelDecoder.parameters():
                        param.requires_grad = False
        else:
            self.modelDecoder = None

        # LATENT MODEL
        if self.p_ml:
            if self.p_ml.arch == "transformerEnc":
                self.modelLatent = LatentModelTransformerEnc(p_d, p_me, p_ml, False)
            elif self.p_ml.arch == "transformerDec":
                self.modelLatent = LatentModelTransformerDec(p_d, p_me, p_ml)
            elif self.p_ml.arch == "transformer":
                self.modelLatent = LatentModelTransformer(p_d, p_me, p_ml)
            elif self.p_ml.arch == "transformerMGN":
                self.modelLatent = LatentModelTransformerMGN(p_d, p_me, p_ml)
                self.modelLatentParamEmb = LatentModelTransformerMGNParamEmb(p_d, p_me)
            else:
                raise ValueError("Unknown latent architecture!")

            # load pretraining weights
            if pretrainPath and self.p_ml.pretrained:
                self.modelLatent.load_state_dict(loadedPretrainedWeightDict["stateDictLatent"])

            # freeze weights
            if self.p_ml.frozen:
                for param in self.modelLatent.parameters():
                    param.requires_grad = False
        else:
            self.modelLatent = None

        self.to("cuda" if self.useGPU else "cpu")



    # Multi-step helper functions.
    # These helpers are deliberately backward-compatible:
    # existing baseline scripts do not set p_d.multiStep, so they keep
    # the original single-step behaviour.

    def _is_multistep(self) -> bool:
        return bool(getattr(self.p_d, "multiStep", False))

    def _state_channels(self) -> int:
        return self.p_d.dimension + len(self.p_d.simFields) + len(self.p_d.simParams)

    def _future_steps(self) -> int:
        if self._is_multistep():
            return int(getattr(self.p_d, "futureSteps", 1))
        return 1

    def _history_steps(self) -> int:
        if self._is_multistep():
            return int(getattr(self.p_d, "historySteps", self._arch_prev_steps()))
        return self._arch_prev_steps()

    def _arch_prev_steps(self) -> int:
        if "+Prev" in self.p_md.arch:
            return 2
        elif "+2Prev" in self.p_md.arch:
            return 3
        elif "+3Prev" in self.p_md.arch:
            return 4
        else:
            return 1

    def _concat_time_as_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Convert B,T,C,W,H to B,1,T*C,W,H for diffusion."""
        b, t, c, w, h = x.shape
        return x.reshape(b, 1, t * c, w, h)

    def _split_channels_as_time(self, x: torch.Tensor, steps: int) -> torch.Tensor:
        """Convert B,1,T*C,W,H back to B,T,C,W,H."""
        b, s, tc, w, h = x.shape
        if s != 1:
            raise ValueError("Expected a single diffusion block dimension when splitting multi-step output.")
        if tc % steps != 0:
            raise ValueError("Multi-step output channels are not divisible by futureSteps.")
        c = tc // steps
        return x.reshape(b, steps, c, w, h)

    def _replace_simparams_single(self, result: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """Replace simParam channels for normal single-step output."""
        if self.p_d.simParams:
            result[:, :, -len(self.p_d.simParams):] = truth[:, :, -len(self.p_d.simParams):]
        return result

    def _replace_simparams_block(self, block: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """Replace simParam channels for multi-step output B,P,C,W,H."""
        if self.p_d.simParams:
            block[:, :, -len(self.p_d.simParams):] = truth[:, :, -len(self.p_d.simParams):]
        return block



    def forward(self, data:torch.Tensor, simParameters:torch.Tensor, useLatent:bool=True, stepsLong:int=-1) -> torch.Tensor:
        device = "cuda" if self.useGPU else "cpu"
        d = data.to(device)
        simParam = simParameters.to(device) if simParameters is not None else None

        # ENCODING - LATENT MODEL - DECODING
        if not (self.p_md.arch in ["unet", "unet+Prev", "unet+2Prev", "unet+3Prev",
                "dil_resnet", "dil_resnet+Prev", "dil_resnet+2Prev", "dil_resnet+3Prev",
                "resnet", "resnet+Prev", "resnet+2Prev", "resnet+3Prev",
                "fno", "fno+Prev", "fno+2Prev", "fno+3Prev",
                "dfp", "dfp+Prev", "dfp+2Prev", "dfp+3Prev",
                "refiner",
                "direct-ddpm", "direct-ddim", "direct-ddpm+First", "direct-ddim+First", 
                "direct-ddpm+Prev", "direct-ddim+Prev", "direct-ddpm+2Prev", "direct-ddim+2Prev",
                "direct-ddpm+3Prev", "direct-ddim+3Prev", "direct-ddpm+Enc", "direct-ddim+Enc",
                "dfp-ddpm", "dfp-ddpm+Prev", "dfp-ddpm+2Prev", "dfp-ddpm+3Prev",]):

            latentSpace = torch.zeros(d.shape[0], d.shape[1], self.p_me.latentSize)
            latentSpace = latentSpace.to(device)
            if not self.modelLatent or not useLatent:
                # no latent model -> fully process sequence with AE
                latentSpace = self.modelEncoder(d)
            else:
                if isinstance(self.modelLatent, LatentModelTransformerEnc):
                    latentSpace = self.forwardTransEnc(d, latentSpace, simParam)

                elif isinstance(self.modelLatent, LatentModelTransformerDec) or isinstance(self.modelLatent, LatentModelTransformer):
                    latentSpace = self.forwardTransDec(d, latentSpace, simParam)

                elif isinstance(self.modelLatent, LatentModelTransformerMGN):
                    latentSpace = self.forwardTransMGN(d, latentSpace, simParam)

                else:
                    raise ValueError("Invalid latent model!")

            if "decode" in self.p_md.arch:
                prediction = self.forwardDiffusionDecode(d, latentSpace, simParam)
                return prediction, latentSpace, (None, None)

            elif "finetune" in self.p_md.arch:
                prediction = self.forwardDiffusionFinetune(d, latentSpace, simParam)
                return prediction, latentSpace, (None, None)

            elif "hybrid" in self.p_md.arch:
                prediction = self.forwardDiffusionHybrid(d, latentSpace, simParam)
                return prediction, latentSpace, (None, None)

            else:
                prediction, vaeMeanVar = self.modelDecoder(latentSpace, simParam)
                return prediction, latentSpace, vaeMeanVar


        # DIRECT PREDICTION OF NEXT FRAME WITH DIFFERENT ARCHITECTURES
        else:
            if isinstance(self.modelDecoder, Unet) or isinstance(self.modelDecoder, DilatedResNet) or isinstance(self.modelDecoder, FNO):
                if stepsLong > 0 and (not self.training):
                    prediction = self.forwardDirectLongGPUEfficient(d, steps=stepsLong)
                else:
                    prediction = self.forwardDirect(d)
                return prediction, None, (None, None)

            else:
                if stepsLong > 0 and (not self.training):
                    prediction = self.forwardDiffusionDirectLongGPUEfficient(d, simParam, steps=stepsLong)
                else:
                    prediction = self.forwardDiffusionDirect(d, simParam)

                return prediction, None, (None, None)




    # Transformer encoder latent model
    def forwardTransEnc(self, d:torch.Tensor, latentSpace:torch.Tensor, simParam:torch.Tensor) -> torch.Tensor:
        sizeSeq = d.shape[1]

        # transformer encoder latent model to predict single next step
        if self.training and not self.p_ml.transTrainUnroll:
            encLatentSpace = self.modelEncoder(d)
            transLatentSpace = self.modelLatent(encLatentSpace, simParam)
            latentSpace = torch.concat([encLatentSpace[:,:1], transLatentSpace[:,:-1]], dim=1)

        # transformer encoder latent model to predict all steps from first one
        else:
            latentSpace[:,0:1] = self.modelEncoder(d[:,0:1])
            for i in range(1,sizeSeq):
                start = max(0, i-self.p_ml.maxInputLen) if self.p_ml.maxInputLen > 0 else 0
                transLatentSpace = self.modelLatent(latentSpace[:,start:i], simParam[:,start:i] if simParam is not None else None)
                latentSpace[:,i] = transLatentSpace[:,-1]

        return latentSpace


    # Transformer decoder latent model
    def forwardTransDec(self, d:torch.Tensor, latentSpace:torch.Tensor, simParam:torch.Tensor) -> torch.Tensor:
        sizeSeq = d.shape[1]

        # transformer latent model to predict single next step
        if self.training and not self.p_ml.transTrainUnroll:
            encLatentSpace = self.modelEncoder(d)
            transLatentSpace = self.modelLatent(encLatentSpace[:,:-1], encLatentSpace[:,1:], simParam[:,:-1] if simParam is not None else None, simParam[:,1:] if simParam is not None else None)
            latentSpace = torch.concat([encLatentSpace[:,:1], transLatentSpace], dim=1)

        # transformer latent model to predict all steps from first one
        else:
            latentSpace[:,0:1] = self.modelEncoder(d[:,0:1])
            for i in range(1,sizeSeq):
                if self.p_ml.transTargetFull:
                    start = max(0, i-self.p_ml.maxInputLen) if self.p_ml.maxInputLen > 0 else 0
                    transLatentSpace = self.modelLatent(latentSpace[:,start:i], latentSpace[:,start:i], simParam[:,start:i] if simParam is not None else None, simParam[:,start:i] if simParam is not None else None)
                    latentSpace[:,i] = transLatentSpace[:,-1]
                    #latentSpace[:,i] = latentSpace[:,i-1] + transLatentSpace[:,-1]
                else:
                    transLatentSpace = self.modelLatent(latentSpace[:,:i], latentSpace[:,i-1:i], simParam[:,:i] if simParam is not None else None, simParam[:,i-1:i] if simParam is not None else None)
                    latentSpace[:,i:i+1] = transLatentSpace

        return latentSpace


    # Transformer latent model according to MeshGraphNet paper
    def forwardTransMGN(self, d:torch.Tensor, latentSpace:torch.Tensor, simParam:torch.Tensor) -> torch.Tensor:
        sizeBatch, sizeSeq = d.shape[0], d.shape[1]

        latentSpace = torch.zeros(sizeBatch, sizeSeq+1, self.p_me.latentSize)
        latentSpace = latentSpace.to("cuda" if self.useGPU else "cpu")

        if simParam is not None:
            latentSpace[:,0] = self.modelLatentParamEmb(simParam[:,0]) # only use scalar simParam input
        latentSpace[:,1:2] = self.modelEncoder(d[:,0:1])
        for i in range(2,sizeSeq+1):
            start = max(1, i-self.p_ml.maxInputLen) if self.p_ml.maxInputLen > 0 else 1
            transInput = torch.concat([latentSpace[:,0:1], latentSpace[:,start:i]], dim=1)
            transLatentSpace = self.modelLatent(transInput, latentSpace[:,i-1:i])
            latentSpace[:,i:i+1] = latentSpace[:,i-1:i] + transLatentSpace
        latentSpace = latentSpace[:,1:] # discard param embedding

        return latentSpace


    # Direct prediction of next step via U-Net, ResNet, FNO, etc.
    def forwardDirect(self, d:torch.Tensor) -> torch.Tensor:
        sizeBatch, sizeSeq = d.shape[0], d.shape[1]

        if "+Prev" in self.p_md.arch:
            prevSteps = 2
        elif "+2Prev" in self.p_md.arch:
            prevSteps = 3
        elif "+3Prev" in self.p_md.arch:
            prevSteps = 4
        else:
            prevSteps = 1

        prediction = []
        #for i in range(4):
        for i in range(prevSteps): # no prediction of first steps
            if self.training:
                trainNoise = self.p_md.trainingNoise * torch.normal(torch.zeros_like(d[:,i]), torch.ones_like(d[:,i]))
                prediction += [d[:,i] + trainNoise]
            else:
                prediction += [d[:,i]]

        for i in range(prevSteps, sizeSeq):
            uIn = torch.concat(prediction[i-prevSteps : i], dim=1)

            if isinstance(self.modelDecoder, FNO):
                result = self.modelDecoder(uIn)
            else:
                result = self.modelDecoder(uIn, None)
    
            if self.p_d.simParams:
                result[:,-len(self.p_d.simParams):] = d[:,i,-len(self.p_d.simParams):] # replace simparam prediction with true values
            prediction += [result]

        prediction = torch.stack(prediction, dim=1)
        return prediction


    # GPU EFFICIENT VARIANT OF DIRECT PREDICTION FOR EXTREMELY LONG SEQUENCES
    def forwardDirectLongGPUEfficient(self, d:torch.Tensor, steps:int) -> torch.Tensor:
        sizeBatch, sizeSeq = d.shape[0], steps

        if "+Prev" in self.p_md.arch or "+2Prev" in self.p_md.arch or "+3Prev" in self.p_md.arch:
            raise ValueError("GPU efficient variant only supports 1 previous step!")

        prediction = [d[:,0]]

        uIn = prediction[0].to("cuda")
        self.modelDecoder = self.modelDecoder.to("cuda")
        for i in range(1, sizeSeq):

            if isinstance(self.modelDecoder, FNO):
                result = self.modelDecoder(uIn)
            else:
                result = self.modelDecoder(uIn, None)

            if self.p_d.simParams:
                result[:,-len(self.p_d.simParams):] = d[:,0,-len(self.p_d.simParams):].to("cuda") # replace simparam prediction with true values

            uIn = result

            if i % 100 == 0:
                if i % 10000 == 0:
                    print("Step %d" % i)
                result = result.to("cpu")
                prediction += [result]

        prediction = torch.stack(prediction, dim=1)
        return prediction


    # Diffusion model to directly predict next step based on different conditionings
    def forwardDiffusionDirect(self, d:torch.Tensor, simParams:torch.Tensor) -> torch.Tensor:
        sizeBatch, sizeSeq = d.shape[0], d.shape[1]
        device = "cuda" if self.useGPU else "cpu"

        isMultiStep = self._is_multistep()
        historySteps = self._history_steps()
        futureSteps = self._future_steps()
        stateChannels = self._state_channels()

        # TRAINING
        if self.training:
            if "+Enc" in self.p_md.arch:
                latentSpace = self.modelEncoder(d[:,0:1])
                l = torch.concat((latentSpace, simParams[:,0:1]), dim=2) if simParams is not None else latentSpace
                conditioning = l.unsqueeze(3).unsqueeze(4).expand(-1,-1,-1,d.shape[3],d.shape[4])

                randIndex = torch.randint(1, sizeSeq, (1,), device=d.device)
                conditioning = torch.concat((conditioning, d[:,randIndex-1:randIndex]), dim=2)
                data = d[:,randIndex]

            elif "+First" in self.p_md.arch:
                randIndex = torch.randint(1, sizeSeq, (1,), device=d.device)
                conditioning = torch.concat((d[:,0:1], d[:,randIndex-1:randIndex]), dim=2)
                data = d[:,randIndex:randIndex+1]

            else:
                # Backward-compatible multi-step branch.
                # This branch is used only when p_d.multiStep=True.
                # It converts:
                #     conditioning: B,C_hist,C,W,H -> B,1,C_hist*C,W,H
                #     target:       B,P,C,W,H      -> B,1,P*C,W,H

                if isMultiStep and futureSteps > 1:
                    requiredLen = historySteps + futureSteps
                    if sizeSeq < requiredLen:
                        raise ValueError(
                            "Multi-step training sequence is too short: "
                            f"sizeSeq={sizeSeq}, historySteps={historySteps}, futureSteps={futureSteps}."
                        )

                    # Pick a valid window. For smoke tests this often picks 0.
                    # For longer sequences this increases temporal coverage.
                    maxStart = sizeSeq - requiredLen
                    startIndex = torch.randint(0, maxStart + 1, (1,), device=d.device).item() if maxStart > 0 else 0

                    condFrames = []
                    for i in range(historySteps):
                        frame = d[:, startIndex + i : startIndex + i + 1]
                        trainNoise = self.p_md.trainingNoise * torch.normal(
                            torch.zeros_like(frame),
                            torch.ones_like(frame),
                        )
                        condFrames.append(frame + trainNoise)

                    conditioning = torch.concat(condFrames, dim=2)

                    targetBlock = d[
                        :,
                        startIndex + historySteps : startIndex + historySteps + futureSteps,
                    ]
                    data = self._concat_time_as_channels(targetBlock)

                # Original single-step branch.
                else:
                    prevSteps = self._arch_prev_steps()

                    cond = []
                    for i in range(prevSteps):
                        trainNoise = self.p_md.trainingNoise * torch.normal(
                            torch.zeros_like(d[:,i:i+1]),
                            torch.ones_like(d[:,i:i+1]),
                        )
                        cond += [d[:, i:i+1] + trainNoise] # collect input steps

                    conditioning = torch.concat(cond, dim=2) # combine along channel dimension
                    data = d[:, prevSteps:prevSteps+1]

            return self.modelDecoder(conditioning=conditioning, data=data)


        # INFERENCE
        else:
            prediction = torch.zeros_like(d, device=device)

            if "+Enc" in self.p_md.arch:
                prediction[:,0] = d[:,0] # no prediction of first step
                latentSpace = self.modelEncoder(d[:,0:1])
                l = torch.concat((latentSpace, simParams[:,0:1]), dim=2) if simParams is not None else latentSpace
                conditioning = l.unsqueeze(3).unsqueeze(4).expand(-1,-1,-1,d.shape[3],d.shape[4])
                for i in range(1,sizeSeq):
                    cond = torch.concat((conditioning, prediction[:,i-1:i]), dim=2)
                    result = self.modelDecoder(conditioning=cond, data=d[:,i-1:i])
                    result = self._replace_simparams_single(result, d[:,i:i+1])
                    prediction[:,i:i+1] = result

            elif "+First" in self.p_md.arch:
                prediction[:,0] = d[:,0] # no prediction of first step
                for i in range(1,sizeSeq):
                    cond = torch.concat((d[:,0:1], prediction[:,i-1:i]), dim=2)
                    result = self.modelDecoder(conditioning=cond, data=d[:,i-1:i])
                    result = self._replace_simparams_single(result, d[:,i:i+1])
                    prediction[:,i:i+1] = result

            else:
                # Multi-step autoregressive rollout:
                #     [s0, s1, ...] -> generate [s_next, s_next+1]
                # and then append the generated block to the prediction window.
                
                if isMultiStep and futureSteps > 1:
                    if sizeSeq < historySteps:
                        raise ValueError(
                            "Multi-step inference sequence is shorter than historySteps: "
                            f"sizeSeq={sizeSeq}, historySteps={historySteps}."
                        )

                    for i in range(historySteps):
                        prediction[:, i] = d[:, i]

                    i = historySteps
                    while i < sizeSeq:
                        condFrames = []
                        for j in range(historySteps, 0, -1):
                            condFrames.append(prediction[:, i - j : i - (j - 1)])
                        cond = torch.concat(condFrames, dim=2)

                        dummy = torch.zeros(
                            sizeBatch,
                            1,
                            futureSteps * stateChannels,
                            d.shape[3],
                            d.shape[4],
                            device=device,
                            dtype=d.dtype,
                        )

                        result = self.modelDecoder(conditioning=cond, data=dummy)
                        resultBlock = self._split_channels_as_time(result, futureSteps)

                        # Optional predict-two / rollout-one inference.
                        # The diffusion model still generates P=futureSteps frames,
                        # but only the first horizon is inserted into the rollout.
                        rolloutOne = bool(getattr(self.p_d, "rolloutOne", False))

                        if rolloutOne:
                            remaining = 1
                            resultFirst = resultBlock[:, :1]
                            resultFirst = self._replace_simparams_block(
                                resultFirst,
                                d[:, i:i+1],
                            )

                            prediction[:, i:i+1] = resultFirst
                            i += 1
                        else:
                            remaining = min(futureSteps, sizeSeq - i)
                            resultBlock = resultBlock[:, :remaining]
                            resultBlock = self._replace_simparams_block(
                                resultBlock,
                                d[:, i:i+remaining],
                            )

                            prediction[:, i:i+remaining] = resultBlock
                            i += remaining

                # Original single-step autoregressive inference.
                else:
                    prevSteps = self._arch_prev_steps()

                    for i in range(prevSteps): # no prediction of first steps
                        prediction[:,i] = d[:,i] 

                    for i in range(prevSteps, sizeSeq):
                        cond = []
                        for j in range(prevSteps,0,-1):
                            cond += [prediction[:, i-j : i-(j-1)]] # collect input steps
                        cond = torch.concat(cond, dim=2) # combine along channel dimension

                        result = self.modelDecoder(conditioning=cond, data=d[:,i-1:i]) # auto-regressive inference
                        result = self._replace_simparams_single(result, d[:,i:i+1])
                        prediction[:,i:i+1] = result

            return prediction


    # GPU EFFICIENT VARIANT FOR DIFFUSION MODELS FOR EXTREMELY LONG SEQUENCES
    def forwardDiffusionDirectLongGPUEfficient(self, d:torch.Tensor, simParams:torch.Tensor, steps:int) -> torch.Tensor:
        sizeBatch, sizeSeq = d.shape[0], steps

        # TRAINING
        if self.training:
            raise ValueError("Training not supported for GPU efficient variant!")

        # INFERENCE
        else:
            if not "+Prev" in self.p_md.arch:
                raise ValueError("GPU efficient variant only supports 2 previous step!")

            prediction = [d[:,0:1]]

            uInPrev = d[:,0:1].to("cuda")
            uIn = d[:,1:2].to("cuda")
            self.modelDecoder = self.modelDecoder.to("cuda")

            for i in range(2, sizeSeq):
                cond = torch.concat([uInPrev, uIn], dim=2) # combine along channel dimension

                result = self.modelDecoder(conditioning=cond, data=torch.zeros_like(uIn, device="cuda")) # auto-regressive inference

                if self.p_d.simParams:
                    result[:,:,-len(self.p_d.simParams):] = d[:,0:1,-len(self.p_d.simParams):].to("cuda") # replace simparam prediction with true values
                
                uInPrev = uIn
                uIn = result

                if i % 100 == 0:
                    if i % 10000 == 0:
                        print("Step %d" % i)
                    result = result.to("cpu")
                    prediction += [result]

            prediction = torch.concat(prediction, dim=1)
            return prediction


    # Decoder diffusion model conditioned on latent space
    def forwardDiffusionDecode(self, d:torch.Tensor, latentSpace:torch.Tensor, simParams:torch.Tensor) -> torch.Tensor:
        # add simulation parameters to latent space
        l = torch.concat((latentSpace, simParams), dim=2) if simParams is not None else latentSpace

        # match dimensionality
        cond = l.unsqueeze(3).unsqueeze(4).expand(-1,-1,-1,d.shape[3],d.shape[4])

        if self.training:
            return self.modelDecoder(conditioning=cond, data=d) # prediction conditioned on latent space
        else:
            prediction = self.modelDecoder(conditioning=cond, data=d)
            return prediction


    # Diffusion model conditioned on normal decoder ouput to finetune it
    def forwardDiffusionFinetune(self, d:torch.Tensor, latentSpace:torch.Tensor, simParams:torch.Tensor) -> torch.Tensor:
        sizeBatch, sizeSeq = d.shape[0], d.shape[1]

        cond, _ = self.modelDecoder[0](latentSpace, simParams)

        if self.training:
            return self.modelDecoder[1](conditioning=cond, data=d)
        else:
            prediction = self.modelDecoder[1](conditioning=cond, data=d)
            return prediction


    # Diffusion model predicts next step based on previous step and secondary transformer network conditioning
    def forwardDiffusionHybrid(self, d:torch.Tensor, latentSpace:torch.Tensor, simParams:torch.Tensor) -> torch.Tensor:
        sizeBatch, sizeSeq = d.shape[0], d.shape[1]
        if self.training:
            if "+Lat" in self.p_md.arch:
                randIndex = torch.randint(1, sizeSeq-1, (1,), device=d.device)

                l = torch.concat((latentSpace, simParams), dim=2) if simParams is not None else latentSpace
                conditioning = l.unsqueeze(3).unsqueeze(4).expand(-1,-1,-1,d.shape[3],d.shape[4])

                conditioning = torch.concat((conditioning[:,randIndex:randIndex+1], d[:,randIndex-1:randIndex]), dim=2)
                data = d[:,randIndex:randIndex+1]

                return self.modelDecoder(conditioning=conditioning, data=data)

            elif "skip+" in self.p_md.arch:
                predictionAeDec, _ = self.modelDecoder[0](latentSpace, simParams)

                randIndex = torch.randint(1, sizeSeq-1, (1,), device=d.device)

                conditioning = torch.concat((predictionAeDec[:,randIndex:randIndex+1], d[:,randIndex-1:randIndex]), dim=2)
                data = d[:,randIndex:randIndex+1]

                return self.modelDecoder[1](conditioning=conditioning, data=data)

            raise ValueError("Unknown hybrid diffusion training architecture!")

        else:
            prediction = torch.zeros_like(d, device="cuda" if self.useGPU else "cpu")
            prediction[:,0] = d[:,0] # no prediction of first and last step
            prediction[:,d.shape[1]-1] = d[:,d.shape[1]-1]

            if "+Lat" in self.p_md.arch:
                l = torch.concat((latentSpace, simParams), dim=2) if simParams is not None else latentSpace
                conditioning = l.unsqueeze(3).unsqueeze(4).expand(-1,-1,-1,d.shape[3],d.shape[4])

                for i in range(1,sizeSeq-1):
                    cond = torch.concat((conditioning[:,i:i+1], prediction[:,i-1:i]), dim=2)
                    result = self.modelDecoder(conditioning=cond, data=d[:,i-1:i])
                    if self.p_d.simParams:
                        result[:,:,-len(self.p_d.simParams):] = d[:,i:i+1,-len(self.p_d.simParams):] # replace simparam prediction with true values
                    prediction[:,i:i+1] = result

            elif "skip+" in self.p_md.arch:
                predictionAeDec, _ = self.modelDecoder[0](latentSpace, simParams)

                for i in range(1,sizeSeq-1):
                    cond = torch.concat((predictionAeDec[:,i:i+1], prediction[:,i-1:i]), dim=2)
                    result = self.modelDecoder[1](conditioning=cond, data=d[:,i-1:i])
                    if self.p_d.simParams:
                        result[:,:,-len(self.p_d.simParams):] = d[:,i:i+1,-len(self.p_d.simParams):] # replace simparam prediction with true values
                    prediction[:,i:i+1] = result

            return prediction


    def printModelInfo(self):
        pTrain = filter(lambda p: p.requires_grad, self.parameters())
        paramsTrain = sum([np.prod(p.size()) for p in pTrain])
        params = sum([np.prod(p.size()) for p in self.parameters()])

        if self.modelEncoder:
            pTrainEnc = filter(lambda p: p.requires_grad, self.modelEncoder.parameters())
            paramsTrainEnc = sum([np.prod(p.size()) for p in pTrainEnc])
            paramsEnc = sum([np.prod(p.size()) for p in self.modelEncoder.parameters()])
        if self.modelDecoder:
            pTrainDec = filter(lambda p: p.requires_grad, self.modelDecoder.parameters())
            paramsTrainDec = sum([np.prod(p.size()) for p in pTrainDec])
            paramsDec = sum([np.prod(p.size()) for p in self.modelDecoder.parameters()])
        if self.modelLatent:
            pTrainLat = filter(lambda p: p.requires_grad, self.modelLatent.parameters())
            paramsTrainLat = sum([np.prod(p.size()) for p in pTrainLat])
            paramsLat = sum([np.prod(p.size()) for p in self.modelLatent.parameters()])

        print("Weights Trainable (All): %d (%d)   %s   %s   %s" %
                (paramsTrain, params,
                ("Enc: %d (%d)" % (paramsTrainEnc, paramsEnc)) if self.modelEncoder else "",
                ("Dec: %d (%d)" % (paramsTrainDec, paramsDec)) if self.modelDecoder else "",
                ("Lat: %d (%d)" % (paramsTrainLat, paramsLat)) if self.modelLatent else ""))
        print(self)
        print("Data parameters: %s" % str(self.p_d.asDict()))
        print("Training parameters: %s" % str(self.p_t.asDict()))
        print("Loss parameters: %s" % str(self.p_l.asDict()))
        if self.p_me:
            print("Model Encoder parameters: %s" % str(self.p_me.asDict()))
        if self.p_md:
            print("Model Decoder parameters: %s" % str(self.p_md.asDict()))
        if self.p_ml:
            print("Model Latent parameters: %s" % str(self.p_ml.asDict()))
        print("")

        logging.info("Weights Trainable (All): %d (%d)   %s   %s   %s" %
                (paramsTrain, params,
                ("Enc: %d (%d)" % (paramsTrainEnc, paramsEnc)) if self.modelEncoder else "",
                ("Dec: %d (%d)" % (paramsTrainDec, paramsDec)) if self.modelDecoder else "",
                ("Lat: %d (%d)" % (paramsTrainLat, paramsLat)) if self.modelLatent else ""))
        logging.info(self)
        logging.info("Data parameters: %s" % str(self.p_d.asDict()))
        logging.info("Training parameters: %s" % str(self.p_t.asDict()))
        logging.info("Loss parameters: %s" % str(self.p_l.asDict()))
        if self.p_me:
            logging.info("Model Encoder parameters: %s" % str(self.p_me.asDict()))
        if self.p_md:
            logging.info("Model Decoder parameters: %s" % str(self.p_md.asDict()))
        if self.p_ml:
            logging.info("Model Latent parameters: %s" % str(self.p_ml.asDict()))
        logging.info("")



    @classmethod
    def load(cls, path:str, useGPU:bool=True):
        if useGPU:
            print('Loading model from %s' % path)
            loaded = torch.load(path)
        else:
            print('CPU - Loading model from %s' % path)
            loaded = torch.load(path, map_location=torch.device('cpu'))

        p_me = ModelParamsEncoder().fromDict(loaded['modelParamsEncoder']) if loaded['modelParamsEncoder'] else None
        p_md = ModelParamsDecoder().fromDict(loaded['modelParamsDecoder']) if loaded['modelParamsDecoder'] else None
        p_ml = ModelParamsLatent().fromDict(loaded['modelParamsLatent'])   if loaded['modelParamsLatent'] else None
        p_d = DataParams().fromDict(loaded['dataParams'])                  if loaded['dataParams'] else None
        p_t = TrainingParams().fromDict(loaded['trainingParams'])          if loaded['trainingParams'] else None


        # Backward-compatible multi-step checkpoint restoration.

        if p_d is not None and os.environ.get("LOAD_MULTISTEP", "0") == "1":
            p_d.historySteps = int(
                os.environ.get("LOAD_HISTORY_STEPS", "2")
            )
            p_d.futureSteps = int(
                os.environ.get("LOAD_FUTURE_STEPS", "2")
            )
            p_d.multiStep = True

            print(
                "Restoring legacy multi-step checkpoint:"
                f" historySteps={p_d.historySteps}"
                f" futureSteps={p_d.futureSteps}"
                f" multiStep={p_d.multiStep}"
            )
        p_l = LossParams().fromDict(loaded['lossParams'])                  if loaded['lossParams'] else None

        stateDictEncoder = loaded['stateDictEncoder']
        stateDictDecoder = loaded['stateDictDecoder']
        stateDictLatent = loaded['stateDictLatent']

        model = cls(p_d, p_t, p_l, p_me, p_md, p_ml, "", useGPU)

        if stateDictEncoder:
            model.modelEncoder.load_state_dict(stateDictEncoder)
        if stateDictDecoder:
            model.modelDecoder.load_state_dict(stateDictDecoder)
        if stateDictLatent:
            model.modelLatent.load_state_dict(stateDictLatent)
        model.eval()

        return model


    def save(self, basePath:str, epoch:int=-1, noPrint:bool=False):
        if not noPrint:
            print('Saving model to %s' % basePath)

        saveDict = {
            'stateDictEncoder'   : self.modelEncoder.state_dict() if self.modelEncoder else None,
            'stateDictDecoder'   : self.modelDecoder.state_dict() if self.modelDecoder else None,
            'stateDictLatent'    : self.modelLatent.state_dict() if self.modelLatent else None,
            'modelParamsEncoder' : self.p_me.asDict() if self.p_me else None,
            'modelParamsDecoder' : self.p_md.asDict() if self.p_md else None,
            'modelParamsLatent'  : self.p_ml.asDict() if self.p_ml else None,
            'dataParams'         : self.p_d.asDict() if self.p_d else None,
            'trainingParams'     : self.p_t.asDict() if self.p_t else None,
            'lossParams'         : self.p_l.asDict() if self.p_l else None,
            }

        if epoch > 0:
            path = os.path.join(basePath, "Model_E%03d.pth" % epoch)
        else:
            path = os.path.join(basePath, "Model.pth")
        torch.save(saveDict, path)

