"""Minimal INC training configuration for the portfolio repository.

This example mirrors the final single-step ACDM setup while keeping paths
portable. Set DATA_ROOT to the directory containing `128_inc`.
"""

import os

import torch
from torch.utils.data import DataLoader, RandomSampler

from turbpred.data_transformations import Transforms
from turbpred.logger import Logger
from turbpred.loss import PredictionLoss
from turbpred.loss_history import LossHistory
from turbpred.model import PredictionModel
from turbpred.params import DataParams, LossParams, ModelParamsDecoder, TrainingParams
from turbpred.trainer_diffusion import TrainerDiffusion
from turbpred.turbulence_dataset import TurbulenceDataset


def main() -> None:
    data_root = os.environ.get("DATA_ROOT", "data")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full training configuration.")

    p_d = DataParams(
        batch=int(os.environ.get("BATCH_SIZE", "2")),
        augmentations=["normalize"],
        sequenceLength=[3, 2],
        randSeqOffset=True,
        dataSize=[128, 64],
        dimension=2,
        simFields=["pres"],
        simParams=["rey"],
        normalizeMode="incMixed",
    )
    p_t = TrainingParams(
        epochs=int(os.environ.get("EPOCHS", "70")),
        lr=float(os.environ.get("LR", "1e-4")),
    )
    p_l = LossParams()
    p_l.recLSIM = 0.0
    p_l.predLSIM = 0.0
    p_l.regDiv = float(os.environ.get("REG_DIV", "0.0"))
    p_l.regVort = float(os.environ.get("REG_VORT", "0.01"))

    p_md = ModelParamsDecoder(
        arch="direct-ddpm+Prev",
        diffSteps=20,
        diffSchedule="linear",
        diffCondIntegration="noisy",
        trainingNoise=0.0,
    )

    train_set = TurbulenceDataset(
        "Training",
        [data_root],
        filterTop=["128_inc"],
        filterSim=[(10, 80)],
        filterFrame=[(250, 1100)],
        sequenceLength=[p_d.sequenceLength],
        randSeqOffset=True,
        simFields=p_d.simFields,
        simParams=p_d.simParams,
        printLevel="sim",
    )
    train_set.transform = Transforms(p_d)
    loader = DataLoader(
        train_set,
        sampler=RandomSampler(train_set),
        batch_size=p_d.batch,
        drop_last=True,
        num_workers=int(os.environ.get("NUM_WORKERS", "4")),
        pin_memory=True,
    )

    model = PredictionModel(p_d, p_t, p_l, None, p_md, None, "", True)
    optimizer = torch.optim.Adam(model.parameters(), lr=p_t.lr, weight_decay=p_t.weightDecay)
    logger = Logger("portfolio/inc_vorticity", addNumber=True)
    logger.setup(model, optimizer)
    history = LossHistory(
        "_train", "Training", logger.tfWriter, len(loader), 0, 1,
        printInterval=1, logInterval=1, simFields=p_d.simFields,
    )
    trainer = TrainerDiffusion(model, loader, optimizer, history, logger.tfWriter, p_t, p_l)

    for epoch in range(p_t.epochs):
        trainer.trainingStep(epoch)
        logger.saveTrainState(epoch)

    logger.close()


if __name__ == "__main__":
    main()
