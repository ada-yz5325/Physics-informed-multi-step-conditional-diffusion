import os
import copy
from typing import Dict

import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, SubsetRandomSampler

from turbpred.model import PredictionModel
from turbpred.logger import Logger
from turbpred.params import (
    DataParams,
    TrainingParams,
    LossParams,
    ModelParamsEncoder,
    ModelParamsDecoder,
    ModelParamsLatent,
)
from turbpred.turbulence_dataset import TurbulenceDataset
from turbpred.data_transformations import Transforms
from turbpred.loss import PredictionLoss
from turbpred.loss_history import LossHistory
from turbpred.trainer_diffusion import TrainerDiffusion, TesterDiffusion


if __name__ == "__main__":
    defaultDataRoot = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data")
    )
    dataRoot = os.environ.get("DATA_ROOT", defaultDataRoot)

    useGPU = torch.cuda.is_available()
    gpuID = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    if not useGPU:
        raise RuntimeError(
            "CUDA is not available. Do not run formal training on CPU/login node."
        )

    numWorkers = int(os.environ.get("NUM_WORKERS", "4"))
    testInterval = int(os.environ.get("TEST_INTERVAL", "2"))

    # Set this to 1 if you want to run testStep(0) before training.
    # For sanity checks, keep it as 0 because 60-step rollout can be slow.
    runInitialTest = os.environ.get("RUN_INITIAL_TEST", "0") == "1"

    # torch.manual_seed(1)
    # torch.cuda.manual_seed(1)

    # ACDM full-field physics-aware baseline
    modelName = "2D_Tra/128_acdm-r20_tra_continuity_10h"

    p_d = DataParams(
        batch=2,
        augmentations=["normalize"],
        sequenceLength=[3, 2],
        randSeqOffset=True,
        dataSize=[128, 64],
        dimension=2,
        simFields=["dens", "pres"],
        simParams=["mach"],
        normalizeMode="traMixed",
    )

    # Formal training can be increased later.
    # p_t = TrainingParams(epochs=3100, lr=0.0001)
    p_t = TrainingParams(epochs=int(os.environ.get("EPOCHS", "70")), lr=0.0001)

    p_l = LossParams()

    # Optional LSIM evaluation terms.
    # These require src/lsim/models/LSiM.pth to exist.
    p_l.recLSIM = 0.0
    p_l.predLSIM = 0.0

    # Physics-aware terms.
    # regDiv already exists in your LossParams.
    # regVort needs to be added to LossParams in src/turbpred/params.py.
    p_l.regDiv = 0.0
    p_l.regVort = 0.0

    p_me = None

    p_md = ModelParamsDecoder(
        arch="direct-ddpm+Prev",
        diffSteps=20,
        diffSchedule="linear",
        diffCondIntegration="noisy",
        trainingNoise=0.0,
    )

    p_ml = None
    pretrainPath = ""

    # Dataset


# Transonic dataset following the reference-paper split.
#
# sim indices if ordered by Mach:
#   0..40 = Ma 0.50..0.90
#
# Training:
#   Ma 0.53..0.63 -> sim 3..13
#   Ma 0.69..0.90 -> sim 19..40
#
# Test:
#   Traext  -> Ma 0.50..0.52 -> sim 0..2
#   Tralong -> Ma 0.64..0.65 -> sim 14..15
#   Traint  -> Ma 0.66..0.68 -> sim 16..18

trainSims = list(range(3, 14)) + list(range(19, 41))

trainSet = TurbulenceDataset(
    "Training Transonic",
    [dataRoot],
    filterTop=["128_tra"],
    filterSim=[trainSims],
    filterFrame=[(0, 1000)],
    sequenceLength=[p_d.sequenceLength],
    randSeqOffset=p_d.randSeqOffset,
    simFields=p_d.simFields,
    simParams=p_d.simParams,
    printLevel="sim",
)

testSets = {
    # Paper Traext:
    # Ma = 0.50, 0.51, 0.52
    # two 60-step windows with temporal stride 2
    "Traext_500_620": TurbulenceDataset(
        "Test Traext Ma 0.50-0.52 frames 500-620",
        [dataRoot],
        filterTop=["128_tra"],
        filterSim=[[0, 1, 2]],
        filterFrame=[(500, 620)],
        sequenceLength=[[60, 2]],
        simFields=p_d.simFields,
        simParams=p_d.simParams,
        printLevel="sim",
    ),

    "Traext_620_740": TurbulenceDataset(
        "Test Traext Ma 0.50-0.52 frames 620-740",
        [dataRoot],
        filterTop=["128_tra"],
        filterSim=[[0, 1, 2]],
        filterFrame=[(620, 740)],
        sequenceLength=[[60, 2]],
        simFields=p_d.simFields,
        simParams=p_d.simParams,
        printLevel="sim",
    ),

    # Paper Traint:
    # Ma = 0.66, 0.67, 0.68
    "Traint_500_620": TurbulenceDataset(
        "Test Traint Ma 0.66-0.68 frames 500-620",
        [dataRoot],
        filterTop=["128_tra"],
        filterSim=[[16, 17, 18]],
        filterFrame=[(500, 620)],
        sequenceLength=[[60, 2]],
        simFields=p_d.simFields,
        simParams=p_d.simParams,
        printLevel="sim",
    ),

    "Traint_620_740": TurbulenceDataset(
        "Test Traint Ma 0.66-0.68 frames 620-740",
        [dataRoot],
        filterTop=["128_tra"],
        filterSim=[[16, 17, 18]],
        filterFrame=[(620, 740)],
        sequenceLength=[[60, 2]],
        simFields=p_d.simFields,
        simParams=p_d.simParams,
        printLevel="sim",
    ),

    # Paper Tralong:
    # Ma = 0.64, 0.65
    # two 240-step windows with temporal stride 2
    "Tralong_0_480": TurbulenceDataset(
        "Test Tralong Ma 0.64-0.65 frames 0-480",
        [dataRoot],
        filterTop=["128_tra"],
        filterSim=[[14, 15]],
        filterFrame=[(0, 480)],
        sequenceLength=[[240, 2]],
        simFields=p_d.simFields,
        simParams=p_d.simParams,
        printLevel="sim",
    ),

    "Tralong_480_960": TurbulenceDataset(
        "Test Tralong Ma 0.64-0.65 frames 480-960",
        [dataRoot],
        filterTop=["128_tra"],
        filterSim=[[14, 15]],
        filterFrame=[(480, 960)],
        sequenceLength=[[240, 2]],
        simFields=p_d.simFields,
        simParams=p_d.simParams,
        printLevel="sim",
    ),
}

def train(
    modelName: str,
    trainSet: TurbulenceDataset,
    testSets: Dict[str, TurbulenceDataset],
    p_d: DataParams,
    p_t: TrainingParams,
    p_l: LossParams,
    p_me: ModelParamsEncoder,
    p_md: ModelParamsDecoder,
    p_ml: ModelParamsLatent,
    pretrainPath: str = "",
    useGPU: bool = True,
    gpuID: str = "0",
    numWorkers: int = 4,
    testInterval: int = 50,
    runInitialTest: bool = False,
):


    # Data and model setup
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuID

    logger = Logger(modelName, addNumber=True)

    model = PredictionModel(
        p_d,
        p_t,
        p_l,
        p_me,
        p_md,
        p_ml,
        pretrainPath,
        useGPU,
    )

    model.printModelInfo()

    # PredictionLoss is used by TesterDiffusion for evaluation metrics.
    criterion = PredictionLoss(p_l, p_d.dimension, p_d.simFields, useGPU)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=p_t.lr,
        weight_decay=p_t.weightDecay,
    )

    logger.setup(model, optimizer)


    # Training loader

    transTrain = Transforms(p_d)
    trainSet.transform = transTrain
    trainSet.printDatasetInfo()

    trainSampler = RandomSampler(trainSet)
    # trainSampler = SubsetRandomSampler(range(4))

    trainLoader = DataLoader(
        trainSet,
        sampler=trainSampler,
        batch_size=p_d.batch,
        drop_last=True,
        num_workers=numWorkers,
        pin_memory=useGPU,
    )

    trainHistory = LossHistory(
        "_train",
        "Training",
        logger.tfWriter,
        len(trainLoader),
        0,
        1,
        printInterval=1,
        logInterval=1,
        simFields=p_d.simFields,
    )

    # IMPORTANT:
    # This requires TrainerDiffusion.__init__ to accept p_l.
    # You still need to modify src/turbpred/trainer_diffusion.py accordingly.
    trainer = TrainerDiffusion(
        model,
        trainLoader,
        optimizer,
        trainHistory,
        logger.tfWriter,
        p_t,
        p_l,
    )


    # Test loaders
    testers = []
    testHistories = []

    for shortName, testSet in testSets.items():
        p_d_test = copy.deepcopy(p_d)
        p_d_test.augmentations = ["normalize"]
        p_d_test.sequenceLength = testSet.sequenceLength
        p_d_test.randSeqOffset = False
        p_d_test.batch = 1

        # If memory becomes an issue for 60-step rollout, reduce test batch:
        # p_d_test.batch = 1

        transTest = Transforms(p_d_test)
        testSet.transform = transTest
        testSet.printDatasetInfo()

        testSampler = SequentialSampler(testSet)
        # testSampler = SubsetRandomSampler(range(2))

        testLoader = DataLoader(
            testSet,
            sampler=testSampler,
            batch_size=p_d_test.batch,
            drop_last=False,
            num_workers=numWorkers,
            pin_memory=useGPU,
        )

        testHistory = LossHistory(
            shortName,
            testSet.name,
            logger.tfWriter,
            len(testLoader),
            1,
            1,
            printInterval=0,
            logInterval=0,
            simFields=p_d.simFields,
        )

        tester = TesterDiffusion(
            model,
            testLoader,
            criterion,
            testHistory,
            p_t,
        )

        testers.append(tester)
        testHistories.append(testHistory)


    # Training

    print("Starting Training")
    logger.saveTrainState(0)

    if runInitialTest:
        print("Running initial test at epoch 0")
        for tester in testers:
            tester.testStep(0)
    else:
        print("Skipping initial test at epoch 0")

    for epoch in range(0, p_t.epochs):
        trainer.trainingStep(epoch)

        logger.saveTrainState(epoch)

        doTest = ((epoch + 1) % testInterval == 0) or (epoch == p_t.epochs - 1)

        if doTest:
            for tester in testers:
                tester.testStep(epoch + 1)

        trainHistory.updateAccuracy(
            [p_d, p_t, p_l, p_me, p_md, p_ml],
            testHistories,
            epoch == p_t.epochs - 1,
        )

    logger.close()

    print("Finished Training")


if __name__ == "__main__":
    train(
        modelName,
        trainSet,
        testSets,
        p_d,
        p_t,
        p_l,
        p_me,
        p_md,
        p_ml,
        pretrainPath=pretrainPath,
        useGPU=useGPU,
        gpuID=gpuID,
        numWorkers=numWorkers,
        testInterval=testInterval,
        runInitialTest=runInitialTest,
    )  # type: ignore