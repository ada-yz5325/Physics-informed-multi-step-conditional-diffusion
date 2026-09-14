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



# Multi-step P=2 10-hour baseline-matched training run

# Goal:
#   Baseline P=1:
#       input  = previous states
#       output = one future state
#
#   Multi-step P=2:
#       input  = previous states
#       output = two consecutive future states
#
# IMPORTANT:
#   This training script only defines the intended multi-step
#   configuration. To make it actually work, model.py and
#   trainer_diffusion.py must also support:
#
#       p_d.historySteps = 2
#       p_d.futureSteps  = 2
#
#   and target should be:
#
#       [s(t+1), s(t+2)]
#
#   concatenated along channel dimension before diffusion.



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

    # Keep this as 0 for smoke tests.
    runInitialTest = os.environ.get("RUN_INITIAL_TEST", "0") == "1"
    maxTestSamples = int(os.environ.get("MAX_TEST_SAMPLES", "5"))

    # torch.manual_seed(1)
    # torch.cuda.manual_seed(1)

  
    modelName = "2D_Inc/128_acdm-r20_puv_vor001_multistep_p2_hw_10h"

  
    # Multi-step setting
    historySteps = 2
    futureSteps = 2

    # sequenceLength first value should cover:
    #   historySteps + futureSteps = 2 + 2 = 4 frames
    #
    # The second value keeps the same temporal spacing as your baseline.
    p_d = DataParams(
        batch=int(os.environ.get("BATCH_SIZE", "2")),
        augmentations=["normalize"],
        sequenceLength=[historySteps + futureSteps, 2],
        randSeqOffset=True,
        dataSize=[128, 64],
        dimension=2,
        simFields=["pres"],
        simParams=["rey"],
        normalizeMode="incMixed",
    )

    # Add explicit multi-step attributes.
    # DataParams does not originally define these, but Python allows
    # dynamic attributes. model.py / trainer_diffusion.py can read them.
    p_d.historySteps = historySteps
    p_d.futureSteps = futureSteps
    p_d.multiStep = True


    # Training parameters

    # 10h baseline-matched run: same training simulations, frame range,
    # batch size, learning rate, diffusion steps, and epoch count as
    # the 10h single-step baseline. The main difference is P=2 output.
    p_t = TrainingParams(epochs=int(os.environ.get("EPOCHS", "70")), lr=0.0001)

    p_l = LossParams()

    # Keep LSIM disabled for this first medium multi-step run.
    # This keeps the run focused on MSE/physics losses and avoids extra overhead.
    p_l.recLSIM = 0.0
    p_l.predLSIM = 0.0

    # Physics-aware terms.
    # These should be applied to both predicted future frames.
    p_l.regDiv = 0.0
    p_l.regVort = 0.01

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

    # Training range matched to the 10h single-step baseline.
    trainSet = TurbulenceDataset(
        "Training",
        [dataRoot],
        filterTop=["128_inc"],
        filterSim=[(10, 50)],
        filterFrame=[(300, 1000)],
        sequenceLength=[p_d.sequenceLength],
        randSeqOffset=p_d.randSeqOffset,
        simFields=p_d.simFields,
        simParams=p_d.simParams,
        printLevel="sim",
    )

    # Baseline-matched short validation rollout during training.
    testSets = {
        "lowRey": TurbulenceDataset(
            "Test Low Reynolds 100-200",
            [dataRoot],
            filterTop=["128_inc"],
            filterSim=[[82, 84, 86, 88, 90]],
            filterFrame=[(1000, 1040)],
            sequenceLength=[[20, 2]],
            simFields=p_d.simFields,
            simParams=p_d.simParams,
            printLevel="sim",
        ),
        "highRey": TurbulenceDataset(
            "Test High Reynolds 900-1000",
            [dataRoot],
            filterTop=["128_inc"],
            filterSim=[[0, 2, 4, 6, 8]],
            filterFrame=[(1000, 1040)],
            sequenceLength=[[20, 2]],
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
    maxTestSamples: int = 4,
):

    # Data and model setup
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuID

    print("Runtime configuration")
    print("---------------------")
    print(f"modelName:     {modelName}")
    print(f"useGPU:        {useGPU}")
    print(f"gpuID:         {gpuID}")
    print(f"numWorkers:    {numWorkers}")
    print(f"testInterval:  {testInterval}")
    print(f"maxTestSamples:{maxTestSamples}")
    print(f"sequenceLength:{p_d.sequenceLength}")
    print(f"historySteps:  {getattr(p_d, 'historySteps', 'not set')}")
    print(f"futureSteps:   {getattr(p_d, 'futureSteps', 'not set')}")
    print(f"multiStep:     {getattr(p_d, 'multiStep', False)}")
    print("---------------------")

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
    # It also needs to support multi-step clean_pred / clean_true
    # shapes when p_d.futureSteps = 2.
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

        # Keep explicit multi-step attributes for test config.
        p_d_test.historySteps = getattr(p_d, "historySteps", 2)
        p_d_test.futureSteps = getattr(p_d, "futureSteps", 2)
        p_d_test.multiStep = getattr(p_d, "multiStep", True)

        transTest = Transforms(p_d_test)
        testSet.transform = transTest
        testSet.printDatasetInfo()

        if maxTestSamples > 0:
            testCount = min(maxTestSamples, len(testSet))
            testSampler = SubsetRandomSampler(range(testCount))
        else:
            testSampler = SequentialSampler(testSet)

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
        maxTestSamples=maxTestSamples,
    )  # type: ignore