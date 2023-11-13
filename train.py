import os
from argparse import Namespace
from datetime import datetime
import warnings
import yaml

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import WandbLogger

from lprnet import LPRNet
from lprnet import DataModule

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    with open("config/idn_config.yaml") as f:
        args = Namespace(**yaml.load(f, Loader=yaml.FullLoader))

    args.saving_ckpt += datetime.now().strftime("_%m-%d_%H:%M")

    if not os.path.exists(args.saving_ckpt):
        os.mkdir(args.saving_ckpt)

    lprn = LPRNet(args)
    print(lprn.hparams)
    print("Model loaded")

    # Set Data Modulews
    data_module = DataModule(args)

    # Set Trainer
    trainer = L.Trainer(
        callbacks=[
            ModelCheckpoint(
                dirpath=args.saving_ckpt,
                monitor="val-acc",
                mode="max",
                filename="{epoch:02d}-{val-acc:.3f}",
                verbose=True,
                save_last=True,
                save_top_k=5,
            ),
            EarlyStopping(
                monitor="val-acc",
                mode="max",
                min_delta=0.00,
                patience=100,
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        precision=16,
        accelerator="auto",
        # amp_backend="apex",
        devices=1,
        logger=WandbLogger(project="LPRNet-IDN"),
    )

    # Train
    print("training kicked off..")
    print("-" * 10)
    trainer.fit(model=lprn, datamodule=data_module)
