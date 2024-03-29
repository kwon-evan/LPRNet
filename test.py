#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from argparse import Namespace
import warnings
import yaml
import torch

import lightning as L

from lprnet import LPRNet, DataModule

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    with open("config/idn_config.yaml") as f:
        args = Namespace(**yaml.load(f, Loader=yaml.FullLoader))

    load_model_start = time.time()

    if args.pretrained:
        lprnet = LPRNet.load_from_checkpoint(args.pretrained)
        print("Loaded checkpoint from: ", args.pretrained)
    else:
        lprnet = LPRNet(args)
        print("Created new network")

    lprnet.eval()
    print(f"Successful to build network in {time.time() - load_model_start}s")

    dm = DataModule(args)

    trainer = L.Trainer(
        accelerator="auto",
        precision=16,
        devices=torch.cuda.device_count(),
    )

    since = time.time()
    predictions = trainer.test(lprnet, dm)

    img_cnt = len(os.listdir(args.test_dir))
    time_total = time.time() - since

    print("model inference in {:2.3f} seconds".format(time_total))
    print(f"img/ms: {time_total/img_cnt * 1000}")
