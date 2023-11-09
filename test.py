#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from argparse import Namespace
import warnings
import yaml
import torch
from pytorch_lightning import Trainer

from lprnet import LPRNet, DataModule

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    with open('config/idn_config.yaml') as f:
        args = Namespace(**yaml.load(f, Loader=yaml.FullLoader))

    load_model_start = time.time()
    lprnet = LPRNet(args)
    # lprnet.load_state_dict(torch.load(args.pretrained))
    lprnet.load_from_checkpoint(args.pretrained)
    lprnet.eval()
    print(f"Successful to build network in {time.time() - load_model_start}s")

    dm = DataModule(args)

    trainer = Trainer(
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
