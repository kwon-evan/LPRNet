#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from argparse import Namespace
import warnings
import yaml
import torch
import cv2
from rich import print
from imutils import paths
from rich.progress import track
from sklearn.metrics import accuracy_score

from lprnet import LPRNet, numpy2tensor, decode

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    with open('config/idn_config.yaml') as f:
        args = Namespace(**yaml.load(f, Loader=yaml.FullLoader))

    load_model_start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lprnet = LPRNet(args).to(device).eval()
    lprnet.load_state_dict(torch.load(args.pretrained)['state_dict'])
    print(f"Successful to build network in {time.time() - load_model_start}sec")

    imgs = [el for el in paths.list_images(args.test_dir)]
    labels = [os.path.basename(n).split('.')[0].split('-')[0].split('_')[0] for n in track(imgs, description="Making labels... ")]

    times = []
    preds = []
    for i, img in track(enumerate(imgs), description="Inferencing... "):
        im = numpy2tensor(cv2.imread(img), args.img_size).unsqueeze(0).to(device)

        t0 = time.time()
        logit = lprnet(im).detach().to('cpu')
        pred, _ = decode(logit, args.chars)
        t1 = time.time()
        print("Predicted: {} \t Label: {}".format(pred, labels[i]))
        times.append(t1 - t0)
        preds.append(pred)
        
    print("Accuracy: ", accuracy_score(labels, preds))
    print("Avg Time: ", sum(times) / len(times), "sec")
