import numpy as np
from cv2 import resize, INTER_LANCZOS4
from typing import Optional
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from lprnet.utils import decode, accuracy


def sparse_tuple_for_ctc(t_length, lengths):
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(t_length)
        target_lengths.append(ch)

    return torch.tensor(input_lengths), torch.tensor(target_lengths)


class res_block(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1, ks=3, downsample=None, padding=1):
        super(res_block, self).__init__()
        self.downsample = downsample
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=ks,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=ch_out),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=ch_out,
                out_channels=ch_out,
                kernel_size=ks,
                stride=1,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=ch_out),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        out = self.act(out)
        return out


class downsample(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=0):
        super(downsample, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        )

    def forward(self, x):
        out = self.block(x)
        return out


class _LPRNet(nn.Module):
    def __init__(self, lpr_max_len, phase, class_num, dropout_rate, device, drop=False):
        super(_LPRNet, self).__init__()
        self.phase = phase
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num
        self.device = device

        self.stage1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            res_block(ch_in=64, ch_out=64, padding=1),
            res_block(
                ch_in=64,
                ch_out=128,
                padding=1,
                downsample=downsample(64, 128, kernel_size=1, stride=1),
            ),
            # s2
            res_block(
                ch_in=128,
                ch_out=128,
                stride=2,
                padding=1,
                downsample=downsample(128, 128, kernel_size=1, stride=2),
            ),
            res_block(
                ch_in=128,
                ch_out=256,
                padding=1,
                downsample=downsample(128, 256, kernel_size=1, stride=1),
            ),
        )  # (38 x 150)

        self.downsample1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=256),
        )

        self.stage2 = nn.Sequential(
            res_block(
                ch_in=256,
                ch_out=256,
                stride=2,
                padding=1,
                downsample=downsample(256, 256, kernel_size=1, stride=2),
            ),
            res_block(ch_in=256, ch_out=256, padding=1),
        )  # (19 x 75)

        self.downsample2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=256),
        )

        self.stage3 = nn.Sequential(
            res_block(
                ch_in=256,
                ch_out=256,
                stride=2,
                padding=1,
                downsample=downsample(256, 256, kernel_size=1, stride=2),
            ),
            res_block(
                ch_in=256,
                ch_out=256,
                stride=2,
                padding=1,
                downsample=downsample(256, 256, kernel_size=1, stride=2),
            ),
        )  # (5 x 19)
        self.stage4 = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(1, 5),
                stride=1,
                padding=(0, 2),
            ),  # (6 x 24)
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(
                in_channels=256,
                out_channels=class_num,
                kernel_size=(5, 1),
                stride=1,
                padding=(2, 0),
            ),  # (6 x 24)
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),
        )

        self.bn = nn.Sequential(nn.BatchNorm2d(num_features=256))
        self.bn4 = nn.Sequential(nn.BatchNorm2d(num_features=self.class_num))

        self.container = nn.Sequential(
            nn.Conv2d(
                in_channels=768 + self.class_num,
                out_channels=self.class_num,
                kernel_size=(1, 1),
                stride=(1, 1),
            ),
        )

    def forward(self, x):
        stage1 = self.stage1(x)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)

        skip1 = self.downsample1(stage1)
        skip2 = self.downsample2(stage2)
        skip3 = stage3
        skip4 = stage4

        x = torch.cat([skip1, skip2, skip3, skip4], 1)
        x = self.container(x)
        logits = torch.mean(x, dim=2)

        return logits


class LPRNet(L.LightningModule):
    def __init__(self, args: Optional[Namespace] = None):
        super().__init__()
        self.save_hyperparameters(args)
        print(args)
        print(self.hparams)
        self.LPRNet = _LPRNet(
            lpr_max_len=8,
            phase=False,
            class_num=len(self.hparams.chars),
            dropout_rate=self.hparams.dropout_rate,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    def forward(self, x):
        return self.LPRNet(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        imgs, labels, lengths = batch

        logits = self(imgs)
        log_probs = logits.permute(2, 0, 1)
        log_probs = log_probs.log_softmax(2).requires_grad_()
        input_lengths, target_lengths = sparse_tuple_for_ctc(
            self.hparams.t_length, lengths
        )
        loss = F.ctc_loss(
            log_probs=log_probs,
            targets=labels,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank=len(self.hparams.chars) - 1,
            reduction="mean",
        )
        acc = accuracy(logits, labels, lengths, self.hparams.chars)

        self.log("train-loss", abs(loss), prog_bar=True, logger=True, sync_dist=True)
        self.log("train-acc", acc, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels, lengths = batch

        logits = self(imgs)
        log_probs = logits.permute(2, 0, 1)
        log_probs = log_probs.log_softmax(2).requires_grad_()
        input_lengths, target_lengths = sparse_tuple_for_ctc(
            self.hparams.t_length, lengths
        )
        print(log_probs.shape)
        print(labels.shape)
        print(input_lengths.shape)
        print(target_lengths.shape)
        loss = F.ctc_loss(
            log_probs=log_probs,
            targets=labels,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank=len(self.hparams.chars) - 1,
            reduction="mean",
        )
        acc = accuracy(logits, labels, lengths, self.hparams.chars)

        self.log("val-loss", abs(loss), prog_bar=True, logger=True, sync_dist=True)
        self.log("val-acc", acc, prog_bar=True, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        imgs, labels, lengths = batch
        import time

        start = time.time()
        logits = self(imgs)
        log_probs = logits.permute(2, 0, 1)
        log_probs = log_probs.log_softmax(2).requires_grad_()
        input_lengths, target_lengths = sparse_tuple_for_ctc(
            self.hparams.t_length, lengths
        )
        loss = F.ctc_loss(
            log_probs=log_probs,
            targets=labels,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank=len(self.hparams.chars) - 1,
            reduction="mean",
        )
        acc = accuracy(logits, labels, lengths, self.hparams.chars)
        end = time.time()

        self.log("test-loss", abs(loss), prog_bar=True, logger=True, sync_dist=True)
        self.log("test-acc", acc, prog_bar=True, logger=True, sync_dist=True)
        self.log("test-time", end - start, prog_bar=True, logger=True, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        imgs, labels, lengths = batch

        logits = self(imgs)
        preds = logits.cpu().detach().numpy()  # (batch size, 68, 18)
        predict, _ = decode(preds, self.chars)  # list of predict output

        return predict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {
                    "params": self.LPRNet.parameters(),
                    "weight_decay": self.hparams.weight_decay,
                },
            ],
            lr=self.hparams.lr,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 10, 2, 0.0001, -1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "val-loss",
                "strict": True,
                "name": "lr",
            },
        }
