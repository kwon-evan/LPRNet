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


class _STNet(nn.Module):
    def __init__(self):
        super(_STNet, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.Mish(True),
            nn.Conv2d(32, 32, kernel_size=5),
            nn.MaxPool2d(3, stride=3),
            nn.Mish(True),
        )
        # Regressor for the 3x2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 15 * 6, 32), nn.Mish(True), nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 15 * 6)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x


class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.Mish(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.Mish(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.Mish(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class _LPRNet(nn.Module):
    def __init__(self, class_num, dropout_rate):
        super(_LPRNet, self).__init__()
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.Mish(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            small_basic_block(ch_in=64, ch_out=128),
            nn.BatchNorm2d(num_features=128),
            nn.Mish(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            small_basic_block(ch_in=64, ch_out=256),
            nn.BatchNorm2d(num_features=256),
            nn.Mish(),
            small_basic_block(ch_in=256, ch_out=256),
            nn.BatchNorm2d(num_features=256),
            nn.Mish(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 2, 2)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(2, 4), stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(
                in_channels=256, out_channels=class_num, kernel_size=(12, 2), stride=1
            ),
            nn.BatchNorm2d(num_features=class_num),
            nn.Mish(),
        )
        self.container = nn.Sequential(
            nn.Conv2d(
                in_channels=256 + class_num + 128 + 64,
                out_channels=self.class_num,
                kernel_size=(1, 1),
                stride=(1, 1),
            ),
        )

    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]:  # [2, 4, 8, 11, 22]
                keep_features.append(x)

        global_context = list()
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(5, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            global_context.append(f)

        x = torch.cat(global_context, 1)
        x = self.container(x)
        logits = torch.mean(x, dim=2)

        return logits


class LPRNet(L.LightningModule):
    def __init__(self, args: Optional[Namespace] = None):
        super().__init__()
        self.save_hyperparameters(args)
        self.STNet = _STNet()
        self.LPRNet = _LPRNet(
            class_num=len(self.hparams.chars), dropout_rate=self.hparams.dropout_rate
        )

    def forward(self, x):
        return self.LPRNet(self.STNet(x))

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
                    "params": self.STNet.parameters(),
                    "weight_decay": self.hparams.weight_decay,
                },
                {"params": self.LPRNet.parameters()},
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
