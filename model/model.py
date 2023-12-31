import re
from typing import Any

from cv2 import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from data.load_data import LPRDataLoader, collate_fn
from model.stn import STNet
from model.lprn import LPRNet, CHARS

T_length = 18


def decode(preds, CHARS):
    # greedy decode
    pred_labels = list()
    labels = list()
    for i in range(preds.shape[0]):
        pred = preds[i, :, :]
        pred_label = list()
        for j in range(pred.shape[1]):
            pred_label.append(np.argmax(pred[:, j], axis=0))
        no_repeat_blank_label = list()
        pre_c = ''
        # pre_c = pred_label[0]
        for c in pred_label:  # dropout repeate label and blank label
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        pred_labels.append(no_repeat_blank_label)

    for i, label in enumerate(pred_labels):
        lb = ""
        for i in label:
            lb += CHARS[i]
        labels.append(lb)

    return labels, pred_labels


def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)

    return tuple(input_lengths), tuple(target_lengths)


def accuracy(logits, labels, lengths):
    preds = logits.cpu().detach().numpy()  # (batch size, 68, 18)
    _, pred_labels = decode(preds, CHARS)  # list of predict output

    TP, total = 0, 0
    start = 0
    for i, length in enumerate(lengths):
        label = labels[start:start + length]
        start += length
        if np.array_equal(np.array(pred_labels[i]), label.cpu().numpy()):
            TP += 1
        total += 1

    return TP / total


class Model(pl.LightningModule):
    def __init__(
            self,
            img_size: tuple = (100, 50),
            dropout_rate: float = 0.5,
            weight_decay: float = 2e-5,
            batch_size: int = 512,
            lr: float = 0.001,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.STN = STNet()
        self.LPRNet = LPRNet(class_num=len(CHARS), dropout_rate=dropout_rate)
        self.save_hyperparameters()

    def forward(self, x):
        return self.LPRNet(self.STN(x))

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        imgs, labels, lengths = batch

        logits = self(imgs)
        log_probs = logits.permute(2, 0, 1)
        log_probs = log_probs.log_softmax(2).requires_grad_()
        input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)
        loss = F.ctc_loss(log_probs=log_probs, targets=labels,
                          input_lengths=input_lengths, target_lengths=target_lengths,
                          blank=len(CHARS) - 1, reduction='mean')
        acc = accuracy(logits, labels, lengths)

        self.log("train-loss", loss, prog_bar=True, logger=True, sync_dist=True, on_epoch=True)
        self.log("train-acc", acc, prog_bar=True, logger=True, sync_dist=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels, lengths = batch

        logits = self(imgs)
        log_probs = logits.permute(2, 0, 1)
        log_probs = log_probs.log_softmax(2).requires_grad_()
        input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)
        loss = F.ctc_loss(log_probs=log_probs, targets=labels,
                          input_lengths=input_lengths, target_lengths=target_lengths,
                          blank=len(CHARS) - 1, reduction='mean')
        acc = accuracy(logits, labels, lengths)

        self.log("val-loss", loss, prog_bar=True, logger=True, sync_dist=True, on_epoch=True)
        self.log("val-acc", acc, prog_bar=True, logger=True, sync_dist=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        imgs, labels, lengths = batch

        logits = self(imgs)
        log_probs = logits.permute(2, 0, 1)
        log_probs = log_probs.log_softmax(2).requires_grad_()
        input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)
        loss = F.ctc_loss(log_probs=log_probs, targets=labels,
                          input_lengths=input_lengths, target_lengths=target_lengths,
                          blank=len(CHARS) - 1, reduction='mean')
        acc = accuracy(logits, labels, lengths)

        self.log("test-loss", loss, prog_bar=True, logger=True, sync_dist=True, on_epoch=True)
        self.log("test-acc", acc, prog_bar=True, logger=True, sync_dist=True, on_epoch=True)

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0) -> Any:
        imgs, labels, lengths = batch

        logits = self(imgs)
        preds = logits.cpu().detach().numpy()  # (batch size, 68, 18)
        predict, _ = decode(preds, CHARS)  # list of predict output

        return predict

    def detect(self, image, device):
        size = (100, 50)
        if all(image.shape):
            # image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
            base_pic = np.zeros((size[1],size[0],3),np.uint8)
            pic1 = image
            h, w = pic1.shape[:2]
            ash = size[1] / h
            asw = size[0] / w

            if asw < ash:
                sizeas = (int(w * asw), int(h * asw))
            else:
                sizeas = (int(w * ash), int(h * ash))

            pic1 = cv2.resize(pic1, dsize=sizeas)
            base_pic[int(size[1] / 2 - sizeas[1] / 2):int(size[1] / 2 + sizeas[1] / 2),
                    int(size[0] / 2 - sizeas[0] / 2):int(size[0] / 2 + sizeas[0] / 2),:] = pic1
            image = base_pic

            image = (np.transpose(np.float32(image), (2, 0, 1)) - 127.5) * 0.0078125
            data = torch.from_numpy(image).float().unsqueeze(0).to(device)
            logits = self(data)
            preds = logits.cpu().detach().numpy()  # (batch size, 68, 18)
            predict, _ = decode(preds, CHARS)  # list of predict output
            return predict[0]
        else:
            return ""

    def check(self, label):
        kor_plate_pattern = re.compile('[가-힣]{0,5}\d{0,3}[가-힣]\d{4}')
        plate_name = kor_plate_pattern.findall(label)
        return True if plate_name else False

    def image2data(self, image):
        return (np.transpose(
            np.float32(
                cv2.resize(image, (100, 50), interpolation=cv2.INTER_CUBIC)
            ), (2, 0, 1)
        ) - 127.5) * 0.0078125

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{'params': self.STN.parameters(), 'weight_decay': self.hparams.weight_decay},
                                      {'params': self.LPRNet.parameters()}], lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 2, 0.0001, -1)
        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "monitor": "val-loss",
                    "strict": True,
                    "name": "lr"
                }}

    def train_dataloader(self):
        return DataLoader(LPRDataLoader([self.hparams.img_dirs_train], imgSize=(100, 50)),
                          shuffle=False, num_workers=4, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(LPRDataLoader([self.hparams.img_dirs_val], imgSize=(100, 50)),
                          shuffle=False, num_workers=4, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(LPRDataLoader([self.hparams.img_dirs_test], imgSize=(100, 50)),
                          shuffle=False, num_workers=4, collate_fn=collate_fn)
