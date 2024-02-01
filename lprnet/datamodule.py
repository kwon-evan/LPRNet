import os
import re
import random
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from imutils import paths
import lightning as L

from lprnet.utils import encode


def resize_pad(img, size):
    base_pic = np.zeros((size[1], size[0], 3), np.uint8)
    pic1 = img
    h, w = pic1.shape[:2]
    ash = size[1] / h
    asw = size[0] / w

    if asw < ash:
        sizeas = (int(w * asw), int(h * asw))
    else:
        sizeas = (int(w * ash), int(h * ash))

    pic1 = cv2.resize(pic1, dsize=sizeas)
    base_pic[
        int(size[1] / 2 - sizeas[1] / 2) : int(size[1] / 2 + sizeas[1] / 2),
        int(size[0] / 2 - sizeas[0] / 2) : int(size[0] / 2 + sizeas[0] / 2),
        :,
    ] = pic1

    return base_pic


def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.float32)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)


class LPRNetDataset(Dataset):
    def __init__(self, args, stage, PreprocFun=None):
        self.args = args
        self.stage = stage
        self.img_paths = []
        self.img_size = self.args.img_size

        if stage == "train":
            self.img_dir = self.args.train_dir
        elif stage == "valid":
            self.img_dir = self.args.valid_dir
        elif stage == "test":
            self.img_dir = self.args.test_dir
        elif stage == "predict":
            self.img_dir = self.args.test_dir
        else:
            assert f"No Such Stage. Your input -> {self.stage}"

        self.img_paths = [img_path for img_path in paths.list_images(self.img_dir)]

        if stage == "train":
            random.shuffle(self.img_paths)

        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        Image = cv2.imread(filename)
        height, width, _ = Image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            Image = cv2.resize(Image, self.img_size, interpolation=cv2.INTER_CUBIC)
        Image = self.PreprocFun(Image)

        basename = os.path.basename(filename)
        imgname, suffix = os.path.splitext(basename)
        imgname = imgname.split("-")[0].split("_")[0]
        imgname = imgname.upper()
        label = encode(imgname, self.args.chars)

        if label:
            if not self.check(label):
                assert 0, f"{imgname} <- Error label ^~^!!!"

        return Image, label, len(label)

    def transform(self, img):
        img = img.astype("float32")
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))

        return img

    def check(self, label):
        # kor_plate_pattern = re.compile('[가-힣]{0,5}[0-9]{0,3}[가-힣][0-9]{4}')
        idn_plate_pattern = re.compile("[A-Z]{0,3}[0-9]{0,4}[A-Z]{0,3}")
        plate_name = idn_plate_pattern.findall(
            "".join([self.args.chars[c] for c in label])
        )

        return True if plate_name else False


class DataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        print("dm loaded")
        # print(self.args)

    def setup(self, stage: str):
        if stage == "fit":
            self.train = LPRNetDataset(self.args, "train")
            print("train: ", len(self.train))
            self.val = LPRNetDataset(self.args, "valid")
            print("val: ", len(self.val))

        if stage == "test":
            self.test = LPRNetDataset(self.args, "test")

        if stage == "predict":
            self.predict = LPRNetDataset(self.args, "predict")

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )
