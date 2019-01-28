import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from glob import glob
import os
import numpy as np
import PIL
from zipfile import BadZipFile

class CityScapeLoader(Dataset):
    def __init__(self, dataroot, partition, transforms=torchvision.transforms.ToTensor(), RandomHorizontalFlip=0.5, RandomVerticalFlip=0.0, mres=True, transpose=False):
        assert(partition in ['test', 'train', 'val'])
        self.dataroot = dataroot
        self.partition = partition
        self.filelist = sorted(glob(os.path.join(dataroot, partition, "*.npz")))
        self.RandomHorizontalFlip = RandomHorizontalFlip
        self.RandomVerticalFlip = RandomVerticalFlip
        self.transforms = transforms
        self.mres = mres
        self.transpose = transpose

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        if self.mres:
            targets = ['target', 'target_2', 'target_4', 'target_8']
        else:
            targets = ['target']
        dat = np.load(self.filelist[idx])
        target_ = [dat[t].T for t in targets] if self.transpose else [dat[t] for t in targets]
        input = PIL.Image.fromarray(np.moveaxis(dat['image'], 0, -1))       
        label = dat['label_id'].astype(np.int)

        # flip horizontal
        if np.random.rand() < self.RandomHorizontalFlip:
            input = F.hflip(input)
            target_ = [t[:, ::-1] for t in target_]
        # flip vertical
        if np.random.rand() < self.RandomVerticalFlip:
            input = F.vflip(input)
            target_ = [t[::-1, :] for t in target_]

        # final transform of input
        input = self.transforms(input)
        target = [torch.from_numpy(t.copy()).double() for t in target_]
        label = torch.from_numpy(label.copy()).long()

        return input, target, label
