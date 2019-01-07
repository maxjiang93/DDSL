import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from glob import glob
import os
import numpy as np
import PIL
from preprocess import process_one
from zipfile import BadZipFile

class CityScapeLoader(Dataset):
	def __init__(self, dataroot, partition, transforms=torchvision.transforms.ToTensor(), RandomHorizontalFlip=0.5, RandomVerticalFlip=0.0):
		assert(partition in ['test', 'train', 'val'])
		self.dataroot = dataroot
		self.partition = partition
		self.filelist = sorted(glob(os.path.join(dataroot, partition, "*.npz")))
		self.RandomHorizontalFlip = RandomHorizontalFlip
		self.RandomVerticalFlip = RandomVerticalFlip
		self.transforms = transforms

	def __len__(self):
		return len(self.filelist)

	def __getitem__(self, idx):
		# CATCH CORRUPT DATA
		try:
			dat = np.load(self.filelist[idx])
		except BadZipFile:
			outfile = self.filelist[idx]
			infile = outfile.replace('')

		target = dat['target']
		input = PIL.Image.fromarray(np.moveaxis(dat['image'], 0, -1))			
		# flip horizontal
		if np.random.rand() < self.RandomHorizontalFlip:
			input = F.hflip(input)
			target = target[:, ::-1]
		# flip vertical
		if np.random.rand() < self.RandomVerticalFlip:
			input = F.vflip(input)
			target = target[::-1, :]
		# final transform of input
		input = self.transforms(input)
		target = torch.from_numpy(target.copy()).double()

		return input, target
