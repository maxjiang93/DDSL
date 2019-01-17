from glob import glob
import os
from tqdm import tqdm
import numpy as np
from preprocess import process_one
import torch

def main():
	n = 0
	files = sorted(glob("mres_processed_data/*/*.npz"))
	for f in tqdm(files):
		targets = ['target', 'target_2', 'target_4', 'target_8']
		try:
			dat = np.load(f)
			target_ = [dat[t] for t in targets]
			img = dat['image']
			label = dat['label_id'].astype(np.int)
		except:
			print("[!] Failed to open {}".format(f))
			n += 1
			# catch corrupt data
			outfile = f
			infile = os.path.join("/home/maxjiang/Codes/dsnet/data/original_data/", outfile.split('/')[-2], outfile.split('/')[-1])
			process_one(infile, outfile, res=224, mres=True, dev=torch.device('cpu'))
			dat = np.load(f)
			target_ = [dat[t] for t in targets]
			img = dat['image']
			label = dat['label_id'].astype(np.int)
			print("[V] Recovered {}".format(f))
	print("Total files corrupted: {}".format(n))

if __name__ == '__main__':
	main()