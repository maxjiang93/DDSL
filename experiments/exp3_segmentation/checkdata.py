from glob import glob
import os
from tqdm import tqdm
import numpy as np

def main():
	n = 0
	files = sorted(glob("processed_data/*/*.npz"))
	for f in tqdm(files):
		try:
			dat = np.load(f)
			img = dat['image']
			target = dat['target']
		except:
			print("[!] Failed to open {}".format(f))
			n += 1
	print("Total files corrupted: {}".format(n))

if __name__ == '__main__':
	main()