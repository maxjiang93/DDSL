import torch
from time import time
import numpy as np

# dim = (128, 128, 64)
dim = (64, 64, 32)
d = len(dim)
batch = 100
X = torch.rand(batch, d+1, d).cuda()
K = torch.rand(d, *dim, dtype=torch.float32).cuda()
F = torch.zeros(*dim, 2, dtype=torch.float32).cuda()

t = []
for i in range(int(7000/batch)):
	t0 = time()
	sig = torch.einsum('bjd,d...->bj...', (X, K))
	sig_sub = [[None] * (d+1)] * (d+1)
	esig = torch.stack((torch.cos(sig), torch.sin(sig)), dim=-1)
	for c0 in range(d+1):
		for c1 in range(c0, d+1):
			sig_sub[c0][c1] = sig[:, c0] - sig[:, c1]

	t.append(time()-t0)


print("EXP Time lapse: {} sec".format(np.sum(t)))
