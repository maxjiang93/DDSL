import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import importlib.machinery
import types
import os
import pickle
from shapely import wkt
import json
import numpy as np
import sys; sys.path.append("../../methods")

from transform import simplex_ft_gpu, simplex_ft_bw_gpu
from loader import poly2ve

MEAN = 0.14807655
STD = 0.36801067

class PolyMNIST(object):
    def __init__(self, path):
        with open(os.path.join(path, "mnist_polygon_test.json"), 'r') as infile:
            self.plist = json.load(infile)
        with open(os.path.join(path, "mnist_label_test.json"), 'r') as infile:
            self.label = json.load(infile)
    def __getitem__(self, idx):
        P = wkt.loads(self.plist[idx])
        V, E = poly2ve(P)
        return V, E, self.label[idx]


class ShapeOptimizer(object):
    def __init__(self, V, E, model, target_cls, device):
        self.V0 = self.V = V
        self.E0 = self.E = E
        self.D = np.ones((E.shape[0], 1))
        self.model = model
        self.target_tensor = torch.tensor([target_cls]).to(device)
        self.dV = None
        self.res = self.model.module.signal_sizes[0]

    def _step(self, step_size):
        pass

    def _get_grad(self):
        # compute frequencies from V, E
        Freq = simplex_ft_gpu(self.V, self.E, self.D, (self.res+2, self.res+2), t=(1,1), j=2)
        Freq = np.squeeze(Freq)
        half = int(self.res/2+1)
        Freq = np.concatenate((Freq[:half], Freq[half+2:]), axis=0)
        Freq = np.stack([np.real(Freq), np.imag(Freq)], axis=-1).astype(np.float32)
        Freq = np.expand_dims(np.expand_dims(Freq, 0), 0) # pad to shape (batch(1), channel(1), res, res)
        F_ten = torch.tensor(Freq, requires_grad=True)

        # compute loss wrt. target class
        self.model.train()
        logits = self.model(F_ten)
        loss = F.cross_entropy(logits, self.target_tensor)
        loss.backward()

        # compute grad on V
        dF = np.squeeze(F_ten.grad.detach().cpu().numpy()) # shape (28, 15, 2 (real+imag))
        dF = dF[..., 0] + (1j)*dF[..., 1]
        dF = dF[:, :-1] # shape (28, 14)
        dF = np.expand_dims(dF, axis=-1) # shape (28, 14, 1)
        self.dV = simplex_ft_bw_gpu(dF, self.V, self.E, self.D, (self.res, self.res), t=(1,1), j=2)
        


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='MNIST shape optimization')
    parser.add_argument('-l', '--logdir', type=str, default="logs/log-10-19-2018_14-33-56/", help='log directory path')
    parser.add_argument('-s', '--step_size', type=float, default=1e-3, help='step size for shape optimization')
    parser.add_argument('--no_cuda', action='store_true', help='do not use cuda')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load the model
    loader = importlib.machinery.SourceFileLoader('LeNet5', os.path.join(args.logdir, "model.py"))
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)

    model = mod.LeNet5(mean=MEAN, std=STD)
    if use_cuda:
        model = nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.logdir, "checkpoint_final.pth.tar"))['state_dict'])

    # Shape Optimization
    pmnist = PolyMNIST("data/polyMNIST")
    V, E, label = pmnist[0]
    print("Original Label: {}".format(label))
    optim = ShapeOptimizer(V, E, model, target_cls=1, device=device)

    from pdb import set_trace; set_trace()
    optim._get_grad()


if __name__ == '__main__':
    main()