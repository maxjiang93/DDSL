#!/usr/bin/env python
# coding: utf-8

# Import packages
import sys
sys.path.append("../../ddsl/")

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from ddsl import *
from loader import poly2ve

def plot_airfoil(f, name):
    fig=plt.figure(figsize=(10, 5))
    ax1=fig.add_subplot(111)
    im1=ax1.imshow(torch.squeeze(f).detach().cpu().numpy(), origin='lower')
    fig.colorbar(im1)
    ax1.set_title(name)
    return fig

def construct_VED(airfoil, device):
    assert isinstance(airfoil, str)
   
    dev = torch.device(device)
    
    eps=1e-4;

    # Set dtypes
    int_type=torch.int64;
    float_type=torch.float64;
    complex_type=torch.complex64;

    # Get airfoil
    V=[]
    with open('data/'+airfoil+'/seligdatfile', 'r') as f:
        next(f)
        d=f.readlines()
        for i in d:
            k=i.rstrip().split(' ')
            k=[float(x) for x in k if x!='']
            V.append(k)
    
    V=torch.tensor(V, dtype=float_type)
    V += eps * torch.rand_like(V) # add noise to avoid divide by zero
    
    # Position airfoil in center
    V=0.9*V
    centroid=torch.tensor([(torch.max(V[:,0])-torch.min(V[:,0]))/2,(torch.max(V[:,1])-torch.min(V[:,1]))/2], dtype=float_type)
    offset=torch.tensor([0.5,0.5], dtype=float_type)-centroid
    V=V+offset

    # Construct E
    E=[]
    for v in range(V.shape[0]):
        a=v
        if v+1==V.shape[0]:
            b=0
        else:
            b=v+1
        E.append([a,b])
    E=torch.LongTensor(E)
    
    # Construct D
    D = torch.ones(E.shape[0], 1, dtype=V.dtype)
    
    V, E, D = V.to(dev), E.to(dev), D.to(dev)
    V.requires_grad = True
    
    return V, E, D

def airfoil_spec(airfoil, res, t=(1,1), save_name=None, device='cuda'):
    # Construct V, E, and D
    V, E, D = construct_VED(airfoil, device)
    
    # NUFT of airfoil
    ddsl_spec=DDSL_spec(res,t,2,1)
    ddsl_spec=nn.DataParallel(ddsl_spec)
    F = ddsl_spec(V,E,D)

    # Save spectral image of airfoil, if save name provided
    if save_name!=None:
        torch.save(F,save_name)

    return F

def airfoil_phys(airfoil, res, t=(1,1), save_name=None, device='cuda'):
    # Construct V, E, and D
    V, E, D = construct_VED(airfoil, device)
    
    # NUFT of airfoil + irfft
    ddsl_phys=DDSL_phys(res,t,2,1)
    ddsl_phys=nn.DataParallel(ddsl_phys)
    f = ddsl_phys(V,E,D)

    # Save physical image of airfoil, if save name provided
    if save_name!=None:
        torch.save(f,save_name)

    return f

def format_F():
    if self.filter is not None:
        self.filter = self.filter.to(F.device)
        F *= self.filter # [dim0, dim1, dim2, n_channel, 2]
    dim = len(self.res)
    F = F.permute(*([dim] + list(range(dim)) + [dim+1])) # [n_channel, dim0, dim1, dim2, 2]
