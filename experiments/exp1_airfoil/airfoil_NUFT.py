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

def plot_airfoil(f, name):
    '''
    Plots rasterized airfoil image and returns figure.
    :param f: physical image of airfoil after DDSL transformation
    :param name: name of airfoil
    '''
    fig=plt.figure(figsize=(10, 5))
    ax1=fig.add_subplot(111)
    im1=ax1.imshow(torch.squeeze(f).detach().cpu().numpy(), origin='lower', cmap='gray')
    fig.colorbar(im1)
    ax1.set_title(name)
    return fig

def construct_VED(airfoil, device, grad):
    '''
    Constructs V, E, and D tensors for DDSL.
    :param airfoil: name of airfoil
    :param device: name of device to be used (e.g., 'cuda', 'cpu')
    :param grad: sets requires_grad for V to True or False
    '''
    assert isinstance(airfoil, str)
    
    torch.manual_seed(1)
   
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
    V.requires_grad = grad
    
    return V, E, D

def airfoil_spec(airfoil, res, t=(1,1), save_name=None, device='cuda', grad=True):
    '''
    Performs DDSL transformation on airfoil polygon into the spectral domain.
    :param airfoil: name of airfoil.
    :param res: n_dims int tuple of number of frequency modes
    :param t: n_dims tuple of period in each dimension
    :param device: name of device to be used (e.g., 'cuda', 'cpu')
    :param grad: sets requires_grad for V to True or False
    '''
    # Construct V, E, and D
    V, E, D = construct_VED(airfoil, device, grad)
    
    # NUFT of airfoil
    ddsl_spec=DDSL_spec(res,t,2,1)
    F = ddsl_spec(V,E,D)

    # Save spectral image of airfoil, if save name provided
    if save_name!=None:
        torch.save(F.cpu(),save_name)

    return F

def airfoil_phys(airfoil, res, t=(1,1), save_name=None, device='cuda', grad=True):
    '''
    Performs DDSL transformation on airfoil polygon into the physical domain.
    :param airfoil: name of airfoil.
    :param res: n_dims int tuple of number of frequency modes
    :param t: n_dims tuple of period in each dimension
    :param device: name of device to be used (e.g., 'cuda', 'cpu')
    :param grad: sets requires_grad for V to True or False
    '''
    # Construct V, E, and D
    V, E, D = construct_VED(airfoil, device, grad)
    
    # NUFT of airfoil + irfft
    ddsl_phys=DDSL_phys(res,t,2,1)
    f = ddsl_phys(V,E,D)

    # Save physical image of airfoil, if save name provided
    if save_name!=None:
        torch.save(f.cpu(),save_name)

    return f