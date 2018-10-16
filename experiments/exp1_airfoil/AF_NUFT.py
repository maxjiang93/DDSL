#!/usr/bin/env python
# coding: utf-8

# Import packages
import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib import path
#from matplotlib import patches
import os
import pandas as pd
import csv
from transform import *
#from scipy.ndimage.interpolation import rotate
import time

def plotAirfoil(af, name):
    fig=plt.figure(figsize=(10, 5))
    ax1=fig.add_subplot(111)
    im1=ax1.imshow(np.fft.irfft2(af), origin='lower')
    fig.colorbar(im1)
    ax1.set_title(name)
    return fig

def convertAirfoil(af_name, aoa, r=128, T=1, save_name=None):
    assert isinstance(af_name, str)

    # Set dtypes
    np_int_type=np.int32;
    np_float_type=np.float32;
    np_complex_type=np.complex64;

    # Get airfoil
    af=[]
    with open('../../data/airfoil_cnn/'+af_name+'/seligdatfile', 'r') as f:
        next(f)
        d=f.readlines()
        for i in d:
            k=i.rstrip().split(' ')
            k=[float(x) for x in k if x!='']
            af.append(k)
    V=np.array(af).astype(np_float_type)


    # Construct E
    E=[]
    for v in range(V.shape[0]):
        a=v
        if v+1==V.shape[0]:
            b=0
        else:
            b=v+1
        E.append([a,b])
    E=np.array(E).astype(np_int_type)

    # Construct D
    D = np.ones([E.shape[0], 1])

    # Rotate airfoil
    orig_aoa=aoa
    aoa=-aoa*np.pi/180
    V=np.dot(V, np.array([[np.cos(aoa),np.sin(aoa)],[-np.sin(aoa),np.cos(aoa)]]))
    centroid=np.array([(np.max(V[:,0])-np.min(V[:,0]))/2,np.sign(aoa)*(np.max(V[:,1])-np.min(V[:,1]))/2])

    # Position airfoil in center
    offset=np.array([0.5,np.sign(aoa)*0.5])-centroid
    V=V+offset

    # NUFT of airfoil
    F = simplex_ft_cpu(V, E, D, res=(r, r), t=(T, T), j=2, mode='density')

    # Squeeze F (dump last dimension)
    F = np.squeeze(F).astype(np_complex_type)

    # Save parameterized airfoil matrix
    if save_name==None:
        aoa_str=str(orig_aoa).replace('.','_')
        aoa_str='aoa_p_'+aoa_str
        aoa_str=aoa_str.replace('aoa_p_-', 'aoa_n_')
        save_file='../../data/airfoil_cnn/processed-data/'+af_name+'_'+aoa_str+'.npy'
    else:
        save_file='../../data/airfoil_cnn/processed-data/'+save_name
    np.save(save_file,F)

    return F
