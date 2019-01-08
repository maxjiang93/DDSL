from AFNet_dataset import *
import matplotlib.pyplot as plt
import numpy as np
import torch

trainset = AirfoilDataset(csv_file='data/processed-data/airfoil_data.csv', shape_dir='data/processed-data', set_type='train')

stats=np.load('data/processed-data/shape_stats.npy')
mean=stats[0]
std=stats[1]
print(mean,std)

smean=0
for i in range(1,100):
    shape=trainset[i]['shape']
    shape=torch.Tensor(shape)
    shape=torch.cat((shape,torch.zeros(shape.shape[0], 1, shape.shape[2])), dim=1)
    shape=torch.irfft(shape, 2, signal_sizes=[224,224])
    shape=(shape-mean)/std
    
    smean+=torch.mean(shape).item()/100
    print(torch.std(shape).item())
print(smean)
    
    


