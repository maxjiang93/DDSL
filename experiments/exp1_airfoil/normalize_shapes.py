from AFNet_dataset import *
import os
import torch
import numpy as np

def calc_stats(m,oldmean,oldstd,shapes):
    n=shapes.shape[0]
    
    newmean=torch.mean(shapes)
    newstd=torch.std(shapes)
    
    wold=m/(m+n)
    wnew=n/(m+n)
    
    mean=wold*oldmean+wnew*newmean
    std=torch.sqrt(wold*oldstd**2 + wnew*newstd**2 + m*n/(m+n)**2*(oldmean-newmean)**2)
    
    m+=n
    
    return m,mean.item(),std.item()

torch.manual_seed(1)
    
batch_size=1024

trainset = AirfoilDataset(csv_file='processed_data/airfoil_data_normalized.csv', shape_dir='processed_data', set_type='train')
trainloader= DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)

directory='data/processed_data'  
m=0
mean=0
std=0
for i,data in enumerate(trainloader, 0):
    shapes=data['shape']
    shapes=torch.cat((shapes,torch.zeros(shapes.shape[0], shapes.shape[1], 1, shapes.shape[3])), dim=2)
    shapes=torch.irfft(shapes, 2, signal_sizes=[224,224])
#     shapes=shapes.view(shapes.shape[0]*224*224,1)
    m,mean,std=calc_stats(m,mean,std,shapes)
    
    if i%50==0:
        print('Batch {}: mean={:.8f}, std={:.8f}'.format(i,mean,std)) 
        
print('Batch {}: mean={:.8f}, std={:.8f}'.format(i,mean,std)) 
savefile='processed_data/shape_stats.npy'
np.save(savefile,np.array([mean,std]))
print('Mean and standard deviation saved to '+savefile)