from airfoil_dataset import *
from airfoil_NUFT import *
import torch
import matplotlib.pyplot as plt

# Load data
csv_file='processed_data/airfoil_data_normalized.csv'
shape_dir='processed_data'

trainset=AirfoilDataset(csv_file=csv_file, shape_dir=shape_dir, set_type='train')
testset=AirfoilDataset(csv_file=csv_file, shape_dir=shape_dir, set_type='test')
validset=AirfoilDataset(csv_file=csv_file, shape_dir=shape_dir, set_type='valid')

# Check total number of data
print('Training set contains: ', len(trainset))
print('Validation set contains: ', len(validset))
print('Test set contains: ', len(testset))
print('Total amount of data: ', len(trainset)+len(validset)+len(testset))

# List some airfoils from the datasets
print()
print('Some training set airfoils:')
for i in range(0, 5):
    print(trainset[i]['name'])
print()
print('Some validation set airfoils:')
for i in range(0, 5):
    print(validset[i]['name'])
print()
print('Some test set airfoils:')
for i in range(0, 5):
    print(testset[i]['name'])
    
# Plot an airfoil in training set
idx=103
f=trainset[idx]['shape']
name=trainset[idx]['name']
savefile='test.png'
plot_airfoil(f, name).savefig(savefile)
print()
print(name+' plotted and saved to '+savefile)
    
