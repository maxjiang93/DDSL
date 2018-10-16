from AFNet_dataset import *
import torch
import matplotlib.pyplot as plt

# Create parameterized airfoil plotter
def plotAirfoil(af, name):
    fig=plt.figure(figsize=(10, 5))
    ax1=fig.add_subplot(111)
    af=torch.irfft(torch.Tensor(af), 2)
    im1=ax1.imshow(af, origin='lower')
    fig.colorbar(im1)
    ax1.set_title(name)
    return fig

# Load data
csv_file='processed-data/airfoil_data.csv'
shape_dir='processed-data'

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
idx=69693
af=trainset[idx]['shape']
name=trainset[idx]['name']
aoa=trainset[idx]['aoa']
savefile='test.png'
plotAirfoil(af, name).savefig(savefile)
print()
print(name+' at aoa='+str(aoa)+' plotted and saved to '+savefile)
    
