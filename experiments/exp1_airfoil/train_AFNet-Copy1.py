from AFNet import *
from AFNet_dataset import *

from torch.autograd import Variable
import torch.optim as optim

from tensorboardX import SummaryWriter
from shutil import rmtree

# Define mean squared error function
def MSE(predicted, labels):
    return torch.mean((predicted-labels)**2, dim=0)

# Set random seed
torch.manual_seed(1)

# Load data
print('\nLoading data...')

csv_file='processed-data/airfoil_data.csv'
shape_dir='processed-data'

trainset=AirfoilDataset(csv_file=csv_file, shape_dir=shape_dir, set_type='train')
testset=AirfoilDataset(csv_file=csv_file, shape_dir=shape_dir, set_type='test')
validset=AirfoilDataset(csv_file=csv_file, shape_dir=shape_dir, set_type='valid')

print('\nTraining on '+str(len(trainset))+' data points.')
print('Validating on '+str(len(validset))+' data points.\n')

# Create data loaders
trainloader=DataLoader(trainset, batch_size=32, num_workers=6, shuffle=True)
validloader=DataLoader(validset, batch_size=32, num_workers=6, shuffle=False)

# Start Tensorboard
logdir='airfoil_net_log'
if os.path.exists(logdir):
    rmtree(logdir)
writer = SummaryWriter(logdir)

print('Tensorboard initiated.')

# Instantiate neural network
net = AFNet()

# Mean squared error loss
criterion=nn.MSELoss()

# Adam
optimizer=optim.Adam(net.parameters(), lr=1e-4)

# Use multiple GPUs 
if(torch.cuda.device_count()>1):
    print('\nUsing', torch.cuda.device_count(), 'GPUs...')
    net=nn.DataParallel(net)
    
# Transfer neural network to GPU
if torch.cuda.is_available():
    print('Cuda is available!')
    net.cuda()
else:
    print('Cuda is not available.')

# from pdb import set_trace; set_trace()
    
print('\nTraining neural network...')

# Train
for epoch in range(100):
    running_loss=0
    valid_running_loss=0
    acc_valid=torch.zeros(2).float().cuda()
    acc_train=torch.zeros(2).float().cuda()
    for i, data in enumerate(validloader, 0):
        # Get inputs
        shapes=data['shape']
        shapes=torch.Tensor(shapes)
        shapes=shapes.float().cuda()
        cfd_data=data['Re'].view(-1, 1).float().cuda()

        # Get labels
        labels=torch.cat((data['Cl'].view(-1, 1), data['Cd'].view(-1, 1)), 1).float()
        labels=labels.cuda()

        # Forward
        outputs=net(shapes, cfd_data)
        loss=criterion(outputs, labels)

        # Running loss
        valid_running_loss+=loss.data.item()
        
        # Accuracy
        acc_valid+=MSE(outputs.data, labels.data)

    for i, data in enumerate(trainloader, 0):
        # Get inputs
        shapes=data['shape']
        shapes=torch.Tensor(shapes)
        shapes=shapes.float().cuda()
        cfd_data=data['Re'].view(-1, 1).float().cuda()

        # Get labels
        labels=torch.cat((data['Cl'].view(-1, 1), data['Cd'].view(-1, 1)), 1).float()
        labels=labels.cuda()

        # Forward
        outputs=net(shapes, cfd_data)
        loss=criterion(outputs, labels)

        # Backward
        loss.backward()

        # Optimize
        optimizer.step()

        # Accuracy
        acc_train+=MSE(outputs.data, labels.data)

        # Running loss
        running_loss+=loss.data.item()

    # Write statistics to Tensorboard
    avg_acc_valid=acc_train/len(validloader)
    avg_acc_train=acc_train/len(trainloader)
    writer.add_scalars('data/accuracy', {'valid_cl_acc': avg_acc_valid[0],\
                                        'valid_cd_acc': avg_acc_valid[1],\
                                        'train_cl_acc': avg_acc_train[0],\
                                        'train_cd_acc': avg_acc_train[1]},\
                       epoch)
    writer.add_scalars('data/loss', {'loss':running_loss/len(trainloader),\
                                     'valid_loss':valid_running_loss/len(validloader)},\
                       epoch)

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model': net.state_dict(),
        'optim':optimizer.state_dict()
    }, 'AFNet_model.checkpoint')
    
    # Zero statistics
    acc_valid=torch.zeros(2).float().cuda()
    acc_train=torch.zeros(2).float().cuda()
    running_loss = 0.0
    valid_running_loss = 0.0
    
    if epoch%5==0:
        print('Epoch '+str(epoch)+' complete!')

print('\nFinished training!')
                         

# Save model
torch.save({
    'epoch': epoch,
    'model': net.state_dict(),
    'optim':optimizer.state_dict()
}, 'AFNet_model')
